/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cucascade/cuda/device_copy_batch.hpp>
#include <cucascade/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>

namespace cucascade::cuda {

namespace {

#if CUDART_VERSION >= 13000
/// Copies at or below this size are latency-bound small transfers that overlap
/// well with compute; any larger copy in the batch makes the whole batch use the
/// default flag.
constexpr std::size_t overlap_with_compute_max_bytes = 128ULL << 10;

/// cudaMemcpyBatchAsync rejects the legacy and per-thread default streams.
[[nodiscard]] bool is_default_stream(cudaStream_t stream) noexcept
{
  return stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread;
}
#endif  // CUDART_VERSION >= 13000

/// Below this many copies, one call each (on the copy engines) costs little.
constexpr std::size_t kernel_min_copies = 8;

/// Whether the GPU reads @p src at the same address: pinned host memory under unified addressing.
[[nodiscard]] bool device_readable_host(void const* src) noexcept
{
  cudaPointerAttributes attr{};
  if (cudaPointerGetAttributes(&attr, src) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeHost && attr.devicePointer == src;
}

[[nodiscard]] bool device_memory(void const* dst) noexcept
{
  cudaPointerAttributes attr{};
  if (cudaPointerGetAttributes(&attr, dst) != cudaSuccess) {
    std::ignore = cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

constexpr int copy_threads = 1024;
constexpr std::size_t copy_tile = copy_threads * 16;
/// Blocks of one copy kernel: 16 of them saturate the host link on 1 MiB pieces and leave most SMs
/// to the kernels the copy overlaps.
constexpr int copy_blocks = 16;

/// Each block copies whole pieces, a tile at a time: aligned 16-byte loads of the source (pinned host
/// memory, read over the bus) into shared memory, then aligned 16-byte stores of the destination
/// assembled from the tile. Only the ragged ends of a tile are stored byte by byte.
__global__ void __launch_bounds__(copy_threads)
  copy_pieces(std::uintptr_t const* table, std::size_t n)
{
  __shared__ uint4 tile[copy_threads + 1];
  auto const* srcs  = table;
  auto const* dsts  = table + n;
  auto const* sizes = table + 2 * n;
  auto const* tb    = reinterpret_cast<unsigned char const*>(tile);
  for (std::size_t p = blockIdx.x; p < n; p += gridDim.x) {
    auto const src = srcs[p];
    auto const dst = dsts[p];
    auto const len = sizes[p];
    for (std::size_t off = 0; off < len; off += copy_tile) {
      auto const seg   = len - off < copy_tile ? len - off : copy_tile;
      auto const s0    = src + off;
      auto const s_lo  = s0 & ~std::uintptr_t{15};
      auto const words = (((s0 + seg + 15) & ~std::uintptr_t{15}) - s_lo) / 16;
      for (std::size_t w = threadIdx.x; w < words; w += copy_threads) {
        tile[w] = reinterpret_cast<uint4 const*>(s_lo)[w];
      }
      __syncthreads();
      auto const shift = s0 - s_lo;  // where the segment starts in the tile
      auto const d0    = dst + off;
      auto const d_lo  = (d0 + 15) & ~std::uintptr_t{15};
      auto const head  = d_lo - d0 < seg ? d_lo - d0 : seg;
      auto const body  = (seg - head) / 16;
      auto const tail  = head + body * 16;
      if (threadIdx.x < head) {
        reinterpret_cast<unsigned char*>(d0)[threadIdx.x] = tb[shift + threadIdx.x];
      }
      if (threadIdx.x < seg - tail) {
        reinterpret_cast<unsigned char*>(d0)[tail + threadIdx.x] = tb[shift + tail + threadIdx.x];
      }
      for (std::size_t w = threadIdx.x; w < body; w += copy_threads) {
        auto const at = shift + head + 16 * w;
        uint4 v;
        auto* vb = reinterpret_cast<unsigned char*>(&v);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
          vb[k] = tb[at + k];
        }
        reinterpret_cast<uint4*>(d_lo)[w] = v;
      }
      __syncthreads();
    }
  }
}

/// All the copies in one kernel that reads the sources over the bus: the pointer and size table
/// goes to the device in one upload (staged from pageable memory before the call returns).
cudaError_t copy_with_kernel(std::vector<void*> const& dsts,
                             std::vector<void const*> const& srcs,
                             std::vector<std::size_t> const& sizes,
                             cudaStream_t stream)
{
  auto const n = dsts.size();
  std::vector<std::uintptr_t> packed(3 * n);
  for (std::size_t i = 0; i < n; ++i) {
    packed[i]         = reinterpret_cast<std::uintptr_t>(srcs[i]);
    packed[n + i]     = reinterpret_cast<std::uintptr_t>(dsts[i]);
    packed[2 * n + i] = sizes[i];
  }
  auto const bytes      = packed.size() * sizeof(std::uintptr_t);
  std::uintptr_t* table = nullptr;
  auto status           = cudaMallocAsync(reinterpret_cast<void**>(&table), bytes, stream);
  if (status != cudaSuccess) { return status; }
  status = cudaMemcpyAsync(table, packed.data(), bytes, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) { return status; }
  // The kernel runs on a stream of the highest priority, so its blocks are scheduled ahead of the
  // compute that fills the SMs (copy engines never wait for SMs; this kernel would). Events order
  // it after the caller's stream and the caller's stream after it.
  struct priority_lane {
    cudaStream_t stream{nullptr};
    cudaEvent_t before{nullptr};
    cudaEvent_t after{nullptr};
  };
  thread_local priority_lane lane = [] {
    priority_lane l;
    int least = 0, greatest = 0;
    if (cudaDeviceGetStreamPriorityRange(&least, &greatest) != cudaSuccess ||
        cudaStreamCreateWithPriority(&l.stream, cudaStreamNonBlocking, greatest) != cudaSuccess ||
        cudaEventCreateWithFlags(&l.before, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&l.after, cudaEventDisableTiming) != cudaSuccess) {
      std::ignore = cudaGetLastError();
      l.stream = nullptr;
    }
    return l;
  }();
  auto const blocks = static_cast<unsigned>(std::min<std::size_t>(n, copy_blocks));
  if (lane.stream != nullptr && (status = cudaEventRecord(lane.before, stream)) == cudaSuccess &&
      (status = cudaStreamWaitEvent(lane.stream, lane.before)) == cudaSuccess) {
    copy_pieces<<<blocks, copy_threads, 0, lane.stream>>>(table, n);
    status = cudaGetLastError();
    if (status == cudaSuccess) { status = cudaEventRecord(lane.after, lane.stream); }
    if (status == cudaSuccess) { status = cudaStreamWaitEvent(stream, lane.after); }
  } else {
    copy_pieces<<<blocks, copy_threads, 0, stream>>>(table, n);
    status = cudaGetLastError();
  }
  auto const freed = cudaFreeAsync(table, stream);
  return status != cudaSuccess ? status : freed;
}

}  // namespace

void device_copy_batch::reserve(std::size_t n)
{
  _dsts.reserve(n);
  _srcs.reserve(n);
  _sizes.reserve(n);
}

void device_copy_batch::add(void* dst, void const* src, std::size_t bytes)
{
  if (bytes == 0 || dst == nullptr || src == nullptr) { return; }
  _dsts.push_back(dst);
  _srcs.push_back(src);
  _sizes.push_back(bytes);
  _bytes += bytes;
}

cudaError_t device_copy_batch::enqueue(::cuda::stream_ref stream) const
{
  if (_dsts.empty()) { return cudaSuccess; }

#if CUDART_VERSION >= 13000
  // cudaMemcpyBatchAsync faults intermittently under concurrent use on driver 595.58 (CUDA 13.2):
  // Xid 13 illegal instructions and Xid 31 stray writes, with or without the overlap flag, with the
  // calls serialized, and with the arrays kept alive. CUCASCADE_SAFE_COPY=1 avoids it (one small
  // copy kernel per batch, or a call per piece), at a cost when the copies are the bottleneck.
  static bool const use_batch = std::getenv("CUCASCADE_SAFE_COPY") == nullptr;
  if (use_batch && !is_default_stream(stream.get())) {
    cudaMemcpyAttributes attr{};
    attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    attr.flags          = std::ranges::all_of(
                   _sizes, [](std::size_t size) { return size <= overlap_with_compute_max_bytes; })
                            ? cudaMemcpyFlagPreferOverlapWithCompute
                            : cudaMemcpyFlagDefault;
    // Single-attribute overload (cuda_runtime.h): deduces direction from the pointer types.
    return cudaMemcpyBatchAsync(
      _dsts.data(), _srcs.data(), _sizes.data(), _dsts.size(), attr, stream.get());
  }
#endif

  // Pieces contiguous at both ends become one copy.
  std::vector<void*> dsts;
  std::vector<void const*> srcs;
  std::vector<std::size_t> sizes;
  dsts.reserve(_dsts.size());
  srcs.reserve(_dsts.size());
  sizes.reserve(_dsts.size());
  for (std::size_t i = 0; i < _dsts.size();) {
    auto* dst     = static_cast<std::byte*>(_dsts[i]);
    auto* src     = static_cast<std::byte const*>(_srcs[i]);
    std::size_t n = _sizes[i];
    std::size_t j = i + 1;
    for (; j < _dsts.size() && static_cast<std::byte*>(_dsts[j]) == dst + n &&
           static_cast<std::byte const*>(_srcs[j]) == src + n;
         ++j) {
      n += _sizes[j];
    }
    dsts.push_back(dst);
    srcs.push_back(src);
    sizes.push_back(n);
    i = j;
  }

  // Many pinned-to-device pieces: one small kernel, which the faults above do not touch and which
  // costs one launch instead of a call per piece (each call queues on the copy engines).
  if (dsts.size() >= kernel_min_copies && device_readable_host(srcs.front()) &&
      device_readable_host(srcs.back()) && device_memory(dsts.front()) &&
      device_memory(dsts.back())) {
    return copy_with_kernel(dsts, srcs, sizes, stream.get());
  }
  for (std::size_t i = 0; i < dsts.size(); ++i) {
    auto const status = cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream.get());
    if (status != cudaSuccess) { return status; }
  }
  return cudaSuccess;
}

void device_copy_batch::clear() noexcept
{
  _dsts.clear();
  _srcs.clear();
  _sizes.clear();
  _bytes = 0;
}

}  // namespace cucascade::cuda
