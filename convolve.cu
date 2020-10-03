#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

void checked_call(std::size_t line, cudaError_t call_result) {
  if (call_result != cudaSuccess) {
    std::cerr << "call failure at line " << line << ' '
              << cudaGetErrorString(call_result) << '\n';
    std::exit(-1);
  }
}

std::vector<float> gen_matrix(std::size_t size) {
  std::vector<float> values(size);
  static std::minstd_rand twister;
  std::uniform_real_distribution<float> dist(-1.0, 1.0);

  std::generate(values.begin(), values.end(),
                [&dist]() { return dist(twister); });
  return values;
}

std::vector<float> gen_mean_kernel() {
  float value = 1.0f / 9.0f;
  return std::vector<float>(9, value);
}

__constant__ float kernel[3][3];

__device__ inline std::size_t to_flat_index(std::size_t rowidx,
                                            std::size_t columnidx,
                                            std::size_t column_count) {
  return rowidx * column_count + columnidx;
}

__global__ void convolve(float *matrix, float *destination,
                         std::size_t row_count, std::size_t column_count) {
  std::size_t column = threadIdx.x + blockDim.x * blockIdx.x;
  std::size_t row = threadIdx.y + blockDim.y * blockIdx.y;
  if (column > column_count || row > row_count) {
    return;
  }

  const auto flat_index = to_flat_index(row, column, column_count);

  constexpr std::size_t kernel_size = 3;
  constexpr auto middle = kernel_size / 2;
  if (row < middle || column < middle || row > row_count - middle - 1 ||
      column > column_count - middle - 1) {
    destination[flat_index] = matrix[flat_index];
    return;
  }

  float sum = 0;
#pragma unroll
  for (int kernel_row_idx = -static_cast<int>(middle);
       kernel_row_idx <= static_cast<int>(middle); ++kernel_row_idx) {
    for (int kernel_column_idx = -static_cast<int>(middle);
         kernel_column_idx <= static_cast<int>(middle); ++kernel_column_idx) {
      const auto matrix_flat_index = to_flat_index(
          row + kernel_row_idx, column + kernel_column_idx, column_count);
      sum += matrix[matrix_flat_index] *
             kernel[kernel_row_idx + middle][kernel_column_idx + middle];
    }
  }

  destination[flat_index] = sum;
}

#include <chrono>
#include <thread>

namespace shino {
template <typename Clock = std::chrono::high_resolution_clock> class stopwatch {
  const typename Clock::time_point start_point;

public:
  stopwatch() : start_point(Clock::now()) {}

  template <typename Rep = typename Clock::duration::rep,
            typename Units = typename Clock::duration>
  Rep elapsed_time() const {
    std::atomic_thread_fence(std::memory_order_relaxed);
    auto counted_time =
        std::chrono::duration_cast<Units>(Clock::now() - start_point).count();
    std::atomic_thread_fence(std::memory_order_relaxed);
    return static_cast<Rep>(counted_time);
  }
};

using precise_stopwatch = stopwatch<>;
using system_stopwatch = stopwatch<std::chrono::system_clock>;
using monotonic_stopwatch = stopwatch<std::chrono::steady_clock>;
} // namespace shino

int main() {
  std::size_t sz = 16 * 1024;
  shino::precise_stopwatch generation_watch;
  auto mat = gen_matrix(sz * sz);
  std::cout << generation_watch
                   .elapsed_time<unsigned int, std::chrono::milliseconds>()
            << " milliseconds for matrix generation\n";
  shino::precise_stopwatch cuda_setup_watch;
  float *device_mat = nullptr;
  checked_call(__LINE__,
               cudaMalloc(&device_mat, mat.size() * sizeof(*mat.data())));
  checked_call(__LINE__, cudaMemcpy(device_mat, mat.data(),
                                    mat.size() * sizeof(*mat.data()),
                                    cudaMemcpyHostToDevice));
  float kernel_cell = 1.0f / 9.0f;
  std::cout << kernel_cell << '\n';
  std::vector<float> kernel_values(9, kernel_cell);
  checked_call(__LINE__,
               cudaMemcpyToSymbol(kernel, kernel_values.data(),
                                  kernel_values.size() * sizeof(*mat.data())));

  auto dest = std::vector<float>(sz * sz);
  float *device_dest = nullptr;
  checked_call(__LINE__,
               cudaMalloc(&device_dest, dest.size() * sizeof(*dest.data())));
  dim3 blockDim(32, 32);
  dim3 blockCount(std::ceil(sz / static_cast<double>(blockDim.x)),
                  std::ceil(sz / static_cast<double>(blockDim.y)));
  std::cout << cuda_setup_watch
                   .elapsed_time<unsigned int, std::chrono::microseconds>()
            << " microseconds for cuda setup (malloc and copy)\n";
  shino::precise_stopwatch kernel_watch;
  convolve<<<blockCount, blockDim>>>(device_mat, device_dest, sz, sz);
  cudaDeviceSynchronize();
  std::cout
      << kernel_watch.elapsed_time<unsigned int, std::chrono::microseconds>()
      << " microseconds for kernel execution\n";
  shino::precise_stopwatch copy_back_watch;
  checked_call(__LINE__, cudaMemcpy(dest.data(), device_dest,
                                    dest.size() * sizeof(*dest.data()),
                                    cudaMemcpyDeviceToHost));
  std::cout
      << copy_back_watch.elapsed_time<unsigned int, std::chrono::microseconds>()
      << " microseconds to copy the data back\n";
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::cout << mat[i * sz + j] << ' ';
    }
    std::cout << '\n';
  }

  std::cout << dest[1 * sz + 1] << '\n';
}