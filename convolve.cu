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
  static std::mt19937_64 twister;
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
  for (int kernel_row_idx = -static_cast<int>(middle); kernel_row_idx <= middle;
       ++kernel_row_idx) {
    for (int kernel_column_idx = -static_cast<int>(middle);
         kernel_column_idx <= middle; ++kernel_column_idx) {
      const auto matrix_flat_index = to_flat_index(
          row + kernel_row_idx, column + kernel_column_idx, column_count);
      const auto kernel_flat_index = to_flat_index(
          kernel_row_idx + middle, kernel_column_idx, kernel_size);
      sum += matrix[matrix_flat_index] *
             kernel[kernel_row_idx + middle][kernel_column_idx + middle];
    }
  }

  destination[flat_index] = sum;
}

int main() {
  std::size_t sz = 16 * 1024;
  auto mat = gen_matrix(sz * sz);
  float *device_mat = nullptr;
  checked_call(__LINE__,
               cudaMalloc(&device_mat, mat.size() * sizeof(*mat.data())));
  checked_call(__LINE__, cudaMemcpy(device_mat, mat.data(),
                                    mat.size() * sizeof(*mat.data()),
                                    cudaMemcpyHostToDevice));
  float kernel_cell = 1.0f / 9.0f;
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
  convolve<<<blockDim, blockCount>>>(device_mat, device_dest, sz, sz);
  checked_call(__LINE__, cudaMemcpy(dest.data(), device_dest,
                                    dest.size() * sizeof(*dest.data()),
                                    cudaMemcpyDeviceToHost));
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      std::cout << mat[i * sz + j] << ' ';
    }
    std::cout << '\n';
  }

  std::cout << dest[1 * sz + 1] << '\n';
}