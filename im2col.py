import numpy as np

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

im2col_kernel = """
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < (n); \
            i += blockDim.x * gridDim.x)
__global__ void im2col_gpu_kernel(const int n, const float* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, float* data_col) {
    CUDA_KERNEL_LOOP(index, n) {
        int w_out = index % width_col;
        index /= width_col;
        int h_out = index % height_col;
        int channel_in = index / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        data_col += (channel_out * height_col + h_out) * width_col + w_out;
        data_im += (channel_in * height + h_in) * width + w_in;
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                
                *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im[i * width + j] : 0;
                data_col += height_col * width_col;
            }
        }
    }
}
"""

sgemm_kernel = """
#include <cuda_runtime.h>
#include <cublas_v2.h>
__global__ void sgemm(const int m, const int n, const int k, const float alpha, const float beta, const float *A, const float* B, float *C)
{
    if (threadIdx.x == 0) {
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n);
    }
}
"""

def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_im2col(in_array, ksize, pad, stride):
    im2col = get_gpu_func(im2col_kernel, "im2col_gpu_kernel")
    height = np.int32(in_array.shape[0]); width = np.int32(in_array.shape[1]); channels = np.int32(in_array.shape[2]);
    height_col = np.int32((height + 2 * pad - ksize) / stride + 1)
    width_col = np.int32((width + 2 * pad - ksize) / stride + 1)
    num_kernels = np.int32(height_col*width_col*channels)
    threads = 256 
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads -1)/threads, 1, 1)
    result = gpu.empty((ksize*ksize*channels, height_col*width_col), np.float32)
    im2col(num_kernels, in_array, height, width, np.int32(ksize), np.int32(pad), np.int32(stride), height_col, width_col, result, block=blocksize, grid=gridsize)
    return result

def compute_sgemm(col, kernel, bias):
    sgemm = get_gpu_func(sgemm_kernel, "sgemm"); 
    alpha = np.float32(1.0); beta = np.float32(1.0);
    blocksize = (1, 1, 1)
    gridsize = (1, 1, 1)

    #(mxn)x(nxk)
    m = np.int32(kernel.shape[0])
    n = np.int32(kernel.shape[1]) 
    k = np.int32(col.shape[0])
    sgemm(m, n, k, alpha, beta, kernel, col, bias);
    return 

if __name__ == "__main__":
    A = np.float32(np.reshape(np.arange(0, 32, 1), [4, 4, 2]))
    print A
    A_d = gpu.to_gpu(A)
    result = compute_im2col(A_d, 2, 0, 1)

