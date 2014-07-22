import numpy as np

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

global im2col

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

def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_im2col(in_array, ksize, pad, stride):
    height = np.int32(in_array.shape[0]); width = np.int32(in_array.shape[1]); channels = np.int32(in_array.shape[2]);
    height_col = np.int32((height + 2 * pad - ksize) / stride + 1)
    width_col = np.int32((width + 2 * pad - ksize) / stride + 1)
    num_kernels = np.int32(height_col*width_col*channels)
    threads = 256 
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads -1)/threads, 1, 1)
    result = gpu.empty((ksize*ksize*channels, height_col*width_col), np.float32)

    start = cu.Event()
    end = cu.Event()
    start.record()
    im2col(num_kernels, in_array, height, width, np.int32(ksize), np.int32(pad), np.int32(stride), height_col, width_col, result, block=blocksize, grid=gridsize)
    end.record()
    end.synchronize()
    print "im2col took: {0:.4e} seconds".format(end.time_since(start)/1000)
    return result

def init():
    global im2col
    im2col = get_gpu_func(im2col_kernel, "im2col_gpu_kernel")

if __name__ == "__main__":
    A = np.float32(np.reshape(np.arange(0, 32, 1), [4, 4, 2]))
    print A
    A_d = gpu.to_gpu(A)
    result = compute_im2col(A_d, 2, 0, 1)

