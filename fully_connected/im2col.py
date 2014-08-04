import numpy as np
import sys

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

global im2col
global im2col_batched

im2col_kernel = """
#include <stdio.h>
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < (n); \
            i += blockDim.x * gridDim.x)
__global__ void im2col_gpu_kernel(const int n, const float* data_im, const int height, const int width, const int ksize, const int pad, const int stride, const int height_col, const int width_col, const int offset, float* data_col) {
    data_im += offset;
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
/* for batching we use the y dimension to control which batch
 * if the layer is 0 then we read from the same image, but for all other layers we read from one array that is the concatenation
 * of the individual images for each batch
 */
__global__ void im2col_gpu_kernel_batched(const int npbatch, const float* data_im, const int height, const int width, const int channels, const int ksize, const int pad, const int stride, const int height_col, const int width_col, const int *offsets, const int layer_n, float* data_col) {
    int y_idx = blockIdx.y*blockDim.y + threadIdx.y; //y dimension deals with batches
    int input_offset;
    int output_offset = y_idx*height_col*width_col*channels*ksize*ksize;
    int pix_offset;
    if (layer_n == 0) {
        pix_offset = offsets[y_idx];
        input_offset = 0;
    }
    else {
        pix_offset = 0;
        input_offset = y_idx*height*width*channels;
    }
    data_im += pix_offset + input_offset;
    data_col += output_offset;
    CUDA_KERNEL_LOOP(index, npbatch) {
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

def compute_im2col_batched(in_array, window_height, window_width, window_channels, ksize, pad, stride, offsets, layer_n, batchsize, output):
    if (layer_n == 0):
        height = np.int32(in_array.shape[0])
        width = np.int32(in_array.shape[1])
    else:
        height = window_height;
        width = window_width;
    height_col = np.int32((window_height + 2 * pad - ksize) / stride + 1)
    width_col = np.int32((window_width + 2 * pad - ksize) / stride + 1)
    num_kernels = np.int32(height_col*width_col*window_channels*batchsize)
    kernels_per_batch = num_kernels/batchsize
    threads = 256 
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads*batchsize -1)/(threads*batchsize), batchsize, 1)
    
    im2col_batched(kernels_per_batch, in_array, np.int32(height), np.int32(width), np.int32(window_channels), np.int32(ksize), np.int32(pad), np.int32(stride), height_col, width_col, offsets, np.int32(layer_n), output, block=blocksize, grid=gridsize)

def compute_im2col(in_array, window_height, window_width, window_channels, ksize, pad, stride, start_idx):
    height = np.int32(in_array.shape[0]); width = np.int32(in_array.shape[1]); 
    height_col = np.int32((window_height + 2 * pad - ksize) / stride + 1)
    width_col = np.int32((window_width + 2 * pad - ksize) / stride + 1)
    num_kernels = np.int32(height_col*width_col*window_channels)
    threads = 256 
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads -1)/threads, 1, 1)
    result = gpu.empty((ksize*ksize*window_channels, height_col*width_col), np.float32)

    #start = cu.Event()
    #end = cu.Event()
    #start.record()
    #im2col(num_kernels, in_array, np.int32(height), np.int32(width), np.int32(ksize), np.int32(pad), np.int32(stride), height_col, width_col, np.int32(start_idx), result, block=blocksize, grid=gridsize)
    im2col(num_kernels, in_array, np.int32(height), np.int32(width), np.int32(ksize), np.int32(pad), np.int32(stride), height_col, width_col, np.int32(start_idx), result, block=blocksize, grid=gridsize)
    #end.record()
    #end.synchronize()
    #print "im2col took: {0:.4e} seconds".format(end.time_since(start)/1000)
    return result

def init():
    global im2col
    global im2col_batched
    im2col = get_gpu_func(im2col_kernel, "im2col_gpu_kernel")
    im2col_batched = get_gpu_func(im2col_kernel, "im2col_gpu_kernel_batched")

def test_batched():
    init()
    batchsize = 2
    ksize = 2; pad = 0; stride = 1;
    offset = [0, 0];
    offset_d = gpu.to_gpu(np.array(offset));
    A = np.float32(np.reshape(np.arange(0, 64, 1), [4, 4, 2, 2]))
    A_d = gpu.to_gpu(A)
    result = compute_im2col_batched(A_d, 4, 4, 2, ksize, pad, stride, offset_d, 1, batchsize)
    print A_d
    print result

if __name__ == "__main__":
    test_batched()
    
