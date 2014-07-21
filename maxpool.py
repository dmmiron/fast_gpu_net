import numpy as np
import time

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

global maxpool; global maxout;
maxes = """
#include <stdio.h>
__global__ void maxpool_gpu_kernel(const float *input, const int in_height, const int in_width, const int channels, const int ksize, const int out_height, const int out_width, float *output) {
    int x_out_idx = blockIdx.x*blockDim.x + threadIdx.x;
    int y_out_idx = blockIdx.y*blockDim.y + threadIdx.y;
    int z_out_idx = blockIdx.z*blockDim.z + threadIdx.z;

    if (x_out_idx < out_width && y_out_idx < out_height && z_out_idx < channels) {

        int start_x = x_out_idx * ksize;
        int start_y = y_out_idx * ksize;

        int x_idx; int y_idx;
        int z_idx = z_out_idx;

        //float max = -FLT_MAX;
        float temp;
        float max = input[start_x+start_y*in_width+z_idx*in_width*in_height];
        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                x_idx = start_x + j;
                y_idx = start_y + i;
                //check for valid index
                if (x_idx >= 0 && x_idx < in_width && y_idx >= 0 && y_idx < in_height) {
                    temp = input[x_idx + y_idx*in_width + z_idx*in_width*in_height]; //convert from x, y, z to single idx
                    if (temp > max)
                        max = temp;
                 }
            }
        }

        output[x_out_idx + y_out_idx*out_width + z_out_idx*out_width*out_height] = max;
    }
}

__global__ void maxout_gpu_kernel(const float *input, const int height, const int width, const int channels, const int out_channels, const int ksize, float *output) {
    int out_idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (out_idx < height*width*out_channels) {
        int stride = height*width;
        int layer = out_idx / (stride);
        int offset = out_idx % (stride);
        int start = layer*ksize*stride + offset;

        float temp;
        //float max = -FLT_MAX;
        float max = input[start];
        int idx;

        for (int i = 0; i < ksize; ++i) {
            idx = start + i*stride;
            if (idx >=0 && idx < stride*channels) {
                temp = input[idx];
                if (temp > max)
                    max = temp;
            }
        } 

        output[out_idx] = max;
    }
}
"""

def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_max(in_array, max_dims):
    p_height = np.int32((in_array.shape[0]+max_dims[0]-1)/max_dims[0])
    p_width = np.int32((in_array.shape[1]+max_dims[1]-1)/max_dims[1])
    p_channels = np.int32(in_array.shape[2])
    o_height = p_height; o_width = p_width;
    o_channels = np.int32((p_channels+max_dims[2]-1)/max_dims[2])

    block_xp = 8; block_yp = 8; block_zp = 2;
    block_xo = 256; block_yo = 1; block_zo = 1;
    blockp = (block_xp, block_yp, block_zp)
    blocko = (block_xo, block_yo, block_zo)
    gridp = ((p_height + block_xp - 1)/block_xp, (p_width + block_yp - 1)/block_yp, (p_channels+block_zp - 1)/block_zp)
    kernels = o_height*o_width*o_channels
    grido = ((kernels+block_xo-1), 1, 1)

    temp = gpu.empty((p_height, p_width, p_channels), np.float32)
    st = time.time()
    maxpool(in_array, np.int32(in_array.shape[0]), np.int32(in_array.shape[1]), np.int32(in_array.shape[2]), np.int32(max_dims[0]), p_height, p_width, temp, block=blockp, grid=gridp)
    print "maxpool only, took:", time.time()-st

    result = gpu.empty((o_height, o_width, o_channels), np.float32)
    st = time.time()
    maxout(temp, p_height, p_width, p_channels, o_channels, np.int32(max_dims[2]), result, block=blocko, grid=grido)
    print "maxout only, took:", time.time()-st
    return result

def init():
    """compile the kernels"""
    global maxpool
    global maxout
    maxpool = get_gpu_func(maxes, "maxpool_gpu_kernel")
    maxout = get_gpu_func(maxes, "maxout_gpu_kernel")

if __name__ == "__main__":

    A = np.float32(np.random.rand(4, 4, 4))
    A = np.float32(np.reshape(np.arange(0, 64, 1), [4, 4, 4]))
    maxpool_dims = np.int32((2, 2, 2))
    pool_output_dims = (np.int32(np.shape(A)[0]/maxpool_dims[0]), np.int32(np.shape(A)[1]/maxpool_dims[0]), np.int32(np.shape(A)[2]))
    print A
    print pool_output_dims[0].dtype
    print pool_output_dims

    A_d = gpu.to_gpu(A)
    result = compute_max(A_d, maxpool_dims)
    print result.get()
    
