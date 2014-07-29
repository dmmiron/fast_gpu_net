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

__global__ void max_gpu_kernel_batched(const float *input, const int height, const int width, const int channels, const int out_height, const int out_width, const int out_channels, const int max_height, const int max_width, const int max_channels, const int batchsize, float *output) {
    int x_out_idx = blockDim.x*blockIdx.x + threadIdx.x;
    int y_out_idx = blockDim.y*blockIdx.y + threadIdx.y;
    int batch = blockDim.z*blockIdx.z + threadIdx.z;
    int input_offset = batch*height*width*channels;
    int output_offset = batch*out_height*out_width*out_channels;
    input += input_offset; output += output_offset;
    if (x_out_idx >= 0 && x_out_idx < out_width && y_out_idx >= 0 && y_out_idx < out_height) {
        float temp;
        float max;
        int start_x; int start_y; int start_z;
        int x_idx; int y_idx; int z_idx;
        start_x = x_out_idx*max_width; start_y = y_out_idx*max_height;
        for (int out_k = 0; out_k < out_channels; ++out_k) {
            start_z = out_k*max_channels;
            max = input[start_x+start_y*width + start_z*width*height];
            for (int i = 0; i < max_width; ++i) {
                for (int j = 0; j < max_height; ++j) {
                    for (int k = 0; k < max_channels; ++k) {
                        x_idx = start_x + i;
                        y_idx = start_y + j;
                        z_idx = start_z + k;
                        if (x_idx >= 0 && x_idx < width && y_idx >= 0 && y_idx < height && z_idx >= 0 && z_idx < channels) {
                            temp = input[x_idx + y_idx*width+z_idx*width*height];
                            if (temp > max)
                                max = temp;
                        }
                    }
                }
            }
            output[x_out_idx + y_out_idx*out_width + out_k*out_width*out_height] = max;
        }
    }
}


"""

def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_max_batched(in_array, max_dims):
    height = in_array.shape[2]; width = in_array.shape[3]; channels = in_array.shape[1]
    batchsize = in_array.shape[0]
    max_height = max_dims[0]; max_width = max_dims[1]; max_channels = max_dims[2]
    out_height = (height + max_height - 1) / max_height; out_width = (width + max_width -1 ) / max_width; out_channels = (channels + max_channels -1) / max_channels;
    
    num_kernels = out_height*out_width*batchsize;

    #CUDA MALLOC
    output = gpu.empty((batchsize, out_channels, out_height, out_width), np.float32);

    block_x = 16; block_y = 16;
    block = (block_x, block_y, 1);
    grid = (np.asscalar((out_height + block_x - 1)/block_x), np.asscalar((out_width + block_y - 1) / block_y), np.asscalar(batchsize));

    max_batched(in_array, np.int32(height), np.int32(width), np.int32(channels), np.int32(out_height), np.int32(out_width), np.int32(out_channels), np.int32(max_height), np.int32(max_width), np.int32(max_channels), np.int32(batchsize), output, block=block, grid=grid); 
    return output

def compute_max(in_array, max_dims, stream):
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
    grido = ((kernels+block_xo-1)/block_xo, 1, 1)

    temp = gpu.empty((p_height, p_width, p_channels), np.float32)

    #start = cu.Event()
    #end = cu.Event()
    #start.record()
    maxpool(in_array, np.int32(in_array.shape[0]), np.int32(in_array.shape[1]), np.int32(in_array.shape[2]), np.int32(max_dims[0]), p_height, p_width, temp, block=blockp, grid=gridp, stream=stream)
    #end.record()
    #end.synchronize()
    #print "maxpool took: {0:.4e} seconds".format(end.time_since(start)/1000)

    result = gpu.empty((o_height, o_width, o_channels), np.float32)
    #start.record()
    maxout(temp, p_height, p_width, p_channels, o_channels, np.int32(max_dims[2]), result, block=blocko, grid=grido, stream=stream)
    #end.record()
    #end.synchronize()
    #print "maxout took: {0:.4e} seconds".format(end.time_since(start)/1000)
    

    return result

def init():
    """compile the kernels"""
    global maxpool
    global maxout
    global max_batched
    maxpool = get_gpu_func(maxes, "maxpool_gpu_kernel")
    maxout = get_gpu_func(maxes, "maxout_gpu_kernel")
    max_batched = get_gpu_func(maxes, "max_gpu_kernel_batched")

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
    
