import sys
import numpy as np

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

global soft_max

soft_max_kernel = """
__global__ void soft_max_gpu(const float *input, int pixels, float *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pixels) {
        float num = expf(input[idx*2]);
        float den = num + expf(input[idx*2+1]);
        output[idx] = num/den;
    }
}
"""

def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_soft_max(in_array, output):
    threads = 128;
    num_kernels = output.size
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads-1)/threads, 1, 1)
    soft_max(in_array, np.int32(num_kernels), output, block=blocksize, grid=gridsize) 
    return

def init():
    global soft_max
    soft_max = get_gpu_func(soft_max_kernel, "soft_max_gpu")

def test_soft_max():
    in_array = np.float32(np.random.rand(10, 2))
    in_array_d = gpu.to_gpu(in_array)
    output = gpu.empty(10, np.float32)
    print in_array_d, output
    compute_soft_max(in_array_d, output)
    print in_array
    print output

if __name__ == "__main__":
    init()
    test_soft_max()
