import sys
import numpy as np

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

global rectify

rectifier_kernel = """
__global__ void rectifier_gpu(float *input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] < 0) {
            input[idx] = 0;
        }
    }
}
"""


def get_gpu_func(module, func_name):
    return nvcc.SourceModule(module).get_function(func_name)

def compute_rectify(in_array):
    threads = 128;
    num_kernels = in_array.size
    print num_kernels
    blocksize = (threads, 1, 1)
    gridsize = ((num_kernels+threads-1)/threads, 1, 1)

    rectify(in_array, np.int32(num_kernels), block=blocksize, grid=gridsize) 

def init():
    global rectify
    rectify = get_gpu_func(rectifier_kernel, "rectifier_gpu") 


def test_rectify():
    image = np.float32(np.random.rand(10, 10))-.5
    image_d = gpu.to_gpu(image)
    compute_rectify(image_d)
    print image, image_d.get()
    print image-image_d.get()
   


if __name__ == "__main__":
    init()
    test_rectify()
