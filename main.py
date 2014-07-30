from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import time
import mahotas
import cPickle
import numpy as np
import theano
import theano.tensor as T

import scikits.cuda.cublas as cublas
import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

model_file_name = 'berlin.pkl'

def main():
    
    image = np.float32((np.random.rand(1024, 1024) - .5) * 2)
    model = serial.load(model_file_name)
    layers = model.layers
    patch_dims = (39, 39)
    batchsize = 4
    output = gpu_computation(image, patch_dims, batchsize, layers)
    return 

def load():
    return

def compute_sgemm_batched(weights, values, biases, handle, m, k, n):
    batchsize = len(weights)
    alpha = np.float32(1.0); beta = np.float32(1.0);
    #to do C = A*B + C, we actually do C_t = B_t*A_t + C_t and then transpose, but the transposing is all done implicitly in copy to and from gpu, so we just note that we do BA not AB
    print n, m, k
    cublas.cublasSgemmBatched(handle, 'n', 'n', n, m, k, alpha, values.ptr, n, weights.ptr, k, beta, biases.ptr, n, batchsize)

def gpu_computation(image, patch_dims, batchsize, layers):
    handle = cublas.cublasCreate() 
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]

    values_ps = []; bias_ps = []; output_ps = [];

    weights = layers[0].get_weights(); biases = layers[0].b.get_value();
    weights = np.transpose(weights)
    weights_d = gpu.to_gpu(weights)
    #we use the same weights for all of the pixels
    weights_ps = [[weights_d.ptr]*batchsize]
    weights_ps_d = gpu.to_gpu(np.array(weights_ps))
    
    output = np.float32(np.zeros(len(biases)*batchsize))
    output_d = gpu.to_gpu(output)
    #4 for size of float
    output_ps = [output_d.ptr + len(biases)*idx*4 for idx in range(batchsize)]
    output_ps_d = gpu.to_gpu(np.array(output_ps))

    print weights.shape
    pixels = [(x, y) for x in range(2) for y in range(2)]
    batches = [];
    for i in range(len(pixels)/batchsize):
        start = i*batchsize
        values = np.zeros(patchsize*batchsize) 
        for pixn, pixel in enumerate(pixels[start:start+batchsize]):
            #print type(image_d[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel())
            values[pixn*patchsize:(pixn+1)*patchsize] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
        batches.append(values)
    
    values_d = gpu.to_gpu(batches[0])
    values_ps = [values_d.ptr + patchsize*4*idx for idx in range(batchsize)]
    values_ps_d = gpu.to_gpu(np.array(values_ps));
    
    print weights_ps_d, values_ps_d, output_ps_d
    compute_sgemm_batched(weights_ps_d[0], values_ps_d, output_ps_d, handle, weights.shape[0], weights.shape[1], 1) 
    print weights.shape, biases.shape
    print output_d, output_d.shape

    cublas.cublasDestroy(handle);
    return

if __name__ == "__main__":
    main()
