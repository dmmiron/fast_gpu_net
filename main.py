#!/usr/bin/env python
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
#from pycuda.autoinit import context

model_file_name = 'berlin.pkl'
test_weights = np.ones((5, 4))
test_outputs = np.zeros((5, 1))

def main():
    
    image = np.float32((np.random.rand(1024, 1024) - .5) * 100)
    #image = np.float32(np.reshape(np.arange(1, 4*4+1, 1), [4, 4]))
    model = serial.load(model_file_name)
    layers = model.layers
    patch_dims = (39, 39)
    #patch_dims = (2, 2)
    batchsize = 4 
    pixels = [(x, y) for x in range(2) for y in range(2)]
    s_output = serial_computation(image, patch_dims, batchsize, layers, pixels)
    output = gpu_computation(image, patch_dims, batchsize, layers, pixels)
    output = output.get()
    print output, s_output
    #print output-s_output
    print np.allclose(s_output, output, rtol=1e-03, atol=1e-06)
    #print np.allclose(output[:500], s_output)
    #print output[:500], s_output
    
    #print s_output-output[:500]
    return 

def load():
    return

def compute_sgemm(weights, values, biases, handle, m, k, n):
    alpha = np.float32(1.0); beta = np.float32(1.0);
    #to do C = A*B + C, we actually do C_t = B_t*A_t + C_t and then transpose, but the transposing is all done implicitly in copy to and from gpu, so we just note that we do BA not AB
    print m, k, n
    print values.shape, weights.shape, biases.shape
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, values.ptr, n, weights.ptr, k, beta, biases.ptr, n)

def serial_computation(image, patch_dims, batchsize, layers, pixels):
    patchsize = patch_dims[0]*patch_dims[1]
    weights_l = []; biases_l = []; results = []; 
    values = np.float32(np.zeros((patchsize, batchsize)))
    for pixn, pixel in zip(range(batchsize), pixels):
        values[:, pixn] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
    for layer in layers:
        weights_l.append(np.transpose(layer.get_weights()))
        biases = layer.b.get_value()
        biases_l.append(biases)
        result = np.float32(np.zeros((biases.shape[0], batchsize)))
        results.append(result)


    for layer_n, (weights, biases, result) in enumerate(zip(weights_l, biases_l, results)): 
        result = np.float32(np.zeros((biases.shape[0], batchsize)))
        for pixn in range(batchsize):
            result[:, pixn] = np.dot(weights, values[:, pixn]) + biases
        values = result
            
    return result


    

def gpu_computation(image, patch_dims, batchsize, layers, pixels):
    handle = cublas.cublasCreate() 
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]
    
    weights_l = []; biases_l = []; outputs_l = [];
    for layer in layers:
        weights = layer.get_weights(); biases = np.float32(layer.b.get_value());
        weights = np.ascontiguousarray(np.transpose(weights))
        #to prevent tiling from prepending the dimension
        biases = biases.reshape([len(biases), 1])
        batch_biases = np.tile(biases, (1, batchsize))

        #we use the same weights for all of the pixels
        weights_d = gpu.to_gpu(np.float32(weights))
        batch_biases_d = gpu.to_gpu(np.float32(batch_biases))
        weights_l.append(weights_d); biases_l.append(batch_biases_d)
        
        #scratch space to copy biases to for each layer
        outputs = gpu.empty((len(biases), batchsize), np.float32)
        outputs_l.append(outputs)

    values = np.float32(np.zeros((patchsize, batchsize)))
    for pixn, pixel in zip(range(batchsize), pixels):
        start = pixn*batchsize
        values[:, pixn] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
    
    values_d = gpu.to_gpu(np.float32(values))
    inputs = values_d
    
    #print weights_l, biases_l, outputs_l
    for layern, (weights, biases, outputs) in enumerate(zip(weights_l, biases_l, outputs_l)):
        #4 for size of np.float32
        cu.memcpy_dtod(outputs.ptr, biases.ptr, outputs.size*4)
        compute_sgemm(weights, inputs, outputs, handle, weights.shape[0], weights.shape[1], batchsize) 
        inputs = outputs

    cublas.cublasDestroy(handle);
    return outputs_l[-1] 

if __name__ == "__main__":
    main()
