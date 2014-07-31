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
    batchsize = 1
    pixels = [(x, y) for x in range(1) for y in range(1)]
    s_output = serial_computation(image, patch_dims, batchsize, layers, pixels)
    output = gpu_computation(image, patch_dims, batchsize, layers, pixels)
    output = output.get()
    print output, s_output
    print output-s_output
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
    values = []; weights_l = []; biases_l = []; 
    for pixel in pixels:
        values.append(image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel())
    for layer in layers:
        weights_l.append(np.transpose(layer.get_weights()))
        biases_l.append(layer.b.get_value())
    #print weights_l, biases_l, values
    #print weights_l[0], "serial"
    #print biases_l[0], "serial biases"
    #print values[0], "serial values"
    result = np.float32(np.zeros((biases_l[0].shape[0], batchsize)))
    for pixn, patch in enumerate(values):
        units = patch
        for layer_n, (weights, biases) in enumerate(zip([weights_l[0]], [biases_l[0]])): 
            #weights = np.zeros((5, len(patch)))
            #weights[0][1] = 1
            #weights = np.float32(np.reshape(np.arange(0, 5*patch_dims[0]*patch_dims[1], 1), [5, patch_dims[0]*patch_dims[1]]))

            result[:, pixn] = np.dot(weights, units)
            #+ biases
            #in here for testing first layer only
    return result


    

def gpu_computation(image, patch_dims, batchsize, layers, pixels):
    handle = cublas.cublasCreate() 
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]

    values_ps = []; bias_ps = []; output_ps = [];

    weights = layers[0].get_weights(); biases = np.float32(layers[0].b.get_value());
    weights = np.ascontiguousarray(np.transpose(weights))

    #weights = np.float32(np.ones((5, patch_dims[0]*patch_dims[1])))
    #weights[1][0] = 0
    #weights = np.float32(test_weights)

    weights_d = gpu.to_gpu(np.float32(weights))
    #we use the same weights for all of the pixels
    
    outputs = np.float32(np.zeros((len(biases), batchsize)))
    #outputs = np.float32(test_outputs)
    #outputs = np.float32(np.zeros((5, batchsize)))
    outputs_d = gpu.to_gpu(outputs)

    values = np.float32(np.zeros((patchsize, batchsize)))
    for pixn, pixel in zip(range(batchsize), pixels):
        start = pixn*batchsize
        values[:, pixn] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
    
    values_d = gpu.to_gpu(np.float32(values))
    #print weights_d, weights_d.shape

    #print weights_d, values_d, "GPU"
    compute_sgemm(weights_d, values_d, outputs_d, handle, weights.shape[0], weights.shape[1], batchsize) 

    cublas.cublasDestroy(handle);
    return outputs_d 

if __name__ == "__main__":
    main()
