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

def main():
    
    image = np.float32((np.random.rand(1024, 1024) - .5) * 100)
    #image = np.float32(np.reshape(np.arange(1, 4*4+1, 1), [4, 4]))
    model = serial.load(model_file_name)
    layers = model.layers
    patch_dims_l = [(39, 39)]*5
    #patch_dims_l = [(x,y) for x in range(20, 40) for y in range(20, 40)]
    batchsize = 1
    pixels = [(x, y) for x in range(1) for y in range(1)]
    #s_output = serial_computation(image, patch_dims, batchsize, layers, pixels)
    for patch_dims in patch_dims_l:
        print patch_dims
        output = gpu_computation(image, patch_dims, batchsize, layers, pixels)
    sys.exit()
    output = output.get()
    print output, s_output
    print output-s_output
    #print output, s_output
    #print s_output-output
    print np.allclose(s_output, output)
    #print np.allclose(output[:500], s_output)
    #print output[:500], s_output
    
    #print s_output-output[:500]
    return 

def load():
    return

def compute_sgemv_batched(weights, values, biases, handle, m, k):
    batchsize = len(weights)
    batchsize = 1
    alpha = np.float32(1.0); beta = np.float32(1.0);
    #to do C = A*B + C, we actually do C_t = B_t*A_t + C_t and then transpose, but the transposing is all done implicitly in copy to and from gpu, so we just note that we do BA not AB
    print m, k
    cublas.cublasSgemmBatched(handle, 'n', k, m, alpha, values.ptr, n, weights.ptr, k, beta, biases.ptr, n, batchsize)

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

    for patch in values:
        units = patch
        #print units
        for layer_n, (weights, biases) in enumerate(zip(weights_l, biases_l)): 
            #weights = np.zeros((5, len(patch)))
            #weights[0][1] = 1
            #weights = np.float32(np.reshape(np.arange(0, 5*patch_dims[0]*patch_dims[1], 1), [5, patch_dims[0]*patch_dims[1]]))
            #print weights
            result = np.dot(weights, units)
            #+ biases
            #in here for testing first layer only
            return result
    return units


    

def gpu_computation(image, patch_dims, batchsize, layers, pixels):
    handle = cublas.cublasCreate() 
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]

    values_ps = []; bias_ps = []; output_ps = [];

    weights = layers[0].get_weights(); biases = np.float32(layers[0].b.get_value());
    #weights = np.float32(np.random.rand(500, patch_dims[0]*patch_dims[1]))
    #weights = np.transpose(weights)
    #print weights.flags, weights.shape, mod_weights.flags, mod_weights.shape
    #print mod_weights 
    weights = np.asfortranarray(weights, dtype = np.float32)
    #print weights

    weights_d = gpu.to_gpu(weights)
    #we use the same weights for all of the pixels
    weights_ps = [[weights_d.ptr]*batchsize]
    weights_ps_d = gpu.to_gpu(np.array(weights_ps))
    
    output = np.float32(np.zeros(len(biases)*batchsize))
    output = np.float32(np.zeros(500))
    output_d = gpu.to_gpu(output)
    #4 for size of float
    output_ps = [output_d.ptr + len(biases)*idx*4 for idx in range(batchsize)]
    output_ps_d = gpu.to_gpu(np.array(output_ps))

    #print weights.shape
    batches = [];
    for i in range(len(pixels)/batchsize):
        start = i*batchsize
        values = np.zeros(patchsize*batchsize) 
        for pixn, pixel in enumerate(pixels[start:start+batchsize]):
            #print type(image_d[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel())
            values[pixn*patchsize:(pixn+1)*patchsize] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
        batches.append(values)
    
    values_d = gpu.to_gpu(np.float32(batches[0]))
    values_ps = [values_d.ptr + patchsize*4*idx for idx in range(batchsize)]
    values_ps_d = gpu.to_gpu(np.array(values_ps));
    #print weights_d, weights_d.shape

    #weights get transposed
    n = weights.shape[0]; m = weights.shape[1];
    
    print m, n
    cublas.cublasSgemv(handle, 't', n, m, 1.0, weights_d.ptr, n, values_d.ptr, 1, 1.0, output_d.ptr, 1)
    s_output = np.dot(np.transpose(weights), values)
    #print output_d.get()
    #print output_d.get()-s_output 
    print np.allclose(output_d.get(), s_output, rtol=1e-04, atol=1e-07)
    return output_d

    print weights_ps_d, values_ps_d, output_ps_d
    #print output_d.dtype, values_d.dtype, weights_d.dtype, 
    #context.synchronize()
    compute_sgemm_batched(weights_ps_d[0], values_ps_d, output_ps_d, handle, weights.shape[0], weights.shape[1], 1) 
    #print weights.shape, biases.shape
    #print output_d, output_d.shape, output_d.dtype

    cublas.cublasDestroy(handle);
    return output_d, batches[0]

if __name__ == "__main__":
    main()
