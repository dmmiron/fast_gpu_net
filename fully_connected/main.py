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

import glob
import math

import scikits.cuda.cublas as cublas
import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

import rectifier as rect
import soft_max as soft_max

model_file_name = 'berlin.pkl'
#initialize kernels on import
rect.init()
soft_max.init()
 

def load_image(image_name):
    image = np.float32(mahotas.imread(image_name))
    image /= 255
    return image

def save_image(image, out_name):
    mahotas.imsave(out_name, np.int8(image*255))

def classify_image(image, model):
    layers = model.layers
    patch_dims = (39, 39)
    valid_x = image.shape[0]-patch_dims[0]
    valid_y = image.shape[1]-patch_dims[1]
    batchsize = 512
    pixels = [(x,y) for x in range(valid_x) for y in range(valid_y)]
    output = gpu_computation(image, patch_dims, batchsize, layers, pixels)
    output = output.get()
    output = output.reshape(valid_x, valid_y)
    return output

def classify(image_names, model_file_name, output_names):
    model = serial.load(model_file_name)
    outputs = []
    for image_name, output_name in zip(image_names, output_names):
        image = load_image(image_name)
        output = classify_image(image, model)
        save_image(np.int32(np.round(output)), output_name)

def main():
    
    image = np.float32((np.random.rand(1024, 1024) - .5) * 2)
    #image = np.float32(np.reshape(np.arange(1, 4*4+1, 1), [4, 4]))
    model = serial.load(model_file_name)
    layers = model.layers
    #for testing only
    #layers.pop()
    
    patch_dims = (39, 39)
    batchsizes = [2**x for x in range(7, 10)] 
    batchsizes = [128]
    pixels = [(x, y) for x in range(1024-39) for y in range(1024-39)]
    #pixels = [(x,y) for x in range(128) for y in range(128)]
    #p_output = pylearn2_computation(model, image, patch_dims, batchsizes[0], layers, pixels)
    #p_output = np.transpose(p_output)
    #save_image(np.int8(np.round(p_output[0].reshape(128, 128))), "test_pylearn2.tif")
    #s_output = serial_computation(image, patch_dims, batchsize, layers, pixels)
    for batchsize in batchsizes:
        st = time.time()
        output = gpu_computation(image, patch_dims, batchsize, layers, pixels)
        output = output.get()
        tot = time.time()-st
        print "Batchsize {0}".format(batchsize)
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel: {0:.4e} seconds".format(tot/len(pixels))
    output = output.reshape(1024-39, 1024-39)
    save_image(np.int8(np.round(output)), "test_out.tif")
    #print output, p_output
    #print output.shape, s_output.shape, p_output.shape
    
    #print np.allclose(s_output[0], output, rtol=1e-03, atol=1e-06)
    #print np.allclose(p_output[0], output, rtol=1e-04, atol=1e-07)
    #print np.allclose(p_output, s_output, rtol=1e-04, atol=1e-06)
    
    return 

def load():
    return

def compute_sgemm(weights, values, biases, handle, m, k, n):
    alpha = np.float32(1.0); beta = np.float32(1.0);
    #to do C = A*B + C, we actually do C_t = B_t*A_t + C_t and then transpose, but the transposing is all done implicitly in copy to and from gpu, so we just note that we do BA not AB
    #print m, k, n
    #print values.shape, weights.shape, biases.shape
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, values.ptr, n, weights.ptr, k, beta, biases.ptr, n)

def pylearn2_computation(model, image, patch_dims, batchsize, layers, pixels):
    patchsize = patch_dims[0]*patch_dims[1]
    nbatches = (len(pixels) + batchsize -1)/batchsize
    model.set_batch_size(batchsize) 
    data = model.get_input_space().make_batch_theano()
    outputs = np.float32(np.zeros((len(pixels), 2)))
    for batch in range(nbatches):
        start = batch*batchsize
        values = np.float32(np.zeros((batchsize, patchsize)))
        for pixn, pixel in zip(range(batchsize), pixels[start:start+batchsize]):
            values[pixn, :] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
    
        for layer in model.layers:
            y = layer.fprop(data)
            classify = theano.function([data], [y], name='classify')
            output = np.array(classify(values))
            values = output[0]
        outputs[start:start+batchsize, :] = values
    return outputs


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
        values = ser_rect(result)
            
    return result

def ser_rect(in_array):
    for x in np.nditer(in_array, op_flags=['readwrite']):
        if x < 0:
            x[...] = 0
    """for i in range(in_array.shape[0]):
        for j in range(in_array.shape[1]):
            if (in_array[i, j] < 0):
                in_array[i, j] = 0"""
    return in_array  


def softmax(in_array):

    return 

    

def gpu_computation(image, patch_dims, batchsize, layers, pixels):
    handle = cublas.cublasCreate() 
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]
    npixels = len(pixels)
    nbatches = (npixels + batchsize - 1) / batchsize
    
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

    #final output
    classes = gpu.empty(npixels, np.float32)
    values_d_l = []
    for batch in range(nbatches): 
        values = np.zeros((patchsize, batchsize), np.float32)
        start = batch*batchsize
        for pixn, pixel in zip(range(batchsize), pixels[start:start+batchsize]):
            values[:, pixn] = image[pixel[0]:pixel[0]+patch_dims[0], pixel[1]:pixel[1]+patch_dims[1]].ravel()
    
        values_d = gpu.to_gpu(np.float32(values))
        values_d_l.append(values_d)

    for (batch, values_d) in zip(range(nbatches), values_d_l):
        inputs = values_d
        #print inputs
        offset = batch*batchsize 

        #print weights_l, biases_l, outputs_l
        for layern, (weights, biases, outputs) in enumerate(zip(weights_l, biases_l, outputs_l)):
            #4 for size of np.float32
            cu.memcpy_dtod(outputs.ptr, biases.ptr, outputs.size*4)
            compute_sgemm(weights, inputs, outputs, handle, weights.shape[0], weights.shape[1], batchsize) 
            if (layern < len(layers)-1):
                rect.compute_rectify(outputs)
                inputs = outputs
            else:
                #print outputs
                soft_max.compute_soft_max(outputs, classes, offset)
                #print classes
    
    cublas.cublasDestroy(handle);
    return classes 

if __name__ == "__main__":
    main()
    sys.exit()

    if len(sys.argv) != 3:
        print "Usage: python main.py <image_folder> <output_folder> <model_file>"
        sys.exit()
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    #output_path = "/home/dmmiron/cuda/fast_gpu_net/fully_connected/"
    model_file_name = sys.argv[3]
    images = sorted(glob.glob(image_path + "/*"))
    output_names = [output_path.rstrip("/") + "/" + image_name.split("/")[-1].rstrip(".tif") + "_classified.tif" for image_name in images]
    classify(images, model_file_name, output_names)

