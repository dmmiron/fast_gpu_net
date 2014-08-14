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
import im2col as im2col

model_file_name = 'berlin.pkl'

#initialize kernels on import
rect.init()
soft_max.init()
im2col.init()

#only for timing
time_starts = []; time_ends = []; sgemm_gflop = 0;
 
def normalize_image_float(original_image, saturation_level=0.005):
    sorted_image = np.sort( np.uint8(original_image).ravel() )
    minval = np.float32( sorted_image[ len(sorted_image) * ( saturation_level / 2 ) ] )
    maxval = np.float32( sorted_image[ len(sorted_image) * ( 1 - saturation_level / 2 ) ] )
    norm_image = np.float32(original_image - minval) * ( 255 / (maxval - minval))
    norm_image[norm_image < 0] = 0
    norm_image[norm_image > 255] = 255
    return norm_image / 255.0

def load_image(image_name):
    image = np.float32(mahotas.imread(image_name))
    image = normalize_image_float(image)
    return image

def save_image(image, out_name):
    #assumes image is normalized float from 0 to 1
    print "saved image: {0}".format(out_name)
    mahotas.imsave(out_name, np.int8(image))

def classify_image(image, model, handle):
    """Classify a single image based on a given model. Only the valid pixels are classified, which means the output 
    will be smaller than the input. Currently the layers are copied to the gpu for each image. This is unnecessary and should
    be fixed to further increase the speed."""

    st = time.time()
    layers = model.layers
    patch_dims = (39, 39)
    valid_x = image.shape[0]-patch_dims[0] + 1
    valid_y = image.shape[1]-patch_dims[1] + 1


    #We set a max batchsize to prevent running out of memory, but actual batches are done by row
    max_batchsize = 2**14 #16 times 1024 
    batch_rows = max_batchsize/valid_x 
    batchsize = valid_x*batch_rows

    #get the indices for classification. Note that these correspond to the upper left corner of the patch, not to the pixel being classified
    pixels = [(x,y) for x in range(valid_x) for y in range(valid_y)]
    output = gpu_computation(image, patch_dims, batchsize, batch_rows, layers, pixels, handle)
    output = output.get()
    output = output.reshape(valid_x, valid_y)
    print "Classified image in: {0:.4e} seconds", time.time()-st
    return output

def classify(image_names, model_file_name, output_names):
    """
    Classify a set of images using the given model.
    
    Parameters
    ----------
    image_names : iterable of strings
        names of the input images
    model_file_name : string
        name of the file containing the model
    output_names : iterable of strings
        names of the output images
    
    Notes
    -----
    image_names and output_names should have the same length and indices match. i.e. image_names[idx] -> output_names[idx]
    """
    handle = cublas.cublasCreate()
    model = serial.load(model_file_name)
    outputs = []
    for image_name, output_name in zip(image_names, output_names):
        image = load_image(image_name)
        output = classify_image(image, model, handle)
        save_image(np.int32(np.round(output*255)), output_name)
    cublas.cublasDestroy(handle)

def main():
    """
    For testing and timing. 
    """
    handle = cublas.cublasCreate() 
    image = np.float32((np.random.rand(1024, 1024) - .5) * 2)
    model = serial.load(model_file_name)
    layers = model.layers
    
    patch_dims = (39, 39)
    #There is a bug that occurs if running with too long a batch_rows_l
    #Most likely a memory allocation issue that is not being reported correctly
    batch_rows_l = [8] 
    batchsizes = map(lambda x: x*(1024-39+1), batch_rows_l)
    pixels = [(x, y) for x in range(1024-39+1) for y in range(1024-39+1)]
    
    #Uncomment to use pylearn2 to classify to check result
    p_output = pylearn2_computation(model, image, patch_dims, batchsizes[0], pixels)
    p_output = np.transpose(p_output)
    num_trials = 1
    for batchsize, batch_rows in zip(batchsizes, batch_rows_l):
        st = time.time()
        for trial in range(num_trials):
            output = gpu_computation(image, patch_dims, batchsize, batch_rows, layers, pixels, handle)
            output = output.get()
        tot = time.time()-st
        print "Batchsize {0}".format(batchsize)
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel: {0:.4e} seconds".format(tot/len(pixels*num_trials))
        print "Pixels per second: {0:.4e}".format(len(pixels*num_trials)/tot)
    for end in time_ends:
        end.synchronize()
    sgemm_times = map(lambda start, end: end.time_since(start)/1000, time_starts, time_ends)
    tot_sgemm_time = sum(sgemm_times)
    print "Total sgemm time: {0:.4e} seconds\nTotal gflop: {1:.4e}\nGflops: {2:.4e}".format(tot_sgemm_time, sgemm_gflop, sgemm_gflop/tot_sgemm_time)

    #Uncomment to compare results of gpu and pylearn2 classifications 
    #output = output.reshape(1024-39, 1024-39)
    print output, p_output
    
    print np.allclose(p_output[0], output, rtol=1e-04, atol=1e-07)
    cublas.cublasDestroy(handle)
    
    return 

def compute_sgemm(weights, values, biases, handle, m, k, n):
    alpha = np.float32(1.0); beta = np.float32(1.0);
    #to do C = A*B + C, we actually do C_t = B_t*A_t + C_t and then transpose, but the transposing is all done implicitly in copy to and from gpu, so we just note that we do BA not AB
    flop = float(2*m*n*k)
    gflop = flop/10**9
    
    
    #Uncomment the two blocks below for precise sgemm timing
    """
    start = cu.Event()
    end = cu.Event()
    start.record()
    """
    #We want to do biases = weights*values + biases, which has dimensions (m*n) = (m*k)*(k*n) + (m*n)
    #but instead use trasnposes as in above note
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, values.ptr, n, weights.ptr, k, beta, biases.ptr, n)
    """
    end.record()
    global time_starts
    global time_ends
    global sgemm_gflop
    time_starts.append(start); time_ends.append(end);
    sgemm_gflop += gflop
    """

def pylearn2_computation(model, image, patch_dims, batchsize, pixels):
    """ Classify an image using pylearn2. For testing to compare result of pylearn2 with result of cuda code """
    patchsize = patch_dims[0]*patch_dims[1]
    nbatches = (len(pixels) + batchsize -1)/batchsize
    model.set_batch_size(batchsize) 
    data = model.get_input_space().make_batch_theano()
    outputs = np.float32(np.zeros((nbatches*batchsize, 2)))
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
        #print start, batchsize
        outputs[start:start+batchsize, :] = values
    return outputs[:len(pixels), :]

def gpu_computation(image, patch_dims, batchsize, batch_rows, layers, pixels, handle):
    """Classify an image on the gpu.

    Parameters
    ----------
    image: numpy array
        float image to classify
    patch_dims: tuple
        size of patch used (height, width)
    batchsize: int
        total pixels per batch
    batch_rows: int
        rows per batch
    layers: list
        list of the pylearn2 layer objects
    pixels: list
        list of tuples of pixel indices to classify
    handle: cublas handle
    """
        
    image_d = gpu.to_gpu(image)
    patchsize = patch_dims[0]*patch_dims[1]
    npixels = len(pixels)
    nbatches = (npixels + batchsize - 1) / batchsize
    
    #prepare the gpu arrays
    weights_l = []; biases_l = []; outputs_l = [];
    for layer in layers:
        weights = layer.get_weights(); biases = np.float32(layer.b.get_value());
        #note that we have to transpose the weights array to match the row major format of the images
        weights = np.ascontiguousarray(np.transpose(weights))
        #we reshape (just add a dimension) first to prevent tiling from prepending the dimension
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
    
    window = (patch_dims[0]+batch_rows-1, image_d.shape[1], 1)
    height_col = window[0] - patch_dims[0] + 1
    width_col = window[1] - patch_dims[1] + 1
    cols = gpu.empty((patch_dims[0]*patch_dims[1], height_col*width_col), np.float32)

    for batch in range(nbatches):
        #offset for im2col pixel start
        image_offset = batch*batch_rows*image_d.shape[0]
        im2col.compute_im2col(image_d, window[0], window[1], window[2], patch_dims[1], 0, 1, image_offset, cols)
        inputs = cols
        #offset for output array
        offset = batch*batchsize 
        
        #run the whole network for one batch
        for layern, (weights, biases, outputs) in enumerate(zip(weights_l, biases_l, outputs_l)):
            cu.memcpy_dtod(outputs.ptr, biases.ptr, outputs.nbytes)
            compute_sgemm(weights, inputs, outputs, handle, weights.shape[0], weights.shape[1], batchsize) 
            if (layern < len(layers)-1):
                rect.compute_rectify(outputs)
                inputs = outputs
            else:
                soft_max.compute_soft_max(outputs, classes, offset)
    return classes 

if __name__ == "__main__":
    #main()
    #sys.exit()

    if len(sys.argv) != 4:
        print "Usage: python main.py <image_folder> <output_folder> <model_file>"
        sys.exit()
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    #output_path = "/home/dmmiron/cuda/fast_gpu_net/fully_connected/"
    model_file_name = sys.argv[3]
    images = sorted(glob.glob(image_path + "/*"))
    output_names = [output_path.rstrip("/") + "/" + image_name.split("/")[-1].rstrip(".tif") + "_classified.tif" for image_name in images]
    classify(images, model_file_name, output_names)

