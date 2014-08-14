#!/usr/bin/env python
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import numpy as np
import time
import glob
import mahotas
import im2col as im2col_gpu
import maxpool as maxpool_gpu
import soft_max as soft_max
import theano
import theano.tensor as T
from theano.sandbox.cuda import dimshuffle as cuda_dimshuffle

import scikits.cuda.cublas as cublas
import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

from scipy.signal import convolve

im2col_gpu.init()
maxpool_gpu.init()
soft_max.init()

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
    mahotas.imsave(out_name, np.int8(image))
    print "saved image: {0}".format(out_name)

def classify_image(image, model, kernels, biases, max_sizes, soft_weights, soft_bias, window, handle):
    """Classify a single image based on a given model. Only the valid pixels are classified, which means the output 
    will be smaller than the input."""

    st = time.time()
    valid_x = image.shape[0]-window[0] + 1
    valid_y = image.shape[1]-window[1] + 1
    #batchsize = 16 
    batchsize = 64
    #batchsize = 1 
    pixels = [(x,y) for x in range(valid_x) for y in range(valid_y)]
    batches = []
    for i in range(len(pixels)/batchsize):
        start = i*batchsize
        batches.append(pixels[start:start+batchsize])
    #batches = batches[0:1]    
    #p_output = pylearn2_computation(model, image, window, batchsize, pixels)
    #p_output = p_output.reshape(valid_x, valid_y)
    #save_image(np.int8(np.round(255*p_output)), "pylearn2_output.tif")
    
    output = gpu_computation(image, kernels, biases, max_sizes, soft_weights, soft_bias, batches, window)
    output = output.get()
    output = output.reshape(valid_x, valid_y)
    #print output
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
    This network copies the weights to the gpu once to classify all the images as it should. This can be used as a model 
    to make the same change to the fully connected network.
    """
    handle = cublas.cublasCreate()
    model = serial.load(model_file_name)

    layers = model.layers
    convs = layers[:-1]; softmax = layers[-1];
    convs = map(lambda layer: layer.get_params(), convs)
    kernels = map(lambda layer: np.array(layer[0].eval()), convs)

    #This can be simplified
    kernels = map(lambda kernel: np.ascontiguousarray(np.rollaxis(kernel, 0, 3)), kernels)
    kdims = map(lambda kernel: kernel.shape, kernels)
    kernels = map(lambda layer: layer[0].dimshuffle(3, 0, 1, 2).eval(), convs)
    kernels = map(lambda kernel, kdim: kernel.reshape(kdim), kernels, kdims)
    
    
    biases = map(lambda layer: np.array(layer[1].eval()), convs)
    bias_dims = map(lambda bias: bias.shape, biases)
    max_sizes = map(lambda layer: layer.pool_shape + [layer.num_pieces], layers[:-1])
    
    weights = softmax.get_params()[1]; bias = softmax.get_params()[0];
    
    soft_weights = softmax.get_params()[1].reshape((3, 3, 32, 2)).dimshuffle(3, 2, 0, 1).eval()
    soft_weights = np.ascontiguousarray(np.reshape(soft_weights, (2, 288)).transpose())
    soft_bias = softmax.get_params()[0].get_value()[::1]

    window = layers[0].input_space.shape
    outputs = []
    for image_name, output_name in zip(image_names, output_names):
        image = load_image(image_name)
        output = classify_image(image, model, kernels, biases, max_sizes, soft_weights, soft_bias, window, handle)
        save_image(np.int8(np.round(output*255)), output_name)
    cublas.cublasDestroy(handle)

def main():
    """for testing"""
    #compile gpu kernels
    maxpool_gpu.init()
    im2col_gpu.init()

    #batchsizes = [2**x for x in range(2, 8)]
    batchsizes = [32, 64, 128]
    batchsizes = [64]

    
    #set up test data
    image = np.float32((np.random.rand(1024, 1024, 1) - .5) * 2)

    ser_image = to_serial(image)
    #kernels, layers per kernel, width, height

    kernels_0 = np.float32((np.random.rand(4, 4, 1, 64) - .5) * 2) / 10
    ser_kernels_0 = to_serial(kernels_0)
    bias_0 = np.float32(np.zeros((46, 46, 64)))
    ser_bias_0 = to_serial(bias_0)
    max_0 = np.int32((2, 2, 2))

    kernels_1 = np.float32((np.random.rand(4, 4, 32, 64) - .5 ) * 2)/10
    ser_kernels_1 = to_serial(kernels_1)
    bias_1 = np.float32(np.zeros((20, 20, 64)))
    ser_bias_1 = to_serial(bias_1)
    max_1 = np.int32((2, 2, 2))

    kernels_2 = np.float32((np.random.rand(5, 5, 32, 128) - .5 ) * 2)/10
    ser_kernels_2 = to_serial(kernels_2)
    bias_2 = np.float32(np.zeros((6, 6, 128)))
    ser_bias_2 = to_serial(bias_2)
    max_2 = np.int32((2, 2, 4))
    soft_weights = np.float32(((np.random.rand(288, 2)) - .5) * 2) / 10
    soft_bias = np.float32(np.random.rand(2))
    soft_bias = np.zeros((2))

    pad = np.int32(0)
    stride = np.int32(1)
    window = (49, 49)

    kernels = [kernels_0, kernels_1, kernels_2]
    ser_kernels = map(to_serial, kernels)
    biases = [bias_0, bias_1, bias_2]
    ser_biases = map(to_serial, biases)
    max_sizes = [max_0, max_1, max_2]
    ser_max_sizes = map(to_serial, max_sizes)


    num_trials = 1
    valid_x = image.shape[0]-window[0]; valid_y = image.shape[1]-window[1];

    pixels = [(x, y) for x in range(valid_x) for y in range(valid_y)]
    #pixels = [(x, y) for x in range(4) for y in range(4)]
    #print pixels
    npixels = len(pixels)
    #batchsizes = [4]
     
    #perform parallel computation
    for batchsize in batchsizes:
        batches = []
        for i in range(len(pixels)/batchsize):
            start = i*batchsize
            batches.append(pixels[start:start+batchsize])
        print "Batchsize: {0}\nBatches: {1}\nPixels: {2}\n".format(batchsize, len(batches), batchsize*len(batches))
        st = time.time()
        for i in range(num_trials):
            print "Trial {0}\n".format(i)
            #NOTE THAT OUTPUT IS A DEVICE ARRAY
            output = gpu_computation(image, kernels, biases, max_sizes, soft_weights, soft_bias, batches, window)
            print output.get()
        tot = time.time()-st
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel {0:.4e} seconds".format(tot/(npixels*num_trials))
        print "Pixels per second {0:.4e}".format(npixels*num_trials/tot)
    
    #Uncomment to perform serial computation
    """ 
    for batchsize in batchsizes:
        batches = []
        for i in range(len(pixels)/batchsize):
            start = i*batchsize
            batches.append(pixels[start:start+batchsize])
        print "Batchsize: {0}\nBatches: {1}\nPixels: {2}\n".format(batchsize, len(batches), batchsize*len(batches))
        st = time.time()
        for i in range(num_trials):
            print "Trial {0}\n".format(i)
            s_output = batch_serial(ser_image, window, ser_kernels, ser_biases, ser_max_sizes, pad, stride, batches)
            #comp_results(output, s_output)
            #print "Time so far {0:.4e} seconds".format(time.time()-st)
        tot = time.time()-st
        print "Serial Total time: {0:.4e} seconds".format(tot)
        print "Serial Time per pixel {0:.4e} seconds".format(tot/(npixels*num_trials))
        print "Serial Pixels per second {0:.4e}".format(npixels*num_trials/tot)
    """ 

def comp_results(output, s_output):
    for batch in range(len(output)):
        for pixel in range(output[0].shape[0]):
            print np.allclose(output[batch][pixel].get(), s_output[batch][pixel], rtol=1e-04, atol=1e-07)

def comp_offsets(pixels, image):
    offsets = []
    for pixel in pixels:
        offsets.append(pixel[0]*image.shape[1] + pixel[1])
    return offsets

def batch_serial(image, window, kernels, biases, max_sizes, pad, stride, batches):
    output = []
    for batch in batches:
        batch_out = []
        for pixel in batch:
            batch_out.append(serial(image, window, kernels, biases, max_sizes, pad, stride, pixel))
        output.append(batch_out)
    return output

def comp_serial(image, window, kernels, biases, max_sizes, pad, stride, pixel):
    conv = comp_convolution(image[:, pixel[0]:pixel[0]+window[0], pixel[1]:pixel[1]+window[1]], kernels[0], biases[0], pad, stride)
    conv_max = maxout(conv, max_sizes[0])
    conv = comp_convolution(conv_max, kernels[1], biases[1], pad, stride)
    conv_max = maxout(conv, max_sizes[1])
    conv = comp_convolution(conv_max, kernels[2], biases[2], pad, stride)
    conv_max = maxout(conv, max_sizes[2])

    #for use when computing final output
    #conv_max_r = conv_max.ravel()
    #result = np.dot(weights, conv_max_r)
    return conv_max


def to_serial(array):
    """This method converts the numpy default row-major ordering into an ordering to match the way we are indexing on the gpu. In particular each 2D image slice is indexed in a row-major format (i,j). However, inconsistent with row major ordering we traverse each slice before changing to a new slice. If the array is 4d (kernels) then we traverse the 4th dimension (each 3d kernel last). Thus the indices change (fastest to slowest) in the order column, row, slice, stack"""
    shape_list = [dim for dim in array.shape]
    #reverse dimensions after first two and move to beginning
    shape_list = shape_list[:1:-1] + shape_list[:2]
    return array.reshape(shape_list)

def from_serial(array):
    shape_list = [dim for dim in array.shape]
    shape_list = shape_list[-2:] + shape_list[-3::-1]
    return array.reshape(shape_list)

def comp_convolution(image, kernels, bias, pad, stride):
    height = image.shape[1]
    width = image.shape[2]
    #rowsxcolumnsxlayers of kernels

    ksize = kernels.shape[2]

    height_col = (height + 2 * pad - ksize) / stride + 1
    width_col = (width + 2 * pad - ksize) / stride + 1
    conv = np.zeros((kernels.shape[0], height_col, width_col))

    for ki in range(kernels.shape[0]):
        kernel = kernels[ki, :, :, :]
        full_conv2d = np.zeros((height_col, width_col)) 
        for layer in range(kernels.shape[1]):
            kernel_layer = kernel[layer, :, :]
            image_layer = image[layer, :, :]
            #NOTE THE REVERSED KERNEL
            #STILL NOT CLEAR WHY THIS SHOULD BE NECESSARY
            full_conv2d += convolve(image_layer, kernel_layer[::-1, ::-1], mode='valid')
        conv[ki, :, :] = full_conv2d
    conv += bias
    return conv


def maxout(image, max_ksize):
    maxx = max_ksize[0]; maxy = max_ksize[1]; maxz = max_ksize[2];
    height = image.shape[1]
    width = image.shape[2]
    depth = image.shape[0]
    out_height = (height+maxy-1)/maxy;
    out_width = (width+maxx-1)/maxx;
    out_depth = (depth+maxz-1)/maxz;

    output = np.zeros((out_depth, out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            for k in range(out_depth):
                image_i = i*maxy
                image_j = j*maxx
                image_k = k*maxz

                output[k, i, j] = np.amax(image[image_k:image_k+maxz, image_i:image_i+maxy, image_j:image_j+maxx].ravel())
    return output

def compute_sgemm_batched(cols, kernels, biases, handle, m, k, n):
    batchsize = len(cols)
    #takes gpu arrays of pointers to pointers
    alpha = np.float32(1.0); beta = np.float32(1.0);
    flop = 2*m*n*k*batchsize
    cublas.cublasSgemmBatched(handle, 'n', 'n', n, m, k, alpha, cols.ptr, n, kernels.ptr, k, beta, biases.ptr, n, batchsize);

def compute_sgemm(col, kernel, bias, handle):
    alpha = np.float32(1.0); beta = np.float32(1.0);

    #(mxk)x(kxn)
    m = np.int32(kernel.shape[0])
    k = np.int32(kernel.shape[1]) 
    n = np.int32(col.shape[1])
    #2k-1 operatiosn per entry in C matrix. C is m*n so m*n*(2k-1) for mat mult
    #another m*n additions to perform AB+C
    #lower bound ignoring alpha and beta multiplications
    flop = 2*m*n*k

    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, col.ptr, n, kernel.ptr, k, beta, bias.ptr, n);

def compute_dims(image, kernels, biases, max_sizes, batchsize, window_sizes, pad, stride):
    image_dims = []; col_dims = []; kernel_dims = []; bias_dims = []; sgemm_dims = []; out_dims = [];
    ksizes = []; kchannels_s = [];
    height_col = 0; width_col = 0; ksize = 0; kchannels = 0; m = 0; k = 0; n = 0; out_height = 0; out_width = 0; out_channels = 0;
    for layer_n, (bias, kernel, max_size) in enumerate(zip(biases, kernels, max_sizes)):
        bias = bias.reshape((1, bias.shape[2], bias.shape[0]*bias.shape[1]))
        bias_dims.append([bias.shape[1], bias.shape[2]])
        
        ksize = kernel.shape[0]; kchannels = kernel.shape[3];
        ksizes.append(ksize); kchannels_s.append(kchannels);

        if (layer_n == 0):
            height = window_sizes[0]; width = window_sizes[1]; channels = 1;
        else:
            height = out_height; width = out_width; channels = out_channels
            
        image_dims.append([height, width, channels]) 

        height_col = (height + 2 * pad - ksize) / stride + 1
        width_col = (width + 2 * pad - ksize) / stride + 1
        col_dims.append([height_col, width_col])

        m = kchannels; k = ksize*ksize*channels; n = height_col*width_col;
        sgemm_dims.append([m, k, n])

        kernel_dims.append([kchannels, ksize*ksize*channels])

        out_height = (height_col + max_size[0] - 1) / max_size[0]
        out_width =  (width_col + max_size[1] - 1) / max_size[1]
        out_channels = (kchannels + max_size[2] - 1) / max_size[2]
        out_dims.append([out_height, out_width, out_channels])
    return image_dims, col_dims, kernel_dims, bias_dims, sgemm_dims, out_dims, ksizes, kchannels_s

def pylearn2_computation(model, image, window, batchsize, pixels):
    windowsize = window[0]*window[1]
    nbatches = (len(pixels) + batchsize -1)/batchsize
    model.set_batch_size(batchsize) 
    data = model.get_input_space().make_batch_theano()
    
    y0, y1, y2, y3 = model.fprop(data, return_all=True)
    layer0 = theano.function([data], [y0])
    layer1 = theano.function([data], [y1])
    layer2 = theano.function([data], [y2])
    layer3 = theano.function([data], [y3])
    #layer3_dp = theano.function([data], [y3.dot_product_result]) 
    #print layer3_dp
    y = model.fprop(data)
    classify = theano.function([data], [y], name = 'classify')
    outputs = np.float32(np.zeros((nbatches*batchsize)))
    all_layers = []
    for batch in range(nbatches):
        start = batch*batchsize
        values = np.float32(np.zeros((1, window[0], window[1], batchsize)))
        for pixn, pixel in zip(range(batchsize), pixels[start:start+batchsize]):
            values[0, :, :, pixn] = image[pixel[0]:pixel[0]+window[0], pixel[1]:pixel[1]+window[1]]
        dp_output = layer3(values)[0]
        values = layer2(values)[0]
        print np.sum(np.array(values).ravel()), "dot product input"
        return dp_output
        """
        all_layers.append(layer0(values))
        all_layers.append(layer1(values))
        all_layers.append(layer2(values))
        all_layers.append(layer3(values))
        return all_layers
        """
        outputs[start:start+batchsize] = classify(values)[0][:, 0] 
        #outputs[start:start+batchsize, :] = values
        #print outputs

    return outputs

def gpu_computation(image, kernels, biases, max_sizes, soft_weights, soft_bias, batches, window_sizes):
    nbatches = len(batches)
    batchsize = len(batches[0])
    npixels = nbatches*batchsize
    layers = len(kernels)
    handle = cublas.cublasCreate()
    results = []
    result_ps = []
    pad = 0; stride = 1; 
    full_image_d = gpu.to_gpu(image)

    image_dims, col_dims, kernel_dims, bias_dims, sgemm_dims, out_dims, ksizes, kchannels_s = compute_dims(image, kernels, biases, max_sizes, batchsize, window_sizes, pad, stride)
    
    b_result = [];
    b_offsets_d = [];
    
    kernels_d = [];
    cols = []; col_ps = [];
    biases_d = [];
    sgemm_biases = []; sgemm_biases_ps = [];
    outputs = [];

    for layer_n, (bias, kernel, sgemm_dim, im_dim, out_dim, max_ksize, ksize, kchannels) in enumerate(zip(biases, kernels, sgemm_dims, image_dims, out_dims, max_sizes, ksizes, kchannels_s)):
        col = gpu.empty((batchsize, sgemm_dim[1], sgemm_dim[2]), np.float32) 
        cols.append(col)
        col_ps.append([col[idx, :, :].ptr for idx in range(batchsize)])
        
        #reuse the same kernels for every pixel
        kernel_d = gpu.to_gpu(kernel)
        kernel_d = kernel_d.reshape(kchannels, ksize*ksize*im_dim[2])
        kernels_d.append(kernel_d)

 
        #contain the actual data of the biases
        bias = bias.reshape(1, bias.shape[2], bias.shape[0]*bias.shape[1])
        batch_bias = np.tile(bias, (batchsize, 1, 1))
        batch_bias_d = gpu.to_gpu(batch_bias)
        biases_d.append(batch_bias_d)
        
        #scratch space to copy biases to and then write output of sgemm to
        sgemm_bias = gpu.empty(batch_bias.shape, np.float32)
        sgemm_biases.append(sgemm_bias)
        
        sgemm_biases_ps.append([sgemm_bias[idx, :, :].ptr for idx in range(batchsize)])

        #space for output of maxpool
        output = gpu.empty((batchsize, out_dim[2], out_dim[0], out_dim[1]), np.float32)
        outputs.append(output)

    #space for final output
    classes = gpu.empty(npixels, np.float32)
    soft_weights_d = gpu.to_gpu(soft_weights)
    soft_bias = soft_bias.reshape(1, soft_bias.shape[0])
    soft_bias_d = gpu.to_gpu(np.ascontiguousarray(np.reshape(np.tile(soft_bias, (batchsize, 1)), (2, batchsize))))
    soft_bias_scratch = gpu.empty((soft_bias_d.shape[0], soft_bias_d.shape[1]), np.float32)

    col_ps_d = gpu.to_gpu(np.array(col_ps))

    kernel_ps = map(lambda x: [x.ptr]*batchsize, kernels_d)
    kernel_ps_d = gpu.to_gpu(np.array(kernel_ps))

    sgemm_biases_ps_d = gpu.to_gpu(np.array(sgemm_biases_ps))

    for batch in batches:
        offsets = comp_offsets(batch, full_image_d)
        offsets_d = gpu.to_gpu(np.int32(np.array(offsets)))
        b_offsets_d.append(offsets_d);

        #space to hold final result of each layer
        result = gpu.empty((out_dims[layers-1][2], out_dims[layers-1][0], out_dims[layers-1][1]), np.float32)
        b_result.append(result)

    for batchn, (batch, offsets_d, result) in enumerate(zip(batches, b_offsets_d, b_result)):

        image_d = full_image_d
        for layer_n, (im_dim, col_dim, kdim, bias_dim, sgemm_dim, out_dim, ksize, kchannels, max_size) in enumerate(zip(image_dims, col_dims, kernel_dims, bias_dims, sgemm_dims, out_dims, ksizes, kchannels_s, max_sizes)):

            sgemm_bias = sgemm_biases[layer_n]
            cu.memcpy_dtod(sgemm_bias.ptr, biases_d[layer_n].ptr, sgemm_bias.nbytes)

            im2col_gpu.compute_im2col_batched(image_d, im_dim[0], im_dim[1], im_dim[2], np.int32(ksize), np.int32(pad), np.int32(stride), offsets_d, layer_n, batchsize, cols[layer_n])
            compute_sgemm_batched(col_ps_d[layer_n], kernel_ps_d[layer_n], sgemm_biases_ps_d[layer_n], handle, sgemm_dim[0], sgemm_dim[1], sgemm_dim[2])
            sgemm_bias = sgemm_bias.reshape(np.int32(batchsize), np.int32(kchannels), col_dim[0], col_dim[1])
            maxpool_gpu.compute_max_batched(sgemm_bias, outputs[layer_n], np.int32(max_size))
            image_d = outputs[layer_n]
        result = outputs[layers-1]
        result = result.reshape(result.shape[0], result.shape[1]*result.shape[2]*result.shape[3]) 
        cu.memcpy_dtod(soft_bias_scratch.ptr, soft_bias_d.ptr, soft_bias_d.nbytes)
        np_soft_weights = soft_weights_d.get()
        np_result = result.get()
        compute_sgemm(soft_weights_d, result, soft_bias_scratch, handle)
        
        offset = batchn*batchsize
        soft_max_in = soft_bias_scratch
        soft_max.compute_soft_max(soft_max_in, classes, offset)
        result_ps.append(result)
        
    cublas.cublasDestroy(handle)
    return classes

if __name__ == "__main__":
    #main()
    #sys.exit()
    
    if len(sys.argv) != 4:
        print "Usage: python main.py <image_folder> <output_folder> <model_file>"
        sys.exit()
    image_path = sys.argv[1]
    output_path = sys.argv[2]
    model_file_name = sys.argv[3]
    images = sorted(glob.glob(image_path + "/*"))
    output_names = [output_path.rstrip("/") + "/" + image_name.split("/")[-1].rstrip(".tif") + "_classified.tif" for image_name in images]
    classify(images, model_file_name, output_names)
