#!/usr/bin/env python
import sys
import numpy as np
import time
import im2col as im2col_gpu
import maxpool as maxpool_gpu

import scikits.cuda.cublas as cublas
import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

from scipy.signal import convolve

def main():
    #compile gpu kernels
    maxpool_gpu.init()
    im2col_gpu.init()

    #batchsizes = [2**x for x in range(2, 8)]
    batchsizes = [32, 64, 128]
    nstreams = 1
    streams = []
    
    for n in range(nstreams):
        streams.append(cu.Stream())
    #set up test data
    image = np.float32((np.random.rand(1024, 1024, 1) - .5) * 2)
    #image = np.float32((np.reshape(np.arange(0, 100*100, 1), [100, 100, 1])))
    #image = np.float32((np.reshape(np.arange(0, 10*10, 1), [10, 10, 1])))

    #image = np.float32(np.reshape(np.arange(0, 100*100, 1), [100, 100, 1]))
    ser_image = to_serial(image)
    #kernels, layers per kernel, width, height

    #kernels_0 = np.float32((np.random.rand(4, 4, 1, 64) - .5 ) * 2)
    kernels_0 = np.float32(np.reshape(np.arange(0, 4*4*64, 1), [4, 4, 1, 64]))
    kernels_0 = np.float32(np.ones((4, 4, 1, 64)))
    ser_kernels_0 = to_serial(kernels_0)
    bias_0 = np.float32(np.zeros((46, 46, 64)))
    ser_bias_0 = to_serial(bias_0)
    max_0 = np.int32((2, 2, 2))

    kernels_1 = np.float32((np.random.rand(4, 4, 32, 64) - .5 ) * 2)
    ser_kernels_1 = to_serial(kernels_1)
    bias_1 = np.float32(np.zeros((20, 20, 64)))
    ser_bias_1 = to_serial(bias_1)
    max_1 = np.int32((2, 2, 2))

    kernels_2 = np.float32((np.random.rand(5, 5, 32, 128) - .5 ) * 2)
    ser_kernels_2 = to_serial(kernels_2)
    bias_2 = np.float32(np.zeros((6, 6, 128)))
    ser_bias_2 = to_serial(bias_2)
    max_2 = np.int32((2, 2, 4))
    weights = np.float32(np.random.rand(2, 288))

    pad = np.int32(0)
    stride = np.int32(1)
    window = (49, 49)


    #for testing
    """
    image = np.float32((np.reshape(np.arange(0, 10*10, 1), [10, 10, 1])))
    kernels_0 = np.float32(np.ones((2, 2, 1, 2)))
    bias_0 = np.float32(np.zeros((6, 6, 2)))
    max_0 = np.int32((2, 2, 2))
    window = (7, 7)
    """

    kernels = [kernels_0, kernels_1, kernels_2]
    #kernels = [kernels_0]
    ser_kernels = map(to_serial, kernels)
    biases = [bias_0, bias_1, bias_2]
    #biases = [bias_0]
    ser_biases = map(to_serial, biases)
    max_sizes = [max_0, max_1, max_2]
    #max_sizes = [max_0]
    ser_max_sizes = map(to_serial, max_sizes)

    """ 
    #perform serial computation
    s_output = serial(ser_image, window, ser_kernels, ser_biases, ser_max_sizes, pad, stride)
    out_max = from_serial(s_output)

    #when using actual images will need to offset pixels so they are the center of the window
    batchsize = 2
    pixels = [(0, 0), (1, 0)]
    output = gpu_computation(image, kernels, biases, max_sizes, [pixels], window, streams)
    print output
    print np.allclose(output[0][0], s_output, rtol=1e-04, atol=1e-07) 
    """
    valid_x = image.shape[0]-window[0]; valid_y = image.shape[1]-window[1];

    pixels = [(x, y) for x in range(valid_x) for y in range(valid_y)]
    #pixels = [(x, y) for x in range(256) for y in range(256)]
    #pixels = [(x, y) for x in range(2) for y in range(2)]
    #print pixels
    #batchsizes = [4]

    num_trials = 1
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
            output = gpu_computation(image, kernels, biases, max_sizes, batches, window, streams)
            #print output
            #TAKE OUT SERIAL LINE WHEN TIMING
            #s_output = batch_serial(ser_image, window, ser_kernels, ser_biases, ser_max_sizes, pad, stride, batches)
            #comp_results(output, s_output)
            print "Time so far {0:.4e} seconds".format(time.time()-st)
        tot = time.time()-st
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel {0:.4e} seconds".format(tot/(len(batches)*batchsize*num_trials))

def comp_results(output, s_output):
    for batch in range(len(output)):
        for pixel in range(output[0].shape[0]):
            #print "batch, pixel"
            print np.allclose(output[batch][pixel].get(), s_output[batch][pixel], rtol=1e-04, atol=1e-07)
            #print "pixel, batch"
            #print np.allclose(output[batch][pixel].get(), s_output[pixel][batch], rtol=1e-04, atol=1e-07)


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

def serial(image, window, kernels, biases, max_sizes, pad, stride, pixel):
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
    #start = cu.Event()
    #end = cu.Event()
    #start.record()
    cublas.cublasSgemmBatched(handle, 'n', 'n', n, m, k, alpha, cols.ptr, n, kernels.ptr, k, beta, biases.ptr, n, batchsize);
    #end.record()

    #end.synchronize()
    #time = end.time_since(start)/1000
    
    #print "sgemm_batched took:\n\t{0:.4e} seconds\n\t{1:.4e} gflop\n\t{2:.4e} gflops".format(time, flop/10**9, flop/(10**9*time))


def compute_sgemm(col, kernel, bias, stream, handle):
    alpha = np.float32(1.0); beta = np.float32(1.0);

    #(mxk)x(kxn)
    m = np.int32(kernel.shape[0])
    k = np.int32(kernel.shape[1]) 
    n = np.int32(col.shape[1])
    #2k-1 operatiosn per entry in C matrix. C is m*n so m*n*(2k-1) for mat mult
    #another m*n additions to perform AB+C
    #lower bound ignoring alpha and beta multiplications
    flop = 2*m*n*k

    cublas.cublasSetStream(handle, stream.handle)
    #start = cu.Event()
    #end = cu.Event()
    #start.record()
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, col.ptr, n, kernel.ptr, k, beta, bias.ptr, n);
    #end.record()

    #end.synchronize()
    #time = end.time_since(start)/1000
    
    #print "sgemm took:\n\t{0:.4e} seconds\n\t{1:.4e} flop\n\t{2:.4e} flops".format(time, flop, flop/time)

def compute_dims(image, kernels, biases, max_sizes, batchsize, window_sizes, pad, stride):
    image_dims = []; col_dims = []; kernel_dims = []; bias_dims = []; sgemm_dims = []; out_dims = [];
    ksizes = []; kchannels_s = [];
    height_col = 0; width_col = 0; ksize = 0; kchannels = 0; m = 0; k = 0; n = 0; out_height = 0; out_width = 0; out_channels = 0;
    for layer_n, (bias, kernel, max_size) in enumerate(zip(biases, kernels, max_sizes)):
        bias = bias.reshape(1, bias.shape[2], bias.shape[0]*bias.shape[1])
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
        #assert(bias_dims[layer_n][0] == m)
        #assert(bias_dims[layer_n][1] == n)

        kernel_dims.append([kchannels, ksize*ksize*channels])

        out_height = (height_col + max_size[0] - 1) / max_size[0]
        out_width =  (width_col + max_size[1] - 1) / max_size[1]
        out_channels = (kchannels + max_size[2] - 1) / max_size[2]
        out_dims.append([out_height, out_width, out_channels])
    return image_dims, col_dims, kernel_dims, bias_dims, sgemm_dims, out_dims, ksizes, kchannels_s


def preallocate():
    return  


def gpu_computation(image, kernels, biases, max_sizes, batches, window_sizes, streams):
    batchsize = len(batches[0])
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

    for batch, offsets_d, result in zip(batches, b_offsets_d, b_result):

        image_d = full_image_d
        for layer_n, (im_dim, col_dim, kdim, bias_dim, sgemm_dim, out_dim, ksize, kchannels, max_size) in enumerate(zip(image_dims, col_dims, kernel_dims, bias_dims, sgemm_dims, out_dims, ksizes, kchannels_s, max_sizes)):

            sgemm_bias = sgemm_biases[layer_n]
            #4 for size of float
            cu.memcpy_dtod(sgemm_bias.ptr, biases_d[layer_n].ptr, sgemm_bias.size*4)

            im2col_gpu.compute_im2col_batched(image_d, im_dim[0], im_dim[1], im_dim[2], np.int32(ksize), np.int32(pad), np.int32(stride), offsets_d, layer_n, batchsize, cols[layer_n])
            compute_sgemm_batched(col_ps_d[layer_n], kernel_ps_d[layer_n], sgemm_biases_ps_d[layer_n], handle, sgemm_dim[0], sgemm_dim[1], sgemm_dim[2])
            sgemm_bias = sgemm_bias.reshape(np.int32(batchsize), np.int32(kchannels), col_dim[0], col_dim[1])
            maxpool_gpu.compute_max_batched(sgemm_bias, outputs[layer_n], np.int32(max_size))
            image_d = outputs[layer_n]
            
        result = outputs[layers-1].copy()
        result_ps.append(result)
        
    cublas.cublasDestroy(handle)
    return result_ps

if __name__ == "__main__":
    main()
