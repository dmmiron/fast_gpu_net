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

def test_im2col():
    im2col_gpu.init()
    image = np.float32(np.reshape(np.arange(0, 200, 1), [10, 10, 2]))
    image_d = gpu.to_gpu(image)
    """for i in range(5):
        for j in range(5):
            for k in range(2):
                print (image[i, j, k]),
            print "\n"
        print "\n"
    print "\n"
    """
    print image
    for i in range(2):
        for j in range(2):
            offset = i*10 + j
            col = im2col_gpu.compute_im2col(image_d, 5, 5, 2, np.int32(4), np.int32(0), np.int32(1), offset)
            print col


def main():
    #compile gpu kernels
    maxpool_gpu.init()
    im2col_gpu.init()
    nstreams = int(sys.argv[1])
    streams = []
    for n in range(nstreams):
        streams.append(cu.Stream())
    #set up test data
    #image = (np.random.rand(1, 49, 49) - .5) * 2
    image = np.float32((np.random.rand(100, 100, 1) - .5) * 2)
    #image = np.float32((np.reshape(np.arange(0, 10*10, 1), [10, 10, 1])))
    #image = np.float32((np.reshape(np.arange(0, 100*100, 1), [100, 100, 1])))

    #print image
    #image = np.float32(np.reshape(np.arange(0, 49*49, 1), [49, 49, 1]))
    ser_image = to_serial(image)
    #kernels, layers per kernel, width, height

    kernels_0 = np.float32((np.random.rand(4, 4, 1, 64) - .5 ) * 2)
    #kernels_0 = np.float32(np.reshape(np.arange(0, 4*4*64, 1), [4, 4, 1, 64]))
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
     
     
    #perform serial computation
    """ 
    conv = comp_convolution(ser_image, ser_kernels_0, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_1, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_2, pad, stride)
    conv_max = maxout(conv, 2, 4)
    conv_max_r = conv_max.ravel()
    result = np.dot(weights, conv_max_r)
    """ 
    
    
    kernels = [kernels_0, kernels_1, kernels_2]
    biases = [bias_0, bias_1, bias_2]
    max_sizes = [max_0, max_1, max_2]
    #when using actual images will need to offset pixels so they are the center of the window
    pixels = [(0, 0), (1, 0), (0, 1)]
    pixels = [(x, y) for x in range(30) for y in range(30)]
    #window = (49, 49)
    window = (49, 49)
    
    #perform parallel computation
    num_trials = 1
    for i in range(num_trials):
        print "Trial {0}\n".format(i)
        output = gpu_computation(image, kernels, biases, max_sizes, pixels, window, streams)
    #print output
    #out_max = from_serial(conv_max)
    #print out_max-output 
    #print out_max, output
    #print np.allclose(output, out_max, rtol=1e-04, atol=1e-07) 
    #print np.where(np.isclose(output, out_max)==False)

def to_serial(array):
    """This method converts the numpy default row-major ordering into an ordering to match the way we are indexing on the gpu. In particular each 2D image slice is indexed in a row-major format (i,j). However, inconsistent with row major ordering we traverse each slice before changing to a new slice. If the array is 4d (kernels) then we traverse the 4th dimension (each 3d kernel last). Thus the indices change (fastest to slowest) in the order column, row, slice, stack"""
    shape_list = [dim for dim in array.shape]
    #reverse dimensions after first two and move to beginning
    #print shape_list[:1:-1], shape_list[:2]
    shape_list = shape_list[:1:-1] + shape_list[:2]
    return array.reshape(shape_list)

def from_serial(array):
    shape_list = [dim for dim in array.shape]
    shape_list = shape_list[-2:] + shape_list[-3::-1]
    return array.reshape(shape_list)

def comp_convolution(image, kernels, pad, stride):
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
    return conv


def maxout(image, max_ksize_xy, max_ksize_z):
    height = image.shape[1]
    width = image.shape[2]
    depth = image.shape[0]
    out_height = (height+max_ksize_xy-1)/max_ksize_xy;
    out_width = (width+max_ksize_xy-1)/max_ksize_xy;
    out_depth = (depth+max_ksize_z-1)/max_ksize_z;

    output = np.zeros((out_depth, out_height, out_width))
    for i in range(out_height):
        for j in range(out_width):
            for k in range(out_depth):
                image_i = i*max_ksize_xy
                image_j = j*max_ksize_xy
                image_k = k*max_ksize_z

                output[k, i, j] = np.amax(image[image_k:image_k+max_ksize_z, image_i:image_i+max_ksize_xy, image_j:image_j+max_ksize_xy].ravel())

    return output


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
    
 

def gpu_computation(image, kernels, biases, max_sizes, pixels, window_sizes, streams):
    handle = cublas.cublasCreate()
    results = []
    pad = 0; stride = 1; 
    full_image_d = gpu.to_gpu(image)
    biases_d = []
    kernels_d = []
    kernel_dims = []
    results = []
    for bias, kernel in zip(biases, kernels):
        biases_d.append(gpu.to_gpu(bias))
        kernels_d.append(gpu.to_gpu(kernel))
        kernel_dims.append(kernel.shape)

    nstreams = len(streams)
    
    for (pix_id, pixel) in enumerate(pixels):
        stream = streams[pix_id % nstreams]
        image_d = full_image_d
        offset = pixel[0]*full_image_d.shape[1] + pixel[1]
        for layer_n, (kernel_d, bias_d, max_size, kdim) in enumerate(zip(kernels_d, biases_d, max_sizes, kernel_dims)):
            sgemm_bias = bias_d.copy()
            #only the first layer uses the full image with an offset
            if (layer_n == 0):
                height = window_sizes[0]
                width = window_sizes[1]
                channels = 1
            else:
                offset = 0
                height = image_d.shape[0]
                width = image_d.shape[1]
                channels = image_d.shape[2]

            #print height, width, channels
            print "layer {0}".format(layer_n)
            #kernel_d = gpu.to_gpu_async(kernel, stream=stream)
            #bias_d = gpu.to_gpu_async(bias, stream=stream)
    
            ksize = kdim[0]
            kchannels = kdim[3]
            height_col = (height + 2 * pad - ksize) / stride + 1
            width_col = (width + 2 * pad - ksize) / stride + 1 
            kernel_d = kernel_d.reshape(kchannels, ksize*ksize*channels)

            result = im2col_gpu.compute_im2col(image_d, height, width, channels, np.int32(ksize), np.int32(pad), np.int32(stride), offset, stream)
            #print result, "result"
            sgemm_bias = sgemm_bias.reshape(kernel_d.shape[0], result.shape[1])
            
            compute_sgemm(result, kernel_d, sgemm_bias, stream, handle)
            sgemm_bias = sgemm_bias.reshape(np.int32(height_col), np.int32(width_col), np.int32(kchannels))
            image_d = maxpool_gpu.compute_max(sgemm_bias, np.int32(max_size), stream) 
        results.append(image_d)
    results = map(lambda x: x.get(), results)
    cublas.cublasDestroy(handle)
    return results

if __name__ == "__main__":
    main()
    #test_im2col()
