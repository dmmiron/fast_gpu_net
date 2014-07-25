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

    batchsizes = [2**x for x in range(2, 8)]
    nstreams = int(sys.argv[1])
    #nstreams = 1
    streams = []
    
    for n in range(nstreams):
        streams.append(cu.Stream())
    #set up test data
    #image = (np.random.rand(1, 49, 49) - .5) * 2
    #image = np.float32((np.random.rand(100, 100, 1) - .5) * 2)
    #image = np.float32((np.reshape(np.arange(0, 10*10, 1), [10, 10, 1])))
    image = np.float32((np.reshape(np.arange(0, 100*100, 1), [100, 100, 1])))

    #print image
    #image = np.float32(np.reshape(np.arange(0, 49*49, 1), [49, 49, 1]))
    ser_image = to_serial(image)
    #kernels, layers per kernel, width, height

    kernels_0 = np.float32((np.random.rand(4, 4, 1, 64) - .5 ) * 2)
    #kernels_0 = np.float32(np.reshape(np.arange(0, 4*4*64, 1), [4, 4, 1, 64]))
    ser_kernels_0 = to_serial(kernels_0)
    bias_0 = np.float32(np.ones((46, 46, 64)))
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
    #perform serial computation
    """ 
    conv = comp_convolution(ser_image[:, :window[0], :window[1]], ser_kernels_0, ser_bias_0, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_1, ser_bias_1, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_2, ser_bias_2, pad, stride)
    conv_max = maxout(conv, 2, 4)
    conv_max_r = conv_max.ravel()
    result = np.dot(weights, conv_max_r)
    """ 
    
    
    kernels = [kernels_0, kernels_1, kernels_2]
    biases = [bias_0, bias_1, bias_2]
    max_sizes = [max_0, max_1, max_2]
    #when using actual images will need to offset pixels so they are the center of the window
    #pixels = [(0, 0), (1, 0), (0, 1), (1, 1)]
    pixels = [(x, y) for x in range(32) for y in range(32)]
    num_trials = 3
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
            output = gpu_computation(image, kernels, biases, max_sizes, batches, window, streams)
            print "Time so far {0:.4e} seconds".format(time.time()-st)
        tot = time.time()-st
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel {0:.4e} seconds".format(tot/(len(batches)*batchsize*num_trials))
    
    #print output
    #print len(output)

    #out_max = from_serial(conv_max)
    #print out_max-output[0]
    #print out_max, output
    #print np.allclose(output[0], out_max, rtol=1e-04, atol=1e-07) 
    #print np.where(np.isclose(output[0], out_max)==False)

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

def compute_sgemm_batched(cols, kernels, biases, stream, handle, m, k, n):
    batchsize = len(cols)
    #takes gpu arrays of pointers to pointers
    alpha = np.float32(1.0); beta = np.float32(1.0);
    flop = 2*m*n*k
    #print flop 
    cublas.cublasSgemmBatched(handle, 'n', 'n', n, m, k, alpha, cols.ptr, n, kernels.ptr, k, beta, biases.ptr, n, batchsize);

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
    
 

def gpu_computation(image, kernels, biases, max_sizes, batches, window_sizes, streams):
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
    #not currently using streams
    offsets = [] 
    for batch in batches:
        image_ds = [full_image_d]*len(batch)
        for layer_n, (kernel_d, bias_d, max_size, kdim) in enumerate(zip(kernels_d, biases_d, max_sizes, kernel_dims)):
            #only the first layer uses the full image with an offset
            if (layer_n == 0):
                height = window_sizes[0]
                width = window_sizes[1]
                channels = 1
            else:
                #all image_d in image_ds have same shape
                height = image_ds[0].shape[0]
                width = image_ds[0].shape[1]
                channels = image_ds[0].shape[2]
            #print height, width, channels
            #print "layer {0}".format(layer_n)
            #kernel_d = gpu.to_gpu_async(kernel, stream=stream)
            #bias_d = gpu.to_gpu_async(bias, stream=stream)
    
            ksize = kdim[0]
            kchannels = kdim[3]
            height_col = (height + 2 * pad - ksize) / stride + 1
            width_col = (width + 2 * pad - ksize) / stride + 1 
            kernel_d = kernel_d.reshape(kchannels, ksize*ksize*channels)
            
            cols = []
            kernel_ps = []; bias_ps = []; col_ps = [];
            sgemm_biases = []
            for (pix_id, pixel) in enumerate(batch):
                stream = streams[pix_id % nstreams]
                if (layer_n == 0):
                    offset = pixel[0]*full_image_d.shape[1] + pixel[1]
                else:
                    offset = 0
                cols.append(im2col_gpu.compute_im2col(image_ds[pix_id], height, width, channels, np.int32(ksize), np.int32(pad), np.int32(stride), offset, stream))
                col_ps.append(cols[pix_id].ptr)
                #we can use the same kernel in memory for sgemmbatched, but because we change bias
                #we must copy it for each pixel
                kernel_ps.append(kernel_d.ptr)
                copy = bias_d.copy()
                copy = copy.reshape(kernel_d.shape[0], cols[0].shape[1])
                sgemm_biases.append(copy)
                bias_ps.append(copy.ptr)
            
            kernel_ps = gpu.to_gpu(np.array(kernel_ps)); bias_ps = gpu.to_gpu(np.array(bias_ps)); col_ps = gpu.to_gpu(np.array(col_ps));
            m = kernel_d.shape[0]; k = kernel_d.shape[1]; n = cols[0].shape[1];
            compute_sgemm_batched(col_ps, kernel_ps, bias_ps, stream, handle, m, k, n)
            
            sgemm_biases = map(lambda bias: bias.reshape(np.int32(height_col), np.int32(width_col), np.int32(kchannels)), sgemm_biases)
            for (pix_id, pixel) in enumerate(batch):
                 stream = streams[pix_id % nstreams]
                 image_ds[pix_id] = maxpool_gpu.compute_max(sgemm_biases[pix_id], np.int32(max_size), stream) 
        results += image_ds
    results = map(lambda x: x.get(), results)
    cublas.cublasDestroy(handle)
    return results

if __name__ == "__main__":
    main()
    #test_im2col()
