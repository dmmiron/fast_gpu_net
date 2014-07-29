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

    batchsizes = [2**x for x in range(2, 8)]
    batchsizes = [32]
    nstreams = 1
    streams = []
    
    for n in range(nstreams):
        streams.append(cu.Stream())
    #set up test data
    #image = np.float32((np.random.rand(100, 100, 1) - .5) * 2)
    image = np.float32((np.reshape(np.arange(0, 100*100, 1), [100, 100, 1])))
    #image = np.float32((np.reshape(np.arange(0, 10*10, 1), [10, 10, 1])))

    #image = np.float32(np.reshape(np.arange(0, 49*49, 1), [49, 49, 1]))
    ser_image = to_serial(image)
    #kernels, layers per kernel, width, height

    #kernels_0 = np.float32((np.random.rand(4, 4, 1, 64) - .5 ) * 2)
    kernels_0 = np.float32(np.reshape(np.arange(0, 4*4*64, 1), [4, 4, 1, 64]))
    #kernels_0 = np.float32(np.reshape(np.arange(0, 2*2*2, 1), [2, 2, 1, 2]))
    ser_kernels_0 = to_serial(kernels_0)
    bias_0 = np.float32(np.ones((46, 46, 64)))
    ser_bias_0 = to_serial(bias_0)
    max_0 = np.int32((2, 2, 2))

    kernels_1 = np.float32((np.random.rand(4, 4, 32, 64) - .5 ) * 2)
    ser_kernels_1 = to_serial(kernels_1)
    bias_1 = np.float32(np.ones((20, 20, 64)))
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

    kernels = [kernels_0, kernels_1, kernels_2]
    ser_kernels = map(to_serial, kernels)
    biases = [bias_0, bias_1, bias_2]
    ser_biases = map(to_serial, biases)
    max_sizes = [max_0, max_1, max_2]
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
    pixels = [(x, y) for x in range(32) for y in range(32)]
    #pixels = [(x, y) for x in range(3) for y in range(3)]

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
            #TAKE OUT SERIAL LINE WHEN TIMING
            #s_output = batch_serial(ser_image, window, ser_kernels, ser_biases, ser_max_sizes, pad, stride, batches)
            print "Time so far {0:.4e} seconds".format(time.time()-st)
        tot = time.time()-st
        print "Total time: {0:.4e} seconds".format(tot)
        print "Time per pixel {0:.4e} seconds".format(tot/(len(batches)*batchsize*num_trials))

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

def compute_sgemm_batched(cols, kernels, biases, stream, handle, m, k, n):
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
 

def gpu_computation(image, kernels, biases, max_sizes, batches, window_sizes, streams):
    handle = cublas.cublasCreate()
    batchsize = len(batches[0])
    results = []
    result_ps = []
    pad = 0; stride = 1; 
    full_image_d = gpu.to_gpu(image)
    biases_d = []; bias_ps = [];
    kernels_d = []; kernel_ps = [];
    kernel_dims = []
    batch_biases = [];
    for bias, kernel in zip(biases, kernels):
        bias = bias.reshape(1, bias.shape[2], bias.shape[0]*bias.shape[1])
        batch_bias = np.tile(bias, (batchsize, 1, 1))
        batch_bias_d = gpu.to_gpu(batch_bias)
        biases_d.append(batch_bias_d)
        kernels_d.append(gpu.to_gpu(kernel))
        kernel_dims.append(kernel.shape)
    #duplicate the same pointer for each pixel for each layer
    kernel_ps = map(lambda x: [x.ptr]*batchsize, kernels_d)
    kernel_ps_d = gpu.to_gpu(np.array(kernel_ps))
    #get the pointer to each copy of bias for each layer
    bias_p = biases_d[0].ptr
    bias_ps = map(lambda x: [x[idx, :,:].ptr for idx in range(batchsize)] , biases_d)
    bias_ps_d = gpu.to_gpu(np.array(bias_ps));
    #print bias_ps, kernel_ps

    nstreams = len(streams)
    #not currently using streams
    for batch in batches:
        biases_d = []; bias_ps = [];
        kernels_d = []; kernel_ps = [];
        kernel_dims = []
        batch_biases = [];
        for bias, kernel in zip(biases, kernels):
            bias = bias.reshape(1, bias.shape[2], bias.shape[0]*bias.shape[1])
            batch_bias = np.tile(bias, (batchsize, 1, 1))
            batch_bias_d = gpu.to_gpu(batch_bias)
            biases_d.append(batch_bias_d)
            kernels_d.append(gpu.to_gpu(kernel))
            kernel_dims.append(kernel.shape)
        #duplicate the same pointer for each pixel for each layer
        kernel_ps = map(lambda x: [x.ptr]*batchsize, kernels_d)
        kernel_ps_d = gpu.to_gpu(np.array(kernel_ps))
        #get the pointer to each copy of bias for each layer
        bias_p = biases_d[0].ptr
        bias_ps = map(lambda x: [x[idx, :,:].ptr for idx in range(batchsize)] , biases_d)
        bias_ps_d = gpu.to_gpu(np.array(bias_ps));
        #print bias_ps, kernel_ps

        offsets = np.int32(np.zeros(batchsize))
        offsets_d = gpu.to_gpu(offsets)

        image_d = full_image_d
        for layer_n, (kernel_d, bias_d, max_size, kdim) in enumerate(zip(kernels_d, biases_d, max_sizes, kernel_dims)):
            #only the first layer uses the full image with an offset
            if (layer_n == 0):
                height = window_sizes[0]
                width = window_sizes[1]
                channels = 1
            else:
                #all image_d in image_ds have same shape
                height = image_d.shape[2]
                width = image_d.shape[3]
                channels = image_d.shape[1]
            #print height, width, channels
            #print "layer {0}".format(layer_n)
    
            ksize = kdim[0]
            kchannels = kdim[3]
            height_col = (height + 2 * pad - ksize) / stride + 1
            width_col = (width + 2 * pad - ksize) / stride + 1 

            kernel_d = kernel_d.reshape(kchannels, ksize*ksize*channels)
            
            col_ps = [];
            sgemm_biases = biases_d[layer_n] 
            offsets = [];
            for (pix_id, pixel) in enumerate(batch):
                stream = streams[pix_id % nstreams]
                #first layer uses same image and offset within that image
                #later layers have each image concatenated into one image so offset is size of single image
                if (layer_n == 0):
                    offsets.append(pixel[0]*full_image_d.shape[1] + pixel[1])
                else:
                    offsets.append(0)

            #offsets_d = gpu.to_gpu(np.int32(np.array(offsets)));
            col_ps, cols= im2col_gpu.compute_im2col_batched(image_d, height, width, channels, np.int32(ksize), np.int32(pad), np.int32(stride), offsets_d, layer_n, batchsize)
            
            m = kchannels; k = ksize*ksize*channels; n = height_col*width_col;
            compute_sgemm_batched(col_ps, kernel_ps_d[layer_n], bias_ps_d[layer_n], stream, handle, m, k, n)
            sgemm_biases = sgemm_biases.reshape(np.int32(batchsize), np.int32(kchannels), np.int32(height_col), np.int32(width_col))
            image_d = maxpool_gpu.compute_max_batched(sgemm_biases, np.int32(max_size))
            
        #results.append(image_d.get())
        result_ps.append(image_d)
        
    cublas.cublasDestroy(handle)
    return result_ps

if __name__ == "__main__":
    main()
