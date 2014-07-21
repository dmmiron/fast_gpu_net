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

from scipy.signal import convolve as convolve_s
from scipy.ndimage.filters import convolve

def test_im2col():
    image = np.float32(np.reshape(np.arange(0, 50, 1), [5, 5, 2]))
    print to_serial(image)
    image_d = gpu.to_gpu(image)
    for i in range(5):
        for j in range(5):
            for k in range(2):
                print (image[i, j, k]),
            print "\n"
        print "\n"
    print "\n"
    print image
    col = im2col_gpu.compute_im2col(image_d, np.int32(4), np.int32(0), np.int32(1))
    print col


def main():
    #set up test data
    #image = (np.random.rand(1, 49, 49) - .5) * 2
    image = np.float32((np.random.rand(49, 49, 1) - .5) * 2)
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
    conv = comp_convolution(ser_image, ser_kernels_0, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_1, pad, stride)
    conv_max = maxout(conv, 2, 2)
    conv = comp_convolution(conv_max, ser_kernels_2, pad, stride)
    conv_max = maxout(conv, 2, 4)
    conv_max_r = conv_max.ravel()
    result = np.dot(weights, conv_max_r)
    print result
    
    
    kernels = [kernels_0, kernels_1, kernels_2]
    biases = [bias_0, bias_1, bias_2]
    max_sizes = [max_0, max_1, max_2]
    
    #perform parallel computation
    output = gpu_computation(image, kernels, biases, max_sizes)
    out_max = from_serial(conv_max)
    print out_max-output 
    print out_max, output
    print np.allclose(output, out_max, rtol=1e-04, atol=1e-07) 
    print np.where(np.isclose(output, out_max)==False)

def to_serial(array):
    """This method converts the numpy default row-major ordering into an ordering to match the way we are indexing on the gpu. In particular each 2D image slice is indexed in a row-major format (i,j). However, inconsistent with row major ordering we traverse each slice before changing to a new slice. If the array is 4d (kernels) then we traverse the 4th dimension (each 3d kernel last). Thus the indices change (fastest to slowest) in the order column, row, slice, stack"""
    shape_list = [dim for dim in array.shape]
    #reverse dimensions after first two and move to beginning
    #print shape_list[:1:-1], shape_list[:2]
    shape_list = shape_list[:1:-1] + shape_list[:2]
    return array.reshape(shape_list)

def from_serial(array):
    shape_list = [dim for dim in array.shape]
    print shape_list
    print shape_list[-2:], shape_list[-3::-1]
    shape_list = shape_list[-2:] + shape_list[-3::-1]
    return array.reshape(shape_list)

def comp_convolution(image, kernels, pad, stride):
    height = image.shape[1]
    width = image.shape[2]
    #rowsxcolumnsxlayers of kernels

    ksize = kernels.shape[2]

    height_col = (height + 2 * pad - ksize) / stride + 1
    width_col = (width + 2 * pad - ksize) / stride + 1
    print kernels.shape
    conv = np.zeros((kernels.shape[0], height_col, width_col))

    for ki in range(kernels.shape[0]):
        kernel = kernels[ki, :, :, :]
        full_conv2d = np.zeros((height_col, width_col)) 
        for layer in range(kernels.shape[1]):
            kernel_layer = kernel[layer, :, :]
            image_layer = image[layer, :, :]
            #NOTE THE REVERSED KERNEL
            #STILL NOT CLEAR WHY THIS SHOULD BE NECESSARY
            full_conv2d += convolve_s(image_layer, kernel_layer[::-1, ::-1], mode='valid')
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

def compute_sgemm(col, kernel, bias):
    alpha = np.float32(1.0); beta = np.float32(1.0);
    blocksize = (1, 1, 1)
    gridsize = (1, 1, 1)

    #(mxk)x(kxn)
    m = np.int32(kernel.shape[0])
    k = np.int32(kernel.shape[1]) 
    n = np.int32(col.shape[1])

    handle = cublas.cublasCreate()
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, col.gpudata, n, kernel.gpudata, k, beta, bias.gpudata, n);
    cublas.cublasDestroy(handle)
 

def gpu_computation(image, kernels, biases, max_sizes):
    pad = 0; stride = 1; 
    #print image, kernels[0], biases[0]
    print image.shape, kernels[0].shape, biases[0].shape
    image_d = gpu.to_gpu(image)
    
    for (kernel, bias, max_size) in zip(kernels, biases, max_sizes):
        kernel_d = gpu.to_gpu(kernel)
        bias_d = gpu.to_gpu(bias)

        ksize = kernel.shape[0]
        kchannels = kernel.shape[3]
        height_col = (image_d.shape[0] + 2 * pad - ksize) / stride + 1
        width_col = (image_d.shape[1] + 2 * pad - ksize) / stride + 1 
        print kernel_d.shape, ksize, kchannels, image_d.shape 
        kernel_d = kernel_d.reshape(kchannels, ksize*ksize*image_d.shape[2])
        #print image_d
        result = im2col_gpu.compute_im2col(image_d, np.int32(ksize), np.int32(pad), np.int32(stride))
        print bias_d.shape, kernel_d.shape, result.shape
        bias_d = bias_d.reshape(kernel_d.shape[0], result.shape[1])
        print "bias shape", bias_d.shape

        compute_sgemm(result, kernel_d, bias_d)
        bias_d = bias_d.reshape(np.int32(height_col), np.int32(width_col), np.int32(kchannels))
        image_d = maxpool_gpu.compute_max(bias_d, np.int32(max_size)) 
    return image_d.get()

if __name__ == "__main__":
    st = time.time()
    main()
    print time.time()-st
