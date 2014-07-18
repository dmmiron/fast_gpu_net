import numpy as np
import im2col as im2col_gpu
import maxpool as maxpool_gpu

import scikits.cuda.cublas as cublas
import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit


from scipy.ndimage.filters import convolve


def main():
    #image = np.reshape(np.arange(1, 51), [2, 5, 5])
    #image = np.ones((1, 49, 49))
    image = (np.random.rand(1, 49, 49) - .5) * 2
    f = open("arrays", "w")
    image.tofile(f, sep=",", format="%f")
    #kernels, layers per kernel, width, height
    #kernels_1 = np.ones([64, 1, 4, 4])
    #kernels_2 = np.ones([4, 32, 4, 4])
    kernels_1 = (np.random.rand(64, 1, 4, 4) -.5 ) * 2

    kernels_2 = (np.random.rand(64, 32, 4, 4) -.5 ) * 2
    kernels_3 = (np.random.rand(128, 32, 5, 5) - .5) * 2
    weights = np.random.rand(2, 288)
    print "image\n", image[0, :, :]
    pad = 0
    stride = 1
    conv = comp_convolution(image, kernels_1, pad, stride)
    conv_max = maxout(conv, 2, 2)
    print conv_max.shape
    print conv_max
    conv = comp_convolution(conv_max, kernels_2, pad, stride)
    conv_max = maxout(conv, 2, 2)
    print conv_max.shape
    print conv_max
    conv = comp_convolution(conv_max, kernels_3, pad, stride)
    conv_max = maxout(conv, 2, 4)
    print conv_max.shape
    print conv_max
    conv_max = conv_max.ravel()
    result = np.dot(weights, conv_max)
    print result


def comp_convolution(image, kernels, pad, stride):
    full_conv = np.zeros((kernels.shape[0], image.shape[1], image.shape[2]))
    height = image.shape[2]
    width = image.shape[1]

    ksize = kernels.shape[3]

    height_col = (height + 2 * pad - ksize) / stride + 1
    width_col = (width + 2 * pad - ksize) / stride + 1

    for ki in range(kernels.shape[0]):
        kernel = kernels[ki, :, :, :]
        full_conv2d = np.zeros((image.shape[1], image.shape[2])) 
        for layer in range(kernels.shape[1]):
            kernel_layer = kernel[layer, :, :]
            image_layer = image[layer, :, :]
            full_conv2d += convolve(image_layer, kernel_layer, mode='constant', cval=0.0)
        full_conv[ki, :, :] = full_conv2d
    conv = full_conv[:, :width_col, :height_col]
    return conv

def maxout(image, max_ksize_xy, max_ksize_z):
    height = image.shape[2]
    width = image.shape[1]
    depth = image.shape[0]
    out_height = (height+max_ksize_xy-1)/max_ksize_xy;
    out_width = (width+max_ksize_xy-1)/max_ksize_xy;
    out_depth = (depth+max_ksize_z-1)/max_ksize_z;

    output = np.zeros((out_depth, out_width, out_height))
    for k in range(out_depth):
        for j in range(out_width):
            for i in range(out_height):
                image_i = i*max_ksize_xy
                image_j = j*max_ksize_xy
                image_k = k*max_ksize_z

                #output[i, j] = np.amax(image[image_i:image_i+max_ksize, image_j:image_j+max_ksize].ravel())
                output[k, j, i] = np.amax(image[image_k:image_k+max_ksize_z, image_j:image_j+max_ksize_xy, image_i:image_i+max_ksize_xy].ravel())

    return output

def compute_sgemm(col, kernel, bias):
    alpha = np.float32(1.0); beta = np.float32(1.0);
    blocksize = (1, 1, 1)
    gridsize = (1, 1, 1)

    #(mxk)x(kxn)
    m = np.int32(kernel.shape[0])
    k = np.int32(kernel.shape[1]) 
    n = np.int32(col.shape[1])
    print kernel
    print col
    handle = cublas.cublasCreate()
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, col.gpudata, n, kernel.gpudata, k, beta, bias.gpudata, n);
    cublas.cublasDestroy(handle)
 

def gpu_computation():
    #image = np.float32(np.reshape(np.arange(0, 324, 1), [9, 9, 4]))
    image = np.float32(np.reshape(np.arange(0, 25, 1), [5, 5, 1]))
    print image
    image_d = gpu.to_gpu(image)
    ksize = 4; pad = 0; stride = 1;
    height_col = (image.shape[0] + 2 * pad - ksize) / stride + 1
    width_col = (image.shape[1] + 2 * pad - ksize) / stride + 1 
    kchannels = 2
    result = im2col_gpu.compute_im2col(image_d, ksize, pad, stride)
    kernel = np.float32(np.random.rand(kchannels, ksize*ksize*image.shape[2]))
    kernel = np.float32(np.ones((kchannels, ksize*ksize*image.shape[2])))
    kernel_d = gpu.to_gpu(kernel)
    bias = np.float32(np.zeros((kernel.shape[0], result.shape[1])))
    bias_d = gpu.to_gpu(bias)
    compute_sgemm(result, kernel_d, bias_d)
    print height_col, width_col, kchannels, bias_d.shape
    bias_d = bias_d.reshape(height_col, width_col, kchannels)
    output = maxpool_gpu.compute_max(bias_d, (2, 2, 2))
    print output.get()


if __name__ == "__main__":
    main()
    gpu_computation()
