#!/usr/bin/env python
import numpy as np
import sys

import pycuda.compiler as nvcc
import pycuda.driver as cu
import pycuda.gpuarray as gpu
import pycuda.autoinit

import scikits.cuda.cublas as cublas


def compute_sgemm(col, kernel, bias, stream, handle):
    alpha = np.float32(1.0); beta = np.float32(1.0);

    m = np.int32(kernel.shape[0])
    k = np.int32(kernel.shape[1])
    n = np.int32(col.shape[1])

    flop = 2*m*n*k
    #cublas.cublasSetStream(handle, stream.handle)
    cublas.cublasSgemm(handle, 'n', 'n', n, m, k, alpha, col.ptr, n, kernel.ptr, k, beta, bias.ptr, n);

def compute_sgemm_batched(cols, kernels, biases, m, k, n, stream, handle):
    batchsize = len(cols)
    alpha = np.float32(1.0); beta = np.float32(1.0);

    flop = 2*m*n*k
    cublas.cublasSgemmBatched(handle, 'n', 'n', n, m, k, alpha, cols.ptr, n, kernels.ptr, k, beta, biases.ptr, n, batchsize)


def main():
    m = 64; k = 512; n = 400;
    #m = 2; k = 3; n = 4;
    handle = cublas.cublasCreate()
    _, narrays, batchsize = sys.argv
    narrays = int(narrays); batchsize = int(batchsize);
    
    cols = []; kernels = []; biases = [];
    pcols = []; pkernels = []; pbiases= []; #lists to stores pointers to gpu arrays
    kernel = np.float32((np.random.rand(m, k) -.5) * 2)
    kernel = np.float32(np.reshape(np.arange(0, m*k, 1), [m, k]))
    for i in range(narrays):
        col = np.float32((np.random.rand(k, n) - .5) * 2)
        #col = np.float32(np.reshape(np.arange(0, k*n, 1), [k, n]))
        bias = np.float32(np.zeros((m, n)))
        col_d = gpu.to_gpu(col)
        kernel_d = gpu.to_gpu(kernel)
        bias_d = gpu.to_gpu(bias)
        cols.append(col_d); kernels.append(kernel_d); biases.append(bias_d);
        pcols.append(col_d.ptr); pkernels.append(kernel_d.ptr); pbiases.append(bias_d.ptr);
    pcols = np.array(pcols); pkernels = np.array(pkernels); pbiases = np.array(pbiases); 
    pcols_d = gpu.to_gpu(pcols); pkernels_d = gpu.to_gpu(pkernels); pbiases_d = gpu.to_gpu(pbiases);
    
    for i in range(narrays):
        compute_sgemm(cols[i], kernels[i], biases[i], 0, handle);
    #zero out arrays for checking results
    #for i in range(narrays):
        #print biases[i]
    #    biases[i] -= biases[i]
    print "\n\n"
    for i in range((narrays+batchsize-1)/batchsize):
        start = i*batchsize
        compute_sgemm_batched(pcols_d[start:start+batchsize], pkernels_d[start:start+batchsize], pbiases_d[start:start+batchsize], m, k, n, 0, handle)
    #for i in range(narrays):
    #    print biases[i]
    cublas.cublasDestroy(handle)




if __name__ == "__main__":
    main()
