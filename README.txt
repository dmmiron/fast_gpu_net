fast_gpu_net contains two gpu implementations of neural networks for pixel
classification in the rhoana pipeline. Convolutional contains an
implementation of a deep convolutional neural network, while fully_connected
contains an implementation of a deep fully connected network.

Convolutional:
main.py contains the overall architecture of the network. It can be run by
specifying the name of an input folder, an output folder and a model file.
Each image will then be classified and a new image will be saved to the output
folder with the format <input_name>_classified.tif. 
Note:The main function can be used as a test function with random arrays, but may not currently be in working state.

The network calls three gpu kernels, im2col_batched, sgemm_batched, and
maxpool_batched for each layer. 

im2col_batched: im2col_batched is a modified version of the caffe
implementation of im2col from Berkeley Vision and Learning Center. Given a set
of images (already on the gpu) and a kernel size it performs im2col on the
gpu.

sgemm_batched:  sgemm_batched is a cublas call made using cuda scikits
wrapper. It performs C = A*B + C for arrays of matrices A, B and C.

maxpool_batched: maxpool_batched is a gpu kernel to compute the maximum value
in a 3D block for a set of images.

Each of the kernels has a corresponding python wrapper that sets up the cuda
block and grid dimensions and calls the actual kernel.

To reduce cost from memory transfer these wrappers operate on handles to gpu
arrays and the only host to device or device to host transfers occur at the
beginning and end of classifying an image.


ARRAY INDICES:
Special attention should be paid to the order of indices used in the images.
Because of inconsitencies between the row-ordered C/numpy memory layouts and
the column-order gpu layout as well as between image indices and coordinate
indices, the order of dimensions is not obvious and not trivially
generalizable.


Fully Connected:
main.py performs classification like in the convolutional networks with the
same interface. 
The main function is a test function set up to run timing and check that the output matches with output performed
using pylearn2 on the smae image and using the same model.

Image Formats: 
The input images are cast to floats and normalized and the output images are 8
bit integer images.


The fully connected network calls four kernels, im2col, sgemm, rectify, and soft_max.

For this network im2col and sgemm are used unbatched.

Rectify simply computes elementwise max(0, x)

soft_max is applied only in the last layer to convert 2D output to probability
values for the classes.

