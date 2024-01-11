# Source: On Improving the Numerical Stability of Winograd Convolutions
# By: Kevin Vincent and Kevin J. Stephano and Michael A. Frumkin and Boris Ginsburg and Julien Demouth
# (https://openreview.net/pdf?id=H1ZaRZVKg)
# GitHub: Deep-Learning-From-Scratch
# (https://github.com/Pabloo22/dlfs)

import tensorflow as tf
import numpy as np
from typing import Tuple, List, Union
import pathlib
import matplotlib.pyplot as plt


class WinogradConv2D:
    """ This class implements the Winograd convolution for 2D convolutions. The Winograd convolution is a convolution
    that uses the Winograd algorithm to reduce the number of multiplications needed to perform the convolution in
    exchange for more additions. It uses the same principle as the FFT convolution, but without the existence of
    complex numbers, whose products are four times more expensive than with real numbers. 
    
    The Winograd convolution is implemented in the following way: 

    - The image is divided into patches of size patch_size, with an overlap of kernel_size - 1 between patches
    (this is the minimum overlap to not lose information). Extra padding is added to the image if necessary to
    ensure all patches are the same size, and the padding is removed after the convolution.
    - Matrices Y, X and W are calculated for each dimension, which are used to transform both the patch and 
    the kernel into a new domain. It is worth noting that these matrices are calculated only once for each 
    patch and filter size, so it is not necessary to calculate them every time a convolution is performed.
    - The patch is transformed using X, the kernel is transformed using W and, since both are in the frequency 
    domain, they are multiplied using the Hadamard product (element-wise multiplication). 
    - The result is transformed back to the spatial domain using Y. 
    - The patches are concatenated to form the output image. 
    
    The Winograd convolution has the following advantages: 

    - It can be faster than traditional convolution for small kernel sizes and devices with low computational 
    power, since it sacrifices multiplications for additions, which are cheaper (unless you are using a GPU or
    your CPU supports SIMD instructions (Single Instruction, Multiple Data), for products, which could make the
    traditional vecorized convolution faster).
    - It is faster than FFT convolution, since, as mentioned before, complex multiplications are four times more 
    expensive than real multiplications.

    The Winograd convolution has the following disadvantages: 

    - Since we are using points to calculate the matrices,
    the matrices are not exact, and the error increases with the size of the matrix. This is not a problem for small
    matrices, but it can be a problem for large matrices.
    - The Winograd convolution cannot be used with big patch sizes (no greater than around 10), since the matrices 
    would be numerically unstable.

    Args:
        image_size (Tuple[int, int]): The size of the image. It must be a 2D image.
        kernel_size (Tuple[int, int]): The size of the kernel. It must be a 2D kernel.
        patch_size (Tuple[int, int]): The size of the patch. It must be smaller than the kernel size.
        strides (Tuple[int, int]): The strides of the convolution in each dimension. The default value is (1, 1)
        padding (str): The padding of the convolution. It can be 'valid' or 'same'. The default value is 'valid'
        div_points (int): The number of points to be used in the Vandermonde Matrix. The default value is 2, but for
            some big patch sizes, it is recommended to use 4 instead.
    """

    def __init__(self,
                 image_size: Tuple[int, int],
                 kernel_size: Tuple[int, int],
                 patch_size: Tuple[int, int],
                 strides: Tuple[int, int] = (1, 1),
                 padding: str = 'valid',
                 div_points: int = 2) -> None:
        self.strides = np.array(strides)
        self.image_size = np.array(image_size) // self.strides
        self.kernel_size = np.array(kernel_size)
        self.patch_size = np.array(patch_size)
        self.padding = np.array(padding)
        self.div_points = div_points
        self.image_size_h = self.image_size[0]
        self.image_size_w = self.image_size[1]
        self.kernel_size_h = self.kernel_size[0]
        self.kernel_size_w = self.kernel_size[1]
        self.patch_size_h = self.patch_size[0]
        self.patch_size_w = self.patch_size[1]

        if self.image_size_h < self.kernel_size_h or self.image_size_w < self.kernel_size_w:
            raise ValueError("The kernel size must be smaller than the image size")

        if self.patch_size_h < self.kernel_size_h or self.patch_size_w < self.kernel_size_w:
            raise ValueError("The patch size must be greater than the kernel size")

        self.output_patch_size_h = self.patch_size_h - self.kernel_size_h + 1
        self.output_patch_size_w = self.patch_size_w - self.kernel_size_w + 1
        self.num_patches_h = self.calculate_num_patches(self.image_size_h, self.kernel_size_h, self.patch_size_h)
        self.num_patches_w = self.calculate_num_patches(self.image_size_w, self.kernel_size_w, self.patch_size_w)
        self.pad_h = self.calculate_padding(self.image_size_h, self.kernel_size_h, self.patch_size_h,
                                            self.num_patches_h)
        self.pad_w = self.calculate_padding(self.image_size_w, self.kernel_size_w, self.patch_size_w,
                                            self.num_patches_w)
        self.Y_h, self.X_h, self.W_h = self.get_winograd_matrices(self.output_patch_size_h, self.kernel_size_h)
        self.Y_w, self.X_w, self.W_w = self.get_winograd_matrices(self.output_patch_size_w, self.kernel_size_w)

        self.concat_h = None
        self.concat_w = None

    @staticmethod
    def calculate_num_patches(image_size: int, kernel_size: int, patch_size: int) -> int:
        """ Calculate the number of patches for a given image size, kernel size and patch size This code works by
        calculating, on the desired dimension, how many times can you make non-overlapping patches of size
        patch_size-overlap, and then adding 1 to account for the first full patch, which will be overlapping with the
        next one.

        Args:
            image_size (int): The size of the image
            kernel_size (int): The size of the kernel
            patch_size (int): The size of the patch
        Returns:
            int: The number of patches"""
        overlap = kernel_size - 1
        if patch_size - overlap == 1:
            return image_size
        if overlap >= patch_size:
            raise ValueError("The overlap must be strictly smaller than the patch size")
        if image_size == patch_size:
            return 1
        num_patches = (image_size - patch_size) // (patch_size - overlap) + 1

        return num_patches

    @staticmethod
    def calculate_padding(image_size: int, kernel_size: int, patch_size: int, num_patches: int) -> int:
        """ Calculate the padding for a given image size, kernel size, patch size and number of patches This code
        works by calculating, on the desired dimension, how many lists of size patch_size-overlap can you make,
        and we add the patch_size to account for the last patch, and then we subtract the image_size to get the padding.

        Args:
            image_size (int): The size of the image
            kernel_size (int): The size of the kernel
            patch_size (int): The size of the patch
            num_patches (int): The number of patches
        Returns:
            int: The padding"""
        overlap = kernel_size - 1
        if patch_size - overlap == 1:
            return 0
        if overlap >= patch_size:
            raise ValueError("The overlap must be strictly smaller than the patch size")
        if image_size == patch_size:
            return 0

        padding = max(0, (num_patches * (patch_size - overlap) + patch_size) - image_size)

        return padding

    def get_winograd_matrices(self, output_patch_size: int, kernel_size: int) -> Tuple[np.ndarray,
                                                                                       np.ndarray,
                                                                                       np.ndarray]:
        """This is equivalent to F(m, n) on the paper. This function returns the matrices necessary for calculating
        the Winograd convolution given the dimension of the filter and the dimension of the output. With those,
        we can infer the size of the input. The usage of these matrices depends on the dimension of the convolution.
        If we denote image as the signal and w as the filter: 
        - If it is 1D: F(m, n) = image * w = Y.T @ [(X.T @ image) * (W @ w)] 
        - If it is 2D: F(m x i, n x j) = image * w = Y.T @ [(X.T @ image @ X') * (W @ w @ W'.T)] @ Y' 
        In the latter case, if m = i and n = j, then Y = Y', X = X', W = W'. If not, then you would have
        to calculate both F(m, n) to get Y, X, W and F(i, j) to get Y', X', W'.

        Args:
            output_patch_size (int): The size of the output
            kernel_size (int): The size of the filter

        Returns:
            A tuple with the aforementioned Y.T, X, W in this order."""
        num_points = output_patch_size + kernel_size - 1
        points = self.get_points(num_points)
        Y = WinogradConv2D.get_vandermonde_matrix(output_patch_size, points)
        X = np.linalg.inv(WinogradConv2D.get_vandermonde_matrix(num_points, points))
        W = WinogradConv2D.get_vandermonde_matrix(kernel_size, points)
        return Y, X, W

    def get_points(self, num_points: int) -> np.ndarray:
        """This function generates homogeneous coordinates, starting from (0, 1), then, (1/2, 1), then (-1/2, 1),
        and the infinite point (0, 1). [(f_0, g_0), ..., (f_a-1, g_a-1)] This code is a complex version for
        efficiency's sake of the following code:

        >>> points = [(0, 1)] + [(y/self.div_points, 1) for x in range(1, num_points // 2) for y in (x, -x)]
        >>> if num_points % 2:
        >>> points.append((num_points // 2, 1))
        >>> points.append((1, 0))
        >>> return points

        Args:
            num_points (int): The number of points to be generated
        Returns:
            np.ndarray: The points for the Vandermonde Matrix"""
        half_points = np.arange(1, num_points // 2)
        points_x = np.concatenate([half_points, -half_points]) / self.div_points
        points_y = np.ones_like(points_x)
        points = np.column_stack([points_x, points_y])
        # Add special cases (0,1) and (1,0)
        points = np.vstack([[0, 1], points, [num_points // 2, 1], [1, 0]]) if num_points % 2 else np.vstack(
            [[0, 1], points, [1, 0]])

        return points

    @staticmethod
    def get_vandermonde_matrix(b: int, points: np.ndarray) -> np.ndarray:
        """This is equivalent to V_{axb} on the document.
        This function returns a trimmed Vandermonde Matrix of size len(points) image b

        f_0^0 * g_0^b-1       f_0^1 * g_0^b-2       ... f_0^b-1 * g_0^0
        .
        .
        f_a-1^0 * g_a-1^b-1   f_a-1^1 * g_a-1^b-2   ... f_a-1^b-1 * g_a-1^0
        This code is a complex version for efficiency's sake of the following code:

        >>> return np.array([[i[0] ** j * i[1] ** (b - j - 1) for j in range(b)] for i in points])

        Args:
            b (int): The number of columns of the Vandermonde Matrix
            points (np.ndarray): The points to be used in the Vandermonde Matrix
        Returns:
            np.ndarray: The Vandermonde Matrix
        """
        n = len(points)
        powers = np.arange(b)
        V = np.empty((n, b), dtype=float)

        for i, (x, y) in enumerate(points):
            V[i, :] = np.power(x, powers) * np.power(y, b - 1 - powers)

        return V

    def get_patches(self, image: np.ndarray) -> np.ndarray:
        """ Get the patches with the given patch size and required overlap so that the patches can be convolved with
        the kernel using Winograd convolution without losing information.

            Args:
                image (np.ndarray): The image to be patched
            Returns:
                np.ndarray: The patches"""
        image = tf.reshape(image, [1, self.image_size_h, self.image_size_w, 1])
        padded_image = tf.pad(image, [[0, 0], [0, self.pad_h], [0, self.pad_w], [0, 0]])
        overlap = np.array(self.kernel_size) - 1
        stride = np.array(self.patch_size) - overlap
        patches = tf.image.extract_patches(padded_image, sizes=[1, *self.patch_size, 1], strides=[1, *stride, 1],
                                           rates=[1, 1, 1, 1], padding='VALID')
        self.concat_h, self.concat_w = patches.shape[1:3]
        patches = tf.reshape(patches, [np.prod(patches.shape[1:3]), *self.patch_size])

        return patches.numpy()

    def convolve_patches(self, patches: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Convolve the patches with the kernel using Winograd convolution
            This code implements the Winograd convolution in a vectorized way for efficiency's sake

            Args:
                patches (np.ndarray): The patches to be convolved
                kernel (np.ndarray): The kernel to convolve the patches with
            Returns:
                np.ndarray: The convolved patches"""
        # Ensure patches and kernel have an extra dimension for batch processing
        patches = np.expand_dims(patches, axis=0)
        kernel = np.expand_dims(kernel, axis=0)

        # Perform the Winograd convolution for each patch in the batch
        output = self.Y_h.T @ ((self.X_h.T @ patches @ self.X_w) * (self.W_h @ kernel @ self.W_w.T)) @ self.Y_w

        # Remove the extra dimension
        return np.squeeze(output, axis=0)

    def concat_patches(self, patches: np.ndarray) -> np.ndarray:
        """ Concatenate the patches into a single image

            This code is a complex version for efficiency's sake of the following code:

            >>> hstacks = (patches[i * self.concat_w : (i + 1) * self.concat_w] for i in range(self.concat_h))
            >>> output = np.vstack([np.hstack(hstack) for hstack in hstacks])
            >>> return output
            
            Args:
                patches (np.ndarray): The patches to be concatenated
            Returns:
                np.ndarray: The concatenated image"""
        # Reshape patches to a 3D array
        reshaped_patches = patches.reshape(self.concat_h, self.concat_w, self.output_patch_size_h,
                                           self.output_patch_size_w)

        # Reorganize the patches
        # Transpose to align the patches correctly
        # Reshape to flatten the patches into a 2D array
        output = reshaped_patches.transpose(0, 2, 1, 3).reshape(self.concat_h * self.output_patch_size_h,
                                                                self.concat_w * self.output_patch_size_w)
        if self.padding == 'valid':
            # Remove padding added by patching if present
            if self.pad_h:
                output = output[:-self.pad_h]
            if self.pad_w:
                output = output[:, :-self.pad_w]
        elif self.padding == 'same':
            output = output[:self.image_size[0], :self.image_size[1]]

        return output

    def stride_image(self, image: np.ndarray) -> np.ndarray:
        """ Stride the image according to the strides
            Args:
                image (np.ndarray): The image to be strided
            Returns:
                np.ndarray: The strided image"""
        image = image[::self.strides[0], ::self.strides[1]]
        return image
    
    def pad_image(self, image: np.ndarray) -> np.ndarray:
        """ Pad the image according to the padding
            Args:
                image (np.ndarray): The image to be padded
            Returns:
                np.ndarray: The padded image"""
            # Pad the image so that the output has the same size as the input.
        pad_same_h = (self.image_size_h - 1) * self.strides[0] + self.kernel_size_h - self.image_size_h
        pad_same_w = (self.image_size_w - 1) * self.strides[1] + self.kernel_size_w - self.image_size_w
        image = tf.pad(image, [(pad_same_h // 2, pad_same_h // 2), (pad_same_w // 2, pad_same_w // 2)])
        self.image_size_h, self.image_size_w = image.shape
        return image

    def __call__(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Perform the Winograd convolution on the image with the kernel
            Args:
                image (np.ndarray): The image to be convolved
                kernel (np.ndarray): The kernel to convolve the image with
            Returns:
                np.ndarray: The convolved image"""
        if not all(self.strides == 1):
            image = self.stride_image(image)
        if self.padding == 'same':
            image = self.pad_image(image)

        patches = self.get_patches(image)
        output = self.convolve_patches(patches, kernel)
        output = self.concat_patches(output)
        return output
    

    
class WinogradConv3D(WinogradConv2D):
    """ This class implements the Winograd convolution for 3D convolutions. This implementation is based on the
    Winograd convolution for 2D convolutions, but it is adapted to 3D (RGB) convolutions. This implementation is not 
    optimized for efficiency, since it is not vectorized for the channel dimension, instead, it is implemented
    by convolving each channel separately and then concatenating the results. This is done because the Winograd
    convolution, although in theory it can be used for 3D convolutions, on my tests it was not stable, and it
    produced numbers that were too big.

    Theoretically, it can be implemented in the following way:
    $Y=[(\mathcal{G} \times_{n=1}^N W) \odot (\mathcal{D} \times_{n=1}^N X)] \times_{n=1}^N Y^T$

    With the following definitions:
    - $\mathcal{G} is the multi-dimensional input tensor, which is the image in this case.
    - $\mathcal{D} is the multi-dimensional filter tensor, which is the kernel in this case.
    - $\mathcal{G} \times_{n=1}^N G$ is short for $\mathcal{G} \times_1 G_1\times_2 G_2\dots \times_N G_N,$
    - $\times_n$ represents the n-mode product, also known as the tensor-matrix n-mode product, which is a generalization
    of the matrix-vector product to tensors, which is done by first flattening the tensor along the n-th dimension,
    giving you a matrix of size $I_n \times \prod_{i \neq n} I_i$, where $I_i$ is the size of the i-th dimension of the
    tensor, then multiply the flattened tensor with the matrix, and then reshaping the result to the original shape of the
    tensor, but with the n-th dimension replaced by the second dimension of the matrix.

    The generation of the matrices Y, X and W is the same as in the 2D case, for each dimension they can be different,
    depending on the size of the kernel and the output.

    All of this can be done using tensorly.tenalg.multi_mode_dot, but it is not implemented here because it is not stable.

    The implementation of the Winograd convolution for 3D convolutions is done in the following way:
    
    >>> def winograd_chunk_3D(patch, kernel, winograd_matrices):
    >>>     part1 = tensorly.tenalg.multi_mode_dot(patch, [t[1] for t in winograd_matrices], list(range(chunk.ndim)))
    >>>     part2 = tensorly.tenalg.multi_mode_dot(kernel, [t[2] for t in winograd_matrices], list(range(filter.ndim)))
    >>>     part3 = part1 * part2
    >>>     return tensorly.tenalg.multi_mode_dot(part3, [t[0] for t in winograd_matrices], list(range(part3.ndim)))

    With winograd_matrices being a list of tuples of the form (Y, X, W), where Y, X and W are the tranform matrices 
    for each dimension.


    
    Args:
        image_size (Tuple[int, int, int]): The size of the image. It must be a 3D image.
        kernel_size (Tuple[int, int, int]): The size of the kernel. It must be a 3D kernel.
        patch_size (Tuple[int, int, int]): The size of the patch. It must be smaller than the kernel size.
        strides (Tuple[int, int]): The strides of the convolution in each dimension. The default value is (1, 1)
        padding (str): The padding of the convolution. It can be 'valid' or 'same'. The default value is 'valid'
        div_points (int): The number of points to be used in the Vandermonde Matrix. The default value is 2, but for
            some big patch sizes, it is recommended to use 4 instead.
    """
    
    def __init__(self,
                image_size: Tuple[int, int, int],
                kernel_size: Tuple[int, int, int],
                patch_size: Tuple[int, int, int],
                strides: Tuple[int, int] = (1, 1),
                padding: str = 'valid',
                div_points: int = 2) -> None:
        super().__init__(image_size[:-1], kernel_size[:-1], patch_size, strides, padding, div_points)

    @staticmethod
    def split_channels(image: np.ndarray) -> np.ndarray:
        """ Given the image, split the channels
            Args:
                image (np.ndarray): The image to be concatenated
            Returns:
                np.ndarray: The concatenated image"""
        split = np.split(image, image.shape[-1], axis=-1)
        # Drop the last dimension
        return [np.squeeze(channel, axis=-1) for channel in split]

    @staticmethod
    def inverse_split_channels(image: np.ndarray) -> np.ndarray:
        """ Inverse the split
            Args:
                image (np.ndarray): The image to be concatenated
            Returns:
                np.ndarray: The concatenated image"""
        return np.dstack(image)

    def convolve(self, channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Perform the Winograd convolution on a single channel of the image with the kernel
            Args:
                channel (np.ndarray): The channel of the image to be convolved
                kernel (np.ndarray): The kernel to convolve the image with
            Returns:
                np.ndarray: The convolved channel"""
        return super().__call__(channel, kernel)

    def __call__(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """ Perform the Winograd convolution on the image with the kernel
            Args:
                image (np.ndarray): The image to be convolved
                kernel (np.ndarray): The kernel to convolve the image with
            Returns:
                np.ndarray: The convolved image"""
        channels_list = self.split_channels(image)
        # print(channels_list[0].shape)
        kernel_list = self.split_channels(kernel)
        if not all(self.strides == 1):
            channels_list = [self.stride_image(channel) for channel in channels_list]
        if self.padding == 'same':
            channels_list = [self.pad_image(channel) for channel in channels_list]

        patches_list = [self.get_patches(channel) for channel in channels_list]
        convolved_patches = [self.convolve_patches(patch, kernel) for patch, kernel in zip(patches_list, kernel_list)]
        convolved_channels = [self.concat_patches(convolved_patch) for convolved_patch in convolved_patches]
        convolved_image = self.inverse_split_channels(convolved_channels)
        return convolved_image
    

def convolve_with_winograd_and_tf(img: np.ndarray, 
                                  kernel: np.ndarray, 
                                  patch_size: Tuple[int, int] = (7, 7), 
                                  strides: Tuple[int, int] = (1, 1), 
                                  plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """ Convolve the image with the kernel using Winograd convolution and TensorFlow convolution
        Args:
            img (np.ndarray): The image to be convolved
            kernel (np.ndarray): The kernel to convolve the image with
            patch_size (Tuple[int, int]): The size of the patch. The default value is (7, 7)
            strides (Tuple[int, int]): The strides of the convolution in each dimension. The default value is (1, 1)
            plot (bool): Whether to plot the results. The default value is True
        Returns:
            Tuple[np.ndarray, np.ndarray]: The convolved image using Winograd convolution and TensorFlow convolution"""
    winograd_conv = WinogradConv2D(img.shape, kernel.shape, patch_size, strides=strides)
    img_conv = winograd_conv(img, kernel)
    img_reshape = img.reshape(1, *img.shape, 1)
    kernel_reshape = kernel.reshape(*kernel.shape, 1, 1)
    img_conv_tf = tf.nn.conv2d(img_reshape, kernel_reshape, strides=[1, *strides, 1], padding='VALID').numpy().reshape(img_conv.shape)
    
    allclose = np.allclose(img_conv, img_conv_tf)
    print(f'Are the results equal? {allclose}')
    if not allclose:
        print(f'The maximum difference is {np.max(np.abs(img_conv - img_conv_tf))} > {1e-08 + 1e-05 * np.max(np.abs(img_conv_tf))}')

    if plot:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(img_conv, cmap='gray')
        plt.title('Winograd')
        plt.subplot(1, 2, 2)
        plt.imshow(img_conv_tf, cmap='gray')
        plt.title('TensorFlow')
        plt.show()
    return img_conv, img_conv_tf

    
if __name__ == '__main__':
    # We download the flowers dataset from TensorFlow Datasets
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = pathlib.Path(data_dir).with_suffix('')
    roses = list(data_dir.glob('roses/*'))

    # We select the first image
    img = tf.keras.preprocessing.image.load_img(roses[0])
    # We normalize the image
    img_array_rgb = tf.keras.preprocessing.image.img_to_array(img) / 255.0

    # We create a Gaussian blur kernel
    gaussian_blur_5x5 = np.array([[1, 4, 6, 4, 1],
                              [4, 16, 24, 16, 4],
                              [6, 24, 36, 24, 6],
                              [4, 16, 24, 16, 4],
                              [1, 4, 6, 4, 1]]) / 256.0
    # We add another dimension to the kernel for the channels
    gaussian_blur_5x5 = np.stack([gaussian_blur_5x5] * 3, axis=-1)

    # We create the Winograd convolution object with the desired parameters, similar to the Conv2D object
    winograd_conv3d = WinogradConv3D(img_array_rgb.shape, gaussian_blur_5x5.shape, (7, 7), padding='same')

    # We perform the Winograd convolution
    sharpen_concat_conv = winograd_conv3d(img_array_rgb, gaussian_blur_5x5)

    # We plot the original image and the blurred image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array_rgb, cmap='gray')
    plt.title('Original image')
    plt.subplot(1, 2, 2)
    plt.imshow(np.clip(sharpen_concat_conv, 0, 1), cmap='gray') # We clip the values to be between 0 and 1 just in case
    plt.title('Blurred image')
    plt.show()