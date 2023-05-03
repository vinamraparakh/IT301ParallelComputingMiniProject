import numpy as np
import cv2
from matplotlib import pyplot as plt
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time

start_time = time.time()

# Load the image
img = cv2.imread('chess_board_2.jpeg', 0)

# Perform image cropping
height = img.shape[0]
width = img.shape[1]
img = img[3:height-10, 5:width-10]
plt.imshow(img, cmap='gray')
height, width = img.shape

print("Image loading and cropping successful")
print("Shape of the image is ",img.shape)


img = img.astype(np.float32)
# Allocate memory for the images on the GPU
img_gpu = cuda.to_device(img)
Ix_gpu = cuda.mem_alloc(img.nbytes)
Iy_gpu = cuda.mem_alloc(img.nbytes)

# Get information about the available memory on the device
free_mem, total_mem = cuda.mem_get_info()

# Check if the memory allocation was successful
if Ix_gpu and Iy_gpu and (img.nbytes * 2 < free_mem):
    print("Memory allocation successful")
else:
    print("Memory allocation failed")

  



# Define the CUDA kernel
mod = SourceModule("""
    __global__ void sobel_filter(float* img, float* Ix, float* Iy, int height, int width)
    {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        int idx = i * width + j;

        if (i > 0 && j > 0 && i < height - 1 && j < width - 1) {
            float Kx[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
            float Ky[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

            Ix[idx] = 0.0;
            Iy[idx] = 0.0;

            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    Ix[idx] += Kx[x + 1][y + 1] * img[(i + x) * width + (j + y)];
                    Iy[idx] += Ky[x + 1][y + 1] * img[(i + x) * width + (j + y)];
                }
            }
        }
    }
""")

# Launch the CUDA kernel
block_size = (8, 8, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1], 1)
sobel_filter = mod.get_function("sobel_filter")
sobel_filter(img_gpu, Ix_gpu, Iy_gpu, np.int32(height), np.int32(width), block=block_size, grid=grid_size)

# Copy the results back to the CPU
Ix = np.empty_like(img)
Iy = np.empty_like(img)
cuda.memcpy_dtoh(Ix, Ix_gpu)
cuda.memcpy_dtoh(Iy, Iy_gpu)

# Perform Gaussian smoothing
# Create arrays for the Gaussian kernel and the smoothed images
G = np.array([[1/16, 2/16, 1/16],
              [2/16, 4/16, 2/16],
              [1/16, 2/16, 1/16]], dtype=np.float32)
Jx2 = np.empty_like(Ix, dtype=np.float32)
Jy2 = np.empty_like(Iy, dtype=np.float32)
Jxy = np.empty_like(Ix, dtype=np.float32)

# Allocate GPU memory for the Gaussian kernel and the smoothed images
G_gpu = cuda.to_device(G)
Jx2_gpu = cuda.mem_alloc(Jx2.nbytes)
Jy2_gpu = cuda.mem_alloc(Jy2.nbytes)
Jxy_gpu = cuda.mem_alloc(Jxy.nbytes)

# Define the CUDA kernel for smoothing
kernel = """
__global__ void gaussian_smooth(float *img_in, float *kernel, float *img_out, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        float sum = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int row_idx = row + i;
                int col_idx = col + j;
                if (row_idx >= 0 && row_idx < height && col_idx >= 0 && col_idx < width) {
                    sum += img_in[row_idx * width + col_idx] * kernel[(i + 1) * 3 + (j + 1)];
                }
            }
        }
        img_out[row * width + col] = sum;
    }
}
"""

# Compile the CUDA kernel and set the block and grid sizes
mod = SourceModule(kernel)
gaussian_smooth_cuda = mod.get_function("gaussian_smooth")
block_size = (8, 8, 1)
grid_size = ((width + block_size[0] - 1) // block_size[0], (height + block_size[1] - 1) // block_size[1], 1)

# Perform Gaussian smoothing on the x and y gradients
gaussian_smooth_cuda(Ix_gpu, G_gpu, Jx2_gpu, np.int32(height), np.int32(width), block=block_size, grid=grid_size)
gaussian_smooth_cuda(Iy_gpu, G_gpu, Jy2_gpu, np.int32(height), np.int32(width), block=block_size, grid=grid_size)

# Copy the results back to the CPU
Jx2 = np.empty_like(Ix, dtype=np.float32)
Jy2 = np.empty_like(Iy, dtype=np.float32)
Jxy = np.empty_like(Ix, dtype=np.float32)
cuda.memcpy_dtoh(Jx2, Jx2_gpu)
cuda.memcpy_dtoh(Jy2, Jy2_gpu)

# Perform element-wise multiplication of the x and y gradients and thresholding to obtain the corner response function
Jxy = Ix * Iy
R = Jx2 * Jy2 - Jxy * Jxy - 0.04 * (Jx2 + Jy2) ** 2
mask = np.zeros_like(R, dtype=np.uint8)
mask[R > 1e1] = 1

# # Overlay the corners on the original image and save the result
# def mask_over_image(img, mask):
#     img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     img_color[mask] = [255, 0, 0]
#     return img_color
# output = mask_over_image(img, mask)
# # plt.imshow(output, cmap='gray')

# plt.imshow(img, cmap='gray')
plt.imshow(mask)

end_time = time.time()

print(end_time-start_time)

# clearing memory
globals().clear()
