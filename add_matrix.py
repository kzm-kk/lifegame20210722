import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy

from pycuda.compiler import SourceModule

MAT_SIZE_X = 1000
MAT_SIZE_Y = 1000

BLOCKSIZE = 32

mod = SourceModule("""
__global__ void add_matrix_gpu(const float* __restrict__ dMat_A, const float* __restrict__ dMat_B, float* dMat_G, const int mat_size_x, const mat_size_y){
     int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
     int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
     if(mat_x >= mat_size_x){
          return;
     }
     if(mat_y >= mat_size_y){
          return;
     }

     const int index = mat_y * mat_size_x + mat_x;
     dMat_G[index] = dMat_A[index] + dMat_B[index];
}
""")
add_matrix_gpu = mod.get_function("add_matrix_gpu")

block = (BLOCKSIZE, BLOCKSIZE, 1)
grid = ((MAT_SIZE_X + block[0] - 1)// block[0], (MAT_SIZE_Y + block[0] - 1) //block[0])

h_a = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_b = numpy.random.randn(MAT_SIZE_X, MAT_SIZE_Y).astype(numpy.float32)
h_d = numpy.empty_like(h_a)

