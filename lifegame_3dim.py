import numpy
import curses
from curses import wrapper

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

MAT_SIZE_X = 100
MAT_SIZE_Y = 100
MAT_SIZE_Z = 100

BLOCKSIZE = 4

row2str = lambda row:''.join(['0' if c != 0 else '-' for c in row])


def print_world(stdscr, gen, world):
    '''
    show now world
    '''
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width= stdscr.getmaxyx()
    height, width, depth = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width)
    for y in range(height):
        row = world[1][y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

mod = SourceModule("""
__global__ void calc_next_cell_state_3dim(const int* __restrict__ world, int* next_world, int height, int width, int depth){
     int sum_cell = 0, next_cell = 0;
     int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
     int mat_y = threadIdx.y + blockIdx.y * blockDim.y;
     int mat_z = threadIdx.z + blockIdx.z * blockDim.z;

     if(mat_x >= width) return;
     if(mat_y >= height) return;
     if(mat_z >= depth) return;

     const int index = mat_z * height + mat_y * width + mat_x;

     const int upperIndex = mat_z * height + (mat_y - 1) * width + mat_x;
     const int lowerIndex = mat_z * height + (mat_y + 1) * width + mat_x;
     

     sum_cell += world[upperIndex - 1];
     sum_cell += world[upperIndex];
     sum_cell += world[upperIndex + 1];

     sum_cell += world[index - 1];
     sum_cell += world[index + 1];

     sum_cell += world[lowerIndex - 1];
     sum_cell += world[lowerIndex];
     sum_cell += world[lowerIndex + 1];

    if(sum_cell == 2 && world[index] == 1) next_cell = 1;
    else if(sum_cell == 3) next_cell = 1;
    else next_cell = 0;

    next_world[index] = next_cell;
}
""")

calc_next_cell_state_3dim = mod.get_function("calc_next_cell_state_3dim")

def calc_next_world_3dim(world, next_world):
    height, width, depth = world.shape
    block = (BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
    grid = ((MAT_SIZE_X + block[0] - 1) // block[0], (MAT_SIZE_Y + block[1] - 1)//block[1], (MAT_SIZE_Z + block[2] - 1)//block[2])
    calc_next_cell_state_3dim(cuda.In(world), cuda.Out(next_world), numpy.int32(height), numpy.int32(width), numpy.int32(depth), block = block, grid = grid)

def gol(stdscr, height, width, depth):
    world = numpy.random.randint(2, size = (height, width, depth), dtype = numpy.int32)

    gen = 0
    while True:
        print_world(stdscr, gen, world)

        next_world = numpy.empty((height, width, depth), dtype = numpy.int32)
        calc_next_world_3dim(world, next_world)
        world = next_world.copy()
        gen += 1

def main(stdscr):
    gol(stdscr, MAT_SIZE_X, MAT_SIZE_Y, MAT_SIZE_Z)

if __name__ == '__main__':
    curses.wrapper(main)
