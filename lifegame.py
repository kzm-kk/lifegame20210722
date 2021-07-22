import numpy
import curses
from curses import wrapper

import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

MAT_SIZE_X = 100
MAT_SIZE_Y = 100

BLOCKSIZE = 32

row2str = lambda row:''.join(['0' if c != 0 else '-' for c in row])


def print_world(stdscr, gen, world):
    '''
    show now world
    '''
    stdscr.clear()
    stdscr.nodelay(True)
    scr_height, scr_width = stdscr.getmaxyx()
    height, width = world.shape
    height = min(height, scr_height)
    width = min(width, scr_width)
    for y in range(height):
        row = world[y][:width]
        stdscr.addstr(y, 0, row2str(row))
    stdscr.refresh()

mod = SourceModule("""
__global__ void calc_next_cell_state(const int* __restrict__ world, int* next_world, int height, int width){
     int sum_cell = 0, next_cell = 0;
     int mat_x = threadIdx.x + blockIdx.x * blockDim.x;
     int mat_y = threadIdx.y + blockIdx.y * blockDim.y;

     if(mat_x >= width) return;
     if(mat_y >= height) return;

     const int index = mat_y * width + mat_x;

     const int upperIndex = (mat_y - 1) * width + mat_x;
     const int lowerIndex = (mat_y + 1) * width + mat_x;
     

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

calc_next_cell_state = mod.get_function("calc_next_cell_state")

def calc_next_world(world, next_world):
    height, width = world.shape
    block = (BLOCKSIZE, BLOCKSIZE, 1)
    grid = ((MAT_SIZE_X + block[0] - 1) // block[0], (MAT_SIZE_Y + block[1] - 1)//block[1])
    calc_next_cell_state(cuda.In(world), cuda.Out(next_world), numpy.int32(height), numpy.int32(width), block = block, grid = grid)

def gol(stdscr, height, width):
    world = numpy.random.randint(2, size = (height, width), dtype = numpy.int32)

    gen = 0
    while gen < 100:
        '''print_world(stdscr, gen, world)'''

        next_world = numpy.empty((height, width), dtype = numpy.int32)
        calc_next_world(world, next_world)
        world = next_world.copy()
        gen += 1

def main(stdscr):
    gol(stdscr, MAT_SIZE_X, MAT_SIZE_Y)

if __name__ == '__main__':
    curses.wrapper(main)
