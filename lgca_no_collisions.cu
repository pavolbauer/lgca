#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#include <stdio.h>
// P. Bauer, 2011

// Must be identifical with GridDim
#define SHARED_DIM_X  22
#define SHARED_DIM_Y  22

__shared__ unsigned int as[(SHARED_DIM_X)*(SHARED_DIM_Y)];
__shared__ unsigned int bs[(SHARED_DIM_X)*(SHARED_DIM_Y)];
__shared__ unsigned int cs[(SHARED_DIM_X)*(SHARED_DIM_Y)];
__shared__ unsigned int ds[(SHARED_DIM_X)*(SHARED_DIM_Y)];
__shared__ unsigned int es[(SHARED_DIM_X)*(SHARED_DIM_Y)];
__shared__ unsigned int fs[(SHARED_DIM_X)*(SHARED_DIM_Y)];

__device__ unsigned int iLin( int r, int c, int m, int n) {
    r = ( r >= m ? r - m : r );
    r = ( r <  0 ? r + m : r );
 
    c = ( c >= n ? c - n : c );
    c = ( c <  0 ? c + n : c );
    return c * m + r;
}

__global__ void runLGCA(unsigned int* b1, unsigned int* b2, unsigned int* b3, unsigned int* b4, unsigned int* b5, unsigned int* b6, int m, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int tidx=threadIdx.x+1;
	int tidy=threadIdx.y+1;

	// Adressing in shared memory
	int linidx = iLin(tidx,tidy, SHARED_DIM_X, SHARED_DIM_Y);
	// Adressing in global memory
	int linidxG = iLin(x,y,m,n);
	
	// Actual cell
	as[linidx] = b1[linidxG];
	bs[linidx] = b2[linidxG];
	cs[linidx] = b3[linidxG];
	ds[linidx] = b4[linidxG];
	es[linidx] = b5[linidxG];
	fs[linidx] = b6[linidxG];

	// Ghost cells
	if ( threadIdx.x == 0 && threadIdx.y != blockDim.y - 1 ) // Upper left corner
	{
		fs[iLin(0, tidy,      SHARED_DIM_X, SHARED_DIM_Y)] = b6[iLin(x-1, y,   m, n)];
		es[iLin(0, tidy+1, SHARED_DIM_X, SHARED_DIM_Y)] = b5[iLin(x-1, y+1, m, n)];
	}
	if ( threadIdx.x == blockDim.x - 1 && threadIdx.y != 0 ) // Right lower corner
	{
		bs[iLin(blockDim.x + 1, tidy-1,   SHARED_DIM_X, SHARED_DIM_Y)] = b2[iLin(x+1, y-1, m, n)];
		cs[iLin(blockDim.x + 1, tidy,      SHARED_DIM_X, SHARED_DIM_Y)] = b3[iLin(x+1, y,   m, n)];
	}
	if ( threadIdx.y == 0 && threadIdx.x != blockDim.x - 1) // Left upper corner
	{
		as[iLin(tidx,    0, SHARED_DIM_X, SHARED_DIM_Y)] = b1[iLin(x, y-1, m,n)];
		bs[iLin(tidx+1, 0, SHARED_DIM_X, SHARED_DIM_Y)] = b2[iLin(x+1, y-1, m, n)];
	}
	if ( threadIdx.y == blockDim.y - 1  && threadIdx.x != 0) // lower corner
	{
		ds[iLin(tidx,    blockDim.y + 1, SHARED_DIM_X, SHARED_DIM_Y)] = b4[iLin(x,  y+1, m, n)];
		es[iLin(tidx-1,   blockDim.y + 1, SHARED_DIM_X, SHARED_DIM_Y)] = b5[iLin(x-1,y+1, m, n)];
	}
	if ( threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) // Left lower corner
	{
		fs[iLin(0,  blockDim.y    ,   SHARED_DIM_X, SHARED_DIM_Y)] = b6[iLin(x-1, y,   m, n)];
		es[iLin(0, blockDim.y + 1, SHARED_DIM_X, SHARED_DIM_Y)] = b5[iLin(x-1 ,y+1, m, n)];
		ds[iLin(1, blockDim.y + 1, SHARED_DIM_X, SHARED_DIM_Y)] = b4[iLin(x ,  y+1, m, n)];
	}
	if ( threadIdx.x == blockDim.x - 1 &&  threadIdx.y == 0) // Right upper corner
	{
		as[iLin(blockDim.x   , 0, SHARED_DIM_X, SHARED_DIM_Y)] = b1[iLin(x,   y-1, m, n)];
		bs[iLin(blockDim.x +1, 0, SHARED_DIM_X, SHARED_DIM_Y)] = b2[iLin(x+1, y-1, m, n)];
		cs[iLin(blockDim.x +1, 1, SHARED_DIM_X, SHARED_DIM_Y)] = b3[iLin(x+1, y,   m, n)];
	}
	__syncthreads();

	// Wait for all collisions to be computed and prepare for writing propagation

	unsigned int nextA = as[ iLin(tidx,     tidy-1,   SHARED_DIM_X, SHARED_DIM_Y) ];
	unsigned int nextB = bs[ iLin(tidx+1, tidy-1,   SHARED_DIM_X, SHARED_DIM_Y) ];
	unsigned int nextC = cs[ iLin(tidx+1, tidy,      SHARED_DIM_X, SHARED_DIM_Y) ];
	unsigned int nextD = ds[ iLin(tidx,     tidy+1,  SHARED_DIM_X, SHARED_DIM_Y) ];
	unsigned int nextE = es[ iLin(tidx-1,  tidy+1,  SHARED_DIM_X, SHARED_DIM_Y) ];
	unsigned int nextF = fs[ iLin(tidx-1,   tidy,      SHARED_DIM_X, SHARED_DIM_Y) ];
	
	// Collission protection
	__syncthreads(); 

	as[linidx] = nextA;
	bs[linidx] = nextB;
	cs[linidx] = nextC;
	ds[linidx] = nextD;
	es[linidx] = nextE;
	fs[linidx] = nextF;
	
	__syncthreads();
	
	if ( x < n && y < m ) 
	{
		b1[linidxG] = as[linidx];
		b2[linidxG] = bs[linidx];
		b3[linidxG] = cs[linidx];
		b4[linidxG] = ds[linidx];
		b5[linidxG] = es[linidx];
		b6[linidxG] = fs[linidx];
	}
}


#endif // #ifndef _MATRIXMUL_KERNEL_H_
