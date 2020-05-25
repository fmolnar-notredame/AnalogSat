//This file is part of AnalogSAT
//Copyright(C) 2019 Ferenc Molnar
//
//AnalogSAT is free software: you can redistribute it and / or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program. If not, see <http://www.gnu.org/licenses/>.


#include <cstdio>
#include <stdexcept>
#include <vector>

#include "CudaReduce.h"
#include "CudaUtils.h"

#if CUDART_VERSION >= 9000
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
#endif

using namespace std;

namespace analogsat
{
	//CUDA Reduction - based on the CUDA SDK example
	//Generalized to any binary operator via functor
	//Source data is left intact, instead there are 2 temporary buffers needed

	// Utility class used to avoid linker errors with extern unsized shared memory arrays with templated type
	template<class T>
	struct SharedMemory
	{
		__device__ inline operator T *()
		{
			extern __shared__ int __smem[];
			return (T *)__smem;
		}

		__device__ inline operator const T *() const
		{
			extern __shared__ int __smem[];
			return (T *)__smem;
		}
	};


	//very generic reduction
	//T: value type
	//OP: binary operator functor
	//param initval: initial value of the reduction for each thread
	template <typename T, typename OP, unsigned int blockSize, bool nIsPow2>
	__global__ void	reduce(T *g_idata, T *g_odata, unsigned int n, T initval)
	{
#if CUDART_VERSION >= 9000
		cg::thread_block cta = cg::this_thread_block();
#endif

		T *sdata = SharedMemory<T>();

		//initialize the reduction value
		T mySum = initval;

		//instantiate the operator (all on GPU: no need to pass function objects or device func ptr)
		OP op;

		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
		unsigned int gridSize = blockSize * 2 * gridDim.x;

		// reduce multiple elements per thread.  The number is determined by the
		// number of active thread blocks (via gridDim).  More blocks will result
		// in a larger gridSize and therefore fewer elements per thread
		while (i < n)
		{
			mySum = op(mySum, g_idata[i]);

			// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
			if (nIsPow2 || i + blockSize < n) mySum = op(mySum, g_idata[i + blockSize]);

			i += gridSize;
		}

		// each thread puts its local sum into shared memory
		sdata[tid] = mySum;

		if (blockSize < 32) sdata[tid + blockSize] = initval;

#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif

		// do reduction in shared mem
		if ((blockSize >= 512) && (tid < 256)) 	{ sdata[tid] = mySum = op(mySum, sdata[tid + 256]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif


		if ((blockSize >= 256) && (tid < 128)) 	{ sdata[tid] = mySum = op(mySum, sdata[tid + 128]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif

		if ((blockSize >= 128) && (tid < 64)) 	{ sdata[tid] = mySum = op(mySum, sdata[tid + 64]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif


		//finish reduction in warp
#if (__CUDA_ARCH__ >= 300 )
		if (tid < 32)
		{
#if CUDART_VERSION >= 9000
			cg::coalesced_group active = cg::coalesced_threads();

			// Fetch final intermediate sum from 2nd warp
			if (blockSize >= 64) mySum = op(mySum, sdata[tid + 32]);

			//int start = warpSize;
			int start = blockSize < warpSize ? blockSize : warpSize; //CUDA BUG FIX

			// Reduce final warp using shuffle
			for (int offset = start / 2; offset > 0; offset /= 2)  //IF BLOCK < WARP then this shuffle reads from inactive lanes!!! RESULT UNDEFINED but seems to loop around
				mySum = op(mySum, active.shfl_down(mySum, offset));
#else
			// Fetch final intermediate sum from 2nd warp
			if (blockSize >= 64) mySum = op(mySum, sdata[tid + 32]);

			// Reduce final warp using shuffle
			for (int offset = warpSize / 2; offset > 0; offset /= 2)  //this is fine, __shfl_down() uses correct mask
			{
				mySum = op(mySum, __shfl_down(mySum, offset));
			}
#endif
		}
#else
		// fully unroll reduction within a single warp (warp size = 32)
		if ((blockSize >= 64) && (tid < 32)) { sdata[tid] = mySum = op(mySum, sdata[tid + 32]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif
		if ((blockSize >= 32) && (tid < 16)) { sdata[tid] = mySum = op(mySum, sdata[tid + 16]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif
		if ((blockSize >= 16) && (tid < 8))	{ sdata[tid] = mySum = op(mySum, sdata[tid + 8]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif
		if ((blockSize >= 8) && (tid < 4)) { sdata[tid] = mySum = op(mySum, sdata[tid + 4]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif
		if ((blockSize >= 4) && (tid < 2)) { sdata[tid] = mySum = op(mySum, sdata[tid + 2]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif
		if ((blockSize >= 2) && (tid < 1)) { sdata[tid] = mySum = op(mySum, sdata[tid + 1]); }
#if CUDART_VERSION >= 9000
		cg::sync(cta);
#else
		__syncthreads();
#endif

#endif

		// write result for this block to global mem
		if (tid == 0) g_odata[blockIdx.x] = mySum;
	}


	bool isPow2(unsigned int x)
	{
		return ((x&(x - 1)) == 0);
	}

	unsigned int nextPow2(unsigned int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif


	void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
	{
		threads = (n < maxThreads * 2) ? nextPow2((n + 1) / 2) : maxThreads;
		blocks = (n + (threads * 2 - 1)) / (threads * 2);
		blocks = MIN(maxBlocks, blocks);
	}


	////////////////////////////////////////////////////////////////////////////////
	// Wrapper function for kernel launch
	////////////////////////////////////////////////////////////////////////////////
	template <class T, class OP>
	void launchReduce(int size, int threads, int blocks, T *d_idata, T *d_odata, T initval)
	{
		dim3 dimBlock(threads, 1, 1);
		dim3 dimGrid(blocks, 1, 1);

		// when there is only one warp per block, we need to allocate two warps
		// worth of shared memory so that we don't index shared memory out of bounds
		int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

		if (isPow2(size))
		{
			switch (threads)
			{
			case 512:
				reduce<T, OP, 512, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 256:
				reduce<T, OP, 256, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 128:
				reduce<T, OP, 128, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 64:
				reduce<T, OP, 64, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 32:
				reduce<T, OP, 32, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 16:
				reduce<T, OP, 16, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  8:
				reduce<T, OP, 8, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  4:
				reduce<T, OP, 4, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  2:
				reduce<T, OP, 2, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  1:
				reduce<T, OP, 1, true> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;
			}
		}
		else
		{
			switch (threads)
			{
			case 512:
				reduce<T, OP, 512, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 256:
				reduce<T, OP, 256, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 128:
				reduce<T, OP, 128, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 64:
				reduce<T, OP, 64, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 32:
				reduce<T, OP, 32, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case 16:
				reduce<T, OP, 16, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  8:
				reduce<T, OP, 8, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  4:
				reduce<T, OP, 4, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  2:
				reduce<T, OP, 2, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;

			case  1:
				reduce<T, OP, 1, false> KERNEL_ARGS3(dimGrid, dimBlock, smemSize)(d_idata, d_odata, size, initval);
				break;
			}		
		}
	}	

	template<class T>
	void CudaReduceMax(int size, T *d_idata, T *d_odata, T* d_temp)
	{
		int maxThreads = 256;  // number of threads per block	
		int maxBlocks = 64;

		int numBlocks = 0;
		int numThreads = 0;
		getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

		//reduce in blocks
		launchReduce<T, functor_devmax<T>>(size, numThreads, numBlocks, d_idata, d_odata, (T)0);

		// reduce partial block results on GPU
		int s = numBlocks;
		while (s > 1)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
			cudaMemcpyAsync(d_temp, d_odata, s*sizeof(T), cudaMemcpyDeviceToDevice);
			launchReduce<T, functor_devmax<T>>(s, threads, blocks, d_temp, d_odata, (T)0);
			s = (s + (threads * 2 - 1)) / (threads * 2);
		}
	}
	
	
	
	template<class T>
	void CudaReduceSum(int size, T *d_idata, T *d_odata, T* d_temp)
	{
		int maxThreads = 256;  // number of threads per block	
		int maxBlocks = 64;

		int numBlocks = 0;
		int numThreads = 0;
		getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

		//reduce in blocks
		launchReduce<T, functor_devsum<T>>(size, numThreads, numBlocks, d_idata, d_odata, (T)0);

		// reduce partial block results on GPU
		int s = numBlocks;
		while (s > 1)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
			cudaMemcpyAsync(d_temp, d_odata, s*sizeof(T), cudaMemcpyDeviceToDevice);
			launchReduce<T, functor_devsum<T>>(s, threads, blocks, d_temp, d_odata, (T)0);
			s = (s + (threads * 2 - 1)) / (threads * 2);
		}
	}

	template<>
	void CudaReduceSum<int>(int size, int *d_idata, int *d_odata, int* d_temp)
	{
		int maxThreads = 256;  // number of threads per block	
		int maxBlocks = 64;

		int numBlocks = 0;
		int numThreads = 0;
		getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

		//reduce in blocks
		launchReduce<int, functor_devsum<int>>(size, numThreads, numBlocks, d_idata, d_odata, (int)0);

		//DEBUG
		//vector<int> temp(size);
		//cudaMemcpy(temp.data(), d_idata, size * sizeof(int), cudaMemcpyDeviceToHost);
		//int res = 0;
		//for (int i = 0; i < size; i++) res += temp[i];
		//printf("%d\n", res);

		//vector<int> temp2(MAX_BLOCK_DIM_SIZE);
		//cudaMemcpy(temp2.data(), d_odata, MAX_BLOCK_DIM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
		//int res2 = 0;
		//for (int i = 0; i < MAX_BLOCK_DIM_SIZE; i++) res2 += temp2[i];
		//printf("%d\n", res2);


		// reduce partial block results on GPU
		int s = numBlocks;
		while (s > 1)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
			cudaMemcpyAsync(d_temp, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);
			launchReduce<int, functor_devsum<int>>(s, threads, blocks, d_temp, d_odata, (int)0);
			
			//DEBUG
			//cudaMemcpy(temp.data(), d_temp, s * sizeof(int), cudaMemcpyDeviceToHost);
			//int res = 0;
			//for (int i = 0; i < s; i++) res += temp[i];
			//printf("%d\n", res);

			//cudaMemcpy(temp2.data(), d_odata, MAX_BLOCK_DIM_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
			//int res2 = 0;
			//for (int i = 0; i < MAX_BLOCK_DIM_SIZE; i++) res2 += temp2[i];
			//printf("%d\n", res2);

			s = (s + (threads * 2 - 1)) / (threads * 2);
		}
	}

	void CudaReduceAnd(int size, int *d_idata, int *d_odata, int* d_temp)
	{
		int maxThreads = 256;  // number of threads per block	
		int maxBlocks = 64;

		int numBlocks = 0;
		int numThreads = 0;
		getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

		//reduce in blocks
		launchReduce<int, functor_devand>(size, numThreads, numBlocks, d_idata, d_odata, (int)true);

		// reduce partial block results on GPU
		int s = numBlocks;
		while (s > 1)
		{
			int threads = 0, blocks = 0;
			getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);
			cudaMemcpy(d_temp, d_odata, s*sizeof(int), cudaMemcpyDeviceToDevice);
			launchReduce<int, functor_devand>(s, threads, blocks, d_temp, d_odata, (int)true);
			s = (s + (threads * 2 - 1)) / (threads * 2);
		}
	}

	//instantiate
	template void CudaReduceSum<double>(int size, double *d_idata, double *d_odata, double *d_temp);
	template void CudaReduceSum<float>(int size, float *d_idata, float *d_odata, float *d_temp);
	template void CudaReduceSum<int>(int size, int *d_idata, int *d_odata, int *d_temp);

	template void CudaReduceMax<double>(int size, double *d_idata, double *d_odata, double *d_temp);
	template void CudaReduceMax<float>(int size, float *d_idata, float *d_odata, float *d_temp);
}