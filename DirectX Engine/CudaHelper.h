#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "EFException.h"

#ifdef __INTELLISENSE__
template<class T>
void surf2Dwrite(T data, cudaSurfaceObject_t surfObj, int x, int y,
	cudaSurfaceBoundaryMode boundaryMode = cudaBoundaryModeTrap);
void __syncthreads();
#endif

template <class T>
__host__ __device__ void swap(T& a, T& b)
{
	T temp = a;
	a = b;
	b = temp;
}

template <class T>
__host__ __device__ T clamp(T n, T lower, T upper) 
{
#if defined(__CUDA_ARCH__)
	return fmax(lower, fmin(n, upper));
#else
	return max(lower, min(n, upper));
#endif
}

static void CudaAssert_Impl(int line, std::string file, cudaError_t code)
{
	if (code != cudaSuccess) {
		throw EFException(line, file, cudaGetErrorString(code));
	}
}
#define CudaAssert(code) CudaAssert_Impl(__LINE__, __FILE__, code)

class CudaHelper
{
public:
	template <class T>
	static T* CreateManagedCudaObj(size_t count = 1u)
	{
		T* resultPtr;
		CudaAssert(cudaMallocManaged(&resultPtr, count * sizeof(T)));
		return resultPtr;
	}

	template <class T>
	static T* CreateCopyToManagedCudaObj(T* hostObj, size_t count = 1u)
	{
		T* resultPtr;
		CudaAssert(cudaMallocManaged(&resultPtr, count * sizeof(T)));
		CudaAssert(cudaMemcpy(resultPtr, hostObj, count * sizeof(T), cudaMemcpyHostToDevice));
		return resultPtr;
	}

	template <class T>
	void CudaArrayFill(cudaArray_t arr, T value)
	{
		
	}
};