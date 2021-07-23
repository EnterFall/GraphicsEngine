#pragma once
#include "cuda_runtime.h"
#include "EFException.h"

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
};
