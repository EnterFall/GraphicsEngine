#pragma once
#include "cuda_runtime.h"
#include <cmath>
#include <memory>
#include <algorithm>

class ZBuffer
{
private:
	int size;
public:
	float* buffer;
	__host__ __device__ ZBuffer(int size) : size(size)
	{
		buffer = new float[size];
		Clear();
	}

	__host__ __device__ bool Update(int index, float val)
	{
		if (val < buffer[index])
		{
			buffer[index] = val;
			return true;
		}
		return false;
	}

	__host__ __device__ void Clear()
	{
		memset(buffer, 0x7f, size * sizeof(float));
	}

	__host__ __device__ ~ZBuffer()
	{
		delete[] buffer;
	}
};