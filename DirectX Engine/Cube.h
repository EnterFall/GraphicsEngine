#pragma once
#include "cuda_runtime.h"
#include "Vec3.h"

class Cube
{
public:
	Vec3f verts[8];
private:

public:
	__host__ __device__ Cube(const Vec3f& pos, float size = 1.0f);
	__host__ __device__ const Vec3f* GetVerts() const;
};

