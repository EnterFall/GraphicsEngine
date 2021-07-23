#pragma once
#include "cuda_runtime.h"
#include "Vec3.h"

class Cube
{
public:
	Vec3d verts[8];
private:

public:
	__host__ __device__ Cube(const Vec3d& pos, double size = 1.0);
	__host__ __device__ const Vec3d* GetVerts() const;
};

