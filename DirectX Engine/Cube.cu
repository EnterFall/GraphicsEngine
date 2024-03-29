#include "Cube.h"

__host__ __device__ Cube::Cube(const Vec3f& pos, float size)
{
	verts[0] = Vec3f(pos.x, pos.y, pos.z);						//000
	verts[1] = Vec3f(pos.x + size, pos.y, pos.z);				//100
	verts[2] = Vec3f(pos.x, pos.y + size, pos.z);				//010
	verts[3] = Vec3f(pos.x + size, pos.y + size, pos.z);		//110
	verts[4] = Vec3f(pos.x, pos.y, pos.z + size);				//001
	verts[5] = Vec3f(pos.x + size, pos.y, pos.z + size);		//101
	verts[6] = Vec3f(pos.x, pos.y + size, pos.z + size);		//011
	verts[7] = Vec3f(pos.x + size, pos.y + size, pos.z + size);	//111
}

__host__ __device__ const Vec3f* Cube::GetVerts() const
{
	return verts;
}
