#pragma once
#include "Vec3.h"

class Cube
{
public:
	Vec3d verts[8];
private:

public:
	Cube(const Vec3d& pos, double size = 1.0);
	const Vec3d* GetVerts() const;
};

