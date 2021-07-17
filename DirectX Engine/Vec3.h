#pragma once
#include <algorithm>

template <typename T>
struct Vec3
{
	T x;
	T y;
	T z;

	Vec3() = default;

	Vec3(T x, T y, T z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vec3 operator +(const Vec3& other) const
	{
		return Vec3(x + other.x, y + other.y, z + other.z);
	}

	Vec3 operator -(const Vec3& other) const
	{
		return Vec3(x - other.x, y - other.y, z - other.z);
	}

	Vec3 operator *(const Vec3& other) const
	{
		return Vec3(x * other.x, y * other.y, z * other.z);
	}

	Vec3 operator /(const Vec3& other) const
	{
		return Vec3(x / other.x, y / other.y, z / other.z);
	}

	Vec3 operator +(const T& val) const
	{
		return Vec3(x + val, y + val, z + val);
	}

	Vec3 operator -(const T& val) const
	{
		return Vec3(x - val, y - val, z - val);
	}

	Vec3 operator *(const T& val) const
	{
		return Vec3(x * val, y * val, z * val);
	}

	Vec3 operator /(const T& val) const
	{
		T v = 1.0f / val;
		return Vec3(x * v, y * v, z * v);
	}

	Vec3& operator +=(const T& val) const
	{
		x += val;
		y += val;
		z += val;
		return *this;
	}

	bool operator ==(const Vec3& other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}

	bool operator !=(const Vec3& other) const
	{
		return x != other.x || y != other.y || z != other.z;
	}

	Vec3 Cross(const Vec3& other) const
	{
		return Vec3(
			y * other.z - z * other.y, 
			z * other.x - x * other.z, 
			x * other.y - y * other.x);
	}

	T Dot(const Vec3& other) const
	{
		return x * other.x + y * other.y + z * other.z;
	}

	T Length() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	T LengthSquare() const
	{
		return x * x + y * y + z * z;
	}

	T Cos(const Vec3& other) const
	{
		return Dot(other) / (Length() * other.Length());
	}

	void Normalize()
	{
		auto len = 1.0f / Length();
		x *= len;
		y *= len;
		z *= len;
	}
};
typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;
