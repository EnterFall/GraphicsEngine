#pragma once
struct Vec2f
{
	float x;
	float y;

	Vec2f() = default;

	Vec2f(float x1, float y1)
	{
		this->x = x1;
		this->y = y1;
	}

	Vec2f operator+(const Vec2f& r) const
	{
		return Vec2f{ this->x + r.x, this->y + r.y };
	}

	Vec2f operator-(const Vec2f& r) const
	{
		return Vec2f{ this->x - r.x, this->y - r.y };
	}
};