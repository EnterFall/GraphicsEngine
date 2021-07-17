#pragma once
#include <memory>

class ZBuffer
{
public:
	int size;
private:
	std::shared_ptr<double[]> buffer;
public:
	ZBuffer(int size) : size(size)
	{
		buffer = std::make_shared<double[]>(size);
		Clear();
	}

	bool Update(int index, double val)
	{
		if (val < buffer[index])
		{
			buffer[index] = val;
			return true;
		}
		return false;
	}

	void Clear()
	{
		std::fill_n(buffer.get(), size, std::numeric_limits<double>::max());
		//std::fill_n(buffer.get(), size, 0.0);
	}
};