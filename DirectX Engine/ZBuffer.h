#pragma once
#include <memory>

class ZBuffer
{
public:
	int size;
private:
	std::shared_ptr<float[]> buffer;
public:
	ZBuffer(int size) : size(size)
	{
		buffer = std::make_shared<float[]>(size);
		Clear();
	}

	bool Update(int index, float val)
	{
		if (val > buffer[index])
		{
			buffer[index] = val;
			return true;
		}
		return false;
	}

	void Clear()
	{
		//std::fill_n(buffer.get(), size, std::numeric_limits<float>::max());
		std::fill_n(buffer.get(), size, 0.0f);
	}
};