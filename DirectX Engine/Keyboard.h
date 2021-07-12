#pragma once
#include <queue>
#include <bitset>

class Keyboard
{
private:
	static constexpr int bufferSize = 20;
	static constexpr unsigned int stateSize = 256u;
	std::queue<char> keys;
	std::bitset<stateSize> keysState;
public:
	void KeyDownFromMessage(char wParam, char lParam);
	void KeyUpFromMessage(char wParam, char lParam);
	bool IsPressed(char key);
	void Clear();
};

