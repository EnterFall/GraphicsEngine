#include "Keyboard.h"

void Keyboard::KeyDownFromMessage(char wParam, char lParam)
{
	keysState[wParam] = true;
}

void Keyboard::KeyUpFromMessage(char wParam, char lParam)
{
	keysState[wParam] = false;
}

bool Keyboard::IsPressed(char key)
{
	return keysState[key];
}


void Keyboard::Clear()
{
	keysState.reset();
}