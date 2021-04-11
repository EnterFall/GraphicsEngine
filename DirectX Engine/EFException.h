#pragma once
#include <exception>
#include <sstream>
#include <string>

class EFException : public std::exception
{
private:
	int line;
	std::string file;
protected:
	mutable std::string whatBuffer;
public:
	EFException(int line, std::string file);
	int GetLine() const;
	std::string GetFile() const;
	std::string GetString() const;
	
	const char* what() const override;
	virtual std::string GetType() const;
};

