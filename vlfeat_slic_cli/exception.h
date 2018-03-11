#pragma once

#include <string>
#include <exception.h>


namespace ml {

class Exception : public std::exception {
protected:
    std::string msg_;
public:
    explicit Exception(const char* msg)
	:msg_(msg) {}

    explicit Exception(const std::string& msg)
	:msg_(msg) {}
    
    virtual ~Exception() {}

    virtual const char* what() const noexcept {
	return msg_.c_str();
    }
};

} // namespace ml

