#pragma once
#include <stdexcept>

// Тип-обертка для значений, которые могут отсутствовать
template <class T>
class Option {
    bool hasValue_;
    T    value_;

    Option() : hasValue_(false), value_{} {}
    explicit Option(const T& v) : hasValue_(true), value_(v) {}

public:
    static Option Some(const T& v) { return Option(v); }
    static Option None()           { return Option(); }

    bool HasValue()  const { return hasValue_; }
    T    GetValue()  const {
        if (!hasValue_) throw std::runtime_error("Option::GetValue called on None");
        return value_;
    }

    bool operator==(const Option& o) const {
        if (hasValue_ != o.hasValue_) return false;
        return !hasValue_ || value_ == o.value_;
    }
};
