#pragma once
#include <ostream>
#include "Complex.h"
#include "Matrix.h"

inline std::ostream& operator<<(std::ostream& os, const Complex& c) {
    os << c.re;
    if (c.im >= 0.0) os << "+";
    os << c.im << "i";
    return os;
}

template <class T>
std::ostream& operator<<(std::ostream& os, const SquareMatrix<T>& m) {
    int n = m.GetSize();
    for (int i = 0; i < n; ++i) {
        os << "[ ";
        for (int j = 0; j < n; ++j) {
            if (j > 0) os << "  ";
            os << m.Get(i, j);
        }
        os << " ]\n";
    }
    return os;
}
