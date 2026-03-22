#pragma once
#include <cmath>
#include <ostream>
#include <stdexcept>

// Комплексное число

struct Complex {
    double re, im;

    Complex()                         : re(0.0), im(0.0) {}
    explicit Complex(double r)        : re(r),   im(0.0) {}
    Complex(double r, double i)       : re(r),   im(i)   {}

    // арифметика
    Complex operator+(const Complex& o) const { return {re + o.re, im + o.im}; }
    Complex operator-(const Complex& o) const { return {re - o.re, im - o.im}; }

    Complex operator*(const Complex& o) const {
        return {re*o.re - im*o.im,
                re*o.im + im*o.re};
    }

    Complex operator/(const Complex& o) const {
        double d = o.re*o.re + o.im*o.im;
        if (d == 0.0)
            throw std::runtime_error("Complex::operator/: division by zero");
        return {(re*o.re + im*o.im) / d,
                (im*o.re - re*o.im) / d};
    }

    Complex operator-() const { return {-re, -im}; }

    // составное присваивание
    Complex& operator+=(const Complex& o) { re += o.re; im += o.im; return *this; }
    Complex& operator-=(const Complex& o) { re -= o.re; im -= o.im; return *this; }
    Complex& operator*=(const Complex& o) { *this = *this * o; return *this; }
    Complex& operator/=(const Complex& o) { *this = *this / o; return *this; }

    // сравнение
    bool operator==(const Complex& o) const { return re == o.re && im == o.im; }
    bool operator!=(const Complex& o) const { return !(*this == o); }

    // метрика
    double abs()    const { return std::sqrt(re*re + im*im); }
    double normSq() const { return re*re + im*im; }

    // сопряженное
    Complex conj()  const { return {re, -im}; }
};

// вывод в поток
inline std::ostream& operator<<(std::ostream& os, const Complex& c) {
    os << c.re;
    if (c.im >= 0.0) os << "+";
    os << c.im << "i";
    return os;
}

//  Шаблонные вспомогательные функции для Solver.h

//  conjOf<T>     — сопряжение (для вещественных: тождество)
//  absOf<T>      — модуль элемента (для Frobenius-нормы и нормировки QR)
//  divideBy<T>   — деление (нужно для back-substitution)

//  Все три реализованы как constexpr-перегрузки (не специализации шаблонов),
//  чтобы компилятор выбирал нужную версию без явного указания типа.

// conjOf

inline double conjOf(double x)  { return x; }
inline float  conjOf(float  x)  { return x; }
inline int    conjOf(int    x)  { return x; }

inline Complex conjOf(const Complex& x) { return x.conj(); }

// absOf

inline double absOf(double  x) { return std::fabs(x); }
inline double absOf(float   x) { return std::fabs(static_cast<double>(x)); }
inline double absOf(int     x) { return std::fabs(static_cast<double>(x)); }

inline double absOf(const Complex& x) { return x.abs(); }

// normSqOf — |x|^2, используется при вычислении нормы Фробениуса

inline double normSqOf(double  x) { return x * x; }
inline double normSqOf(float   x) { return static_cast<double>(x) * x; }
inline double normSqOf(int     x) { return static_cast<double>(x) * x; }

inline double normSqOf(const Complex& x) { return x.normSq(); }

// innerProduct — <a, b> = conj(a) * b, скалярное произведение в C^n
// Для вещественных типов совпадает с обычным произведением.

inline double  innerProduct(double  a, double  b) { return a * b; }
inline float   innerProduct(float   a, float   b) { return a * b; }
inline int     innerProduct(int     a, int     b) { return a * b; }

inline Complex innerProduct(const Complex& a, const Complex& b) {
    return a.conj() * b;
}
