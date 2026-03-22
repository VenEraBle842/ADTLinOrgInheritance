#pragma once

// Пара значений произвольных типов
template <class T1, class T2>
struct Tuple {
    T1 first;
    T2 second;

    Tuple() = default;
    Tuple(const T1& f, const T2& s) : first(f), second(s) {}

    bool operator==(const Tuple& o) const {
        return first == o.first && second == o.second;
    }
};
