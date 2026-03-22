#pragma once
#include "IEnumerator.h"

// Интерфейс перечисляемой коллекции
template <class T>
class IEnumerable {
public:
    virtual ~IEnumerable() = default;
    virtual IEnumerator<T>* GetEnumerator() const = 0;

    // range-based for
    EnumeratorAdapter<T> begin() const { return EnumeratorAdapter<T>(GetEnumerator()); }
    EnumeratorAdapter<T> end()   const { return EnumeratorAdapter<T>(); }
};
