#pragma once
#include "Exceptions.h"
#include "IEnumerable.h"

// Вспомогательная функция минимума
template <class T>
const T& min(const T& a, const T& b) { return a < b ? a : b; }

// Динамический массив с управлением памятью
template <class T>
class DynamicArray : public IEnumerable<T> {
    T*  data_;
    int size_;
    int capacity_;

    static constexpr int INITIAL_CAPACITY = 8;

    static int calcCapacity(int size) {
        int cap = INITIAL_CAPACITY;
        while (cap < size) cap *= 2;
        return cap;
    }

    // Перевыделить память под newCapacity элементов
    void reallocate(int newCapacity) {
        T* nd = new T[newCapacity]();
        for (int i = 0; i < size_; ++i) nd[i] = data_[i];
        delete[] data_;
        data_     = nd;
        capacity_ = newCapacity;
    }

public:
    // конструкторы
    DynamicArray(const T* items, int count)
        : size_(count), capacity_(calcCapacity(count)) {
        if (count < 0) throw IndexOutOfRange("DynamicArray: negative count");
        data_ = new T[capacity_];
        for (int i = 0; i < count; ++i) data_[i] = items[i];
    }

    explicit DynamicArray(int size)
        : size_(size), capacity_(calcCapacity(size)) {
        if (size < 0) throw IndexOutOfRange("DynamicArray: negative size");
        data_ = new T[capacity_]();
    }

    DynamicArray(const DynamicArray<T>& other)
        : size_(other.size_), capacity_(other.capacity_) {
        data_ = new T[capacity_];
        for (int i = 0; i < size_; ++i) data_[i] = other.data_[i];
    }

    DynamicArray& operator=(const DynamicArray<T>& other) {
        if (this == &other) return *this;
        delete[] data_;
        size_     = other.size_;
        capacity_ = other.capacity_;
        data_     = new T[capacity_];
        for (int i = 0; i < size_; ++i) data_[i] = other.data_[i];
        return *this;
    }

    ~DynamicArray() { delete[] data_; }

    // методы доступа
    const T& Get(int index) const {
        if (index < 0 || index >= size_)
            throw IndexOutOfRange(index, size_ - 1);
        return data_[index];
    }

    int GetSize()     const { return size_; }

    void Set(int index, const T& value) {
        if (index < 0 || index >= size_)
            throw IndexOutOfRange(index, size_ - 1);
        data_[index] = value;
    }

    // изменение размера
    void Resize(int newSize) {
        if (newSize < 0)
            throw IndexOutOfRange("DynamicArray::Resize: negative size");
        if (newSize > capacity_) {
            int newCapacity = capacity_;
            while (newCapacity < newSize) newCapacity *= 2;
            reallocate(newCapacity);
        }
        size_ = newSize;
    }

    // Сжимает capacity вдвое, пока size < capacity / 2
    void ShrinkToFit() {
        int newCapacity = capacity_;
        while (newCapacity / 2 >= size_ && newCapacity / 2 >= INITIAL_CAPACITY)
            newCapacity /= 2;
        if (newCapacity != capacity_) reallocate(newCapacity);
    }


    // операторы
    T& operator[](int index) {
        if (index < 0 || index >= size_) throw IndexOutOfRange(index, size_ - 1);
        return data_[index];
    }
    const T& operator[](int index) const {
        if (index < 0 || index >= size_) throw IndexOutOfRange(index, size_ - 1);
        return data_[index];
    }

    // итератор
    class Enumerator : public IEnumerator<T> {
        const DynamicArray<T>* arr_;
        int pos_;
    public:
        explicit Enumerator(const DynamicArray<T>* a) : arr_(a), pos_(-1) {}
        bool MoveNext() override { return ++pos_ < arr_->size_; }
        T    Current()  const override {
            if (pos_ < 0 || pos_ >= arr_->size_)
                throw IndexOutOfRange(pos_, arr_->size_ - 1);
            return arr_->data_[pos_];
        }
        void Reset() override { pos_ = -1; }
    };

    IEnumerator<T>* GetEnumerator() const override { return new Enumerator(this); }
};
