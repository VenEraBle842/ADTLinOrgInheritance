#pragma once
#include "Sequence.h"   // #pragma once предотвращает повторное включение
#include "DynamicArray.h"

//Базовый класс на основе DynamicArray (не инстанцируется напрямую)
template <class T>
class ArraySequence : public Sequence<T> {
protected:
    DynamicArray<T>* array_;

    // Внутренние методы прямой мутации (без паттерна Instance)
    void AppendInPlace(const T& item) {
        int n = array_->GetSize();
        array_->Resize(n + 1);
        array_->Set(n, item);
    }

    void PrependInPlace(const T& item) {
        int n = array_->GetSize();
        array_->Resize(n + 1);
        for (int i = n; i > 0; --i) array_->Set(i, array_->Get(i - 1));
        array_->Set(0, item);
    }

    void InsertAtInPlace(const T& item, int index) {
        int n = array_->GetSize();
        if (index < 0 || index > n) throw IndexOutOfRange(index, n);
        array_->Resize(n + 1);
        for (int i = n; i > index; --i) array_->Set(i, array_->Get(i - 1));
        array_->Set(index, item);
    }

    void RemoveFirstInPlace() {
        int n = array_->GetSize();
        if (n == 0) throw IndexOutOfRange("ArraySequence::RemoveFirst: empty sequence");
        for (int i = 0; i < n - 1; ++i) array_->Set(i, array_->Get(i + 1));
        array_->Resize(n - 1);
        array_->ShrinkToFit();
    }

    void RemoveLastInPlace() {
        int n = array_->GetSize();
        if (n == 0) throw IndexOutOfRange("ArraySequence::RemoveLast: empty sequence");
        array_->Resize(n - 1);
        array_->ShrinkToFit();
    }

    void RemoveAtInPlace(int index) {
        int n = array_->GetSize();
        if (index < 0 || index >= n) throw IndexOutOfRange(index, n - 1);
        for (int i = index; i < n - 1; ++i) array_->Set(i, array_->Get(i + 1));
        array_->Resize(n - 1);
        array_->ShrinkToFit();
    }

public:
    ArraySequence()
        : array_(new DynamicArray<T>(0)) {}

    ArraySequence(const T* items, int count)
        : array_(new DynamicArray<T>(items, count)) {}

    ArraySequence(const ArraySequence<T>& other)
        : array_(new DynamicArray<T>(*other.array_)) {}

    ~ArraySequence() override { delete array_; }

    // чтение
    const T& GetFirst() const override {
        if (array_->GetSize() == 0)
            throw IndexOutOfRange("ArraySequence::GetFirst: empty sequence");
        return array_->Get(0);
    }

    const T& GetLast() const override {
        int n = array_->GetSize();
        if (n == 0) throw IndexOutOfRange("ArraySequence::GetLast: empty sequence");
        return array_->Get(n - 1);
    }

    const T&   Get(int index) const override { return array_->Get(index); }
    int        GetLength()    const override { return array_->GetSize();  }

    Sequence<T>* GetSubsequence(int startIndex, int endIndex) const override;

    // мутация через Instance()
    Sequence<T>* Append(const T& item) override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->AppendInPlace(item);
        return inst;
    }

    Sequence<T>* Prepend(const T& item) override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->PrependInPlace(item);
        return inst;
    }

    Sequence<T>* InsertAt(const T& item, int index) override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->InsertAtInPlace(item, index);
        return inst;
    }

    Sequence<T>* RemoveFirst() override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->RemoveFirstInPlace();
        return inst;
    }

    Sequence<T>* RemoveLast() override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->RemoveLastInPlace();
        return inst;
    }

    Sequence<T>* RemoveAt(int index) override {
        auto* inst = static_cast<ArraySequence<T>*>(this->Instance());
        inst->RemoveAtInPlace(index);
        return inst;
    }

    // операторы
    T&       operator[](int index)       { return (*array_)[index]; }
    const T& operator[](int index) const { return (*array_)[index]; }

    // итератор
    IEnumerator<T>* GetEnumerator() const override {
        return array_->GetEnumerator();
    }
};

// MutableArraySequence
template <class T>
class MutableArraySequence : public ArraySequence<T> {
public:
    MutableArraySequence() : ArraySequence<T>() {}
    MutableArraySequence(const T* items, int count) : ArraySequence<T>(items, count) {}
    MutableArraySequence(const MutableArraySequence<T>& other) : ArraySequence<T>(other) {}

    Sequence<T>* Clone()       const override { return new MutableArraySequence<T>(*this); }
    Sequence<T>* Instance()          override { return this; }
    Sequence<T>* CreateEmpty() const override { return new MutableArraySequence<T>(); }
};

// ImmutableArraySequence
template <class T>
class ImmutableArraySequence : public ArraySequence<T> {
public:
    ImmutableArraySequence() : ArraySequence<T>() {}
    ImmutableArraySequence(const T* items, int count) : ArraySequence<T>(items, count) {}
    ImmutableArraySequence(const ImmutableArraySequence<T>& other) : ArraySequence<T>(other) {}

    Sequence<T>* Clone()       const override { return new ImmutableArraySequence<T>(*this); }
    Sequence<T>* Instance()          override { return this->Clone(); }
    Sequence<T>* CreateEmpty() const override { return new ImmutableArraySequence<T>(); }
};

// GetSubsequence определяем здесь, когда оба подкласса уже известны
template <class T>
Sequence<T>* ArraySequence<T>::GetSubsequence(int startIndex, int endIndex) const {
    if (startIndex < 0 || endIndex >= GetLength() || startIndex > endIndex)
        throw IndexOutOfRange("ArraySequence::GetSubsequence: invalid range [" +
            std::to_string(startIndex) + ", " + std::to_string(endIndex) + "]");
    // CreateEmpty() вернет Mutable или Immutable в зависимости от типа this
    Sequence<T>* result = this->CreateEmpty();
    for (int i = startIndex; i <= endIndex; ++i)
        appendTracked(result, Get(i));
    return result;
}
