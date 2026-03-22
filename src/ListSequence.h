#pragma once
#include "Sequence.h"
#include "LinkedList.h"

// Базовый класс на основе LinkedList
template <class T>
class ListSequence : public Sequence<T> {
protected:
    LinkedList<T>* list_;

public:
    ListSequence()
        : list_(new LinkedList<T>()) {}

    ListSequence(const T* items, int count)
        : list_(new LinkedList<T>(items, count)) {}

    ListSequence(const ListSequence<T>& other)
        : list_(new LinkedList<T>(*other.list_)) {}

    ~ListSequence() override { delete list_; }

    // чтение
    const T&   GetFirst()     const override { return list_->GetFirst(); }
    const T&   GetLast()      const override { return list_->GetLast();  }
    const T&   Get(int index) const override { return list_->Get(index); }
    int GetLength()    const override { return list_->GetLength(); }

    // Объявление без определения — MutableListSequence ещё не известен
    Sequence<T>* GetSubsequence(int startIndex, int endIndex) const override;

    // внутренние мутации
    void AppendInPlace(const T& item)              { list_->Append(item);          }
    void PrependInPlace(const T& item)             { list_->Prepend(item);         }
    void InsertAtInPlace(const T& item, int index) { list_->InsertAt(item, index); }
    void RemoveFirstInPlace()                      { list_->RemoveFirst();         }
    void RemoveLastInPlace()                       { list_->RemoveLast();          }
    void RemoveAtInPlace(int index)                { list_->RemoveAt(index);       }

    // мутация через Instance()
    Sequence<T>* Append(const T& item) override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->AppendInPlace(item);
        return inst;
    }
    Sequence<T>* Prepend(const T& item) override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->PrependInPlace(item);
        return inst;
    }
    Sequence<T>* InsertAt(const T& item, int index) override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->InsertAtInPlace(item, index);
        return inst;
    }

    Sequence<T>* RemoveFirst() override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->RemoveFirstInPlace();
        return inst;
    }

    Sequence<T>* RemoveLast() override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->RemoveLastInPlace();
        return inst;
    }

    Sequence<T>* RemoveAt(int index) override {
        auto* inst = static_cast<ListSequence<T>*>(this->Instance());
        inst->RemoveAtInPlace(index);
        return inst;
    }

    // операторы
    T&       operator[](int index)       { return list_->GetRef(index); }
    const T& operator[](int index) const {
        return const_cast<LinkedList<T>*>(list_)->GetRef(index);
    }

    // итератор
    IEnumerator<T>* GetEnumerator() const override {
        return list_->GetEnumerator();
    }
};

// MutableListSequence
template <class T>
class MutableListSequence : public ListSequence<T> {
public:
    MutableListSequence() : ListSequence<T>() {}
    MutableListSequence(const T* items, int count) : ListSequence<T>(items, count) {}
    MutableListSequence(const MutableListSequence<T>& other) : ListSequence<T>(other) {}

    Sequence<T>* Clone()       const override { return new MutableListSequence<T>(*this); }
    Sequence<T>* Instance()          override { return this; }
    Sequence<T>* CreateEmpty() const override { return new MutableListSequence<T>(); }
};

// ImmutableListSequence
template <class T>
class ImmutableListSequence : public ListSequence<T> {
public:
    ImmutableListSequence() : ListSequence<T>() {}
    ImmutableListSequence(const T* items, int count) : ListSequence<T>(items, count) {}
    ImmutableListSequence(const ImmutableListSequence<T>& other) : ListSequence<T>(other) {}

    Sequence<T>* Clone()       const override { return new ImmutableListSequence<T>(*this); }
    Sequence<T>* Instance()          override { return this->Clone(); }
    Sequence<T>* CreateEmpty() const override { return new ImmutableListSequence<T>(); }
};

// GetSubsequence определяем здесь, когда оба подкласса уже известны
template <class T>
Sequence<T>* ListSequence<T>::GetSubsequence(int startIndex, int endIndex) const {
    if (startIndex < 0 || endIndex >= GetLength() || startIndex > endIndex)
        throw IndexOutOfRange("ListSequence::GetSubsequence: invalid range [" +
            std::to_string(startIndex) + ", " + std::to_string(endIndex) + "]");
    Sequence<T>* result = this->CreateEmpty();
    for (int i = startIndex; i <= endIndex; ++i)
        appendTracked(result, Get(i));
    return result;
}
