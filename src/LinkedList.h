#pragma once
#include "Exceptions.h"
#include "IEnumerable.h"

// Однонаправленный связный список
template <class T>
class LinkedList : public IEnumerable<T> {
    struct Node {
        T     data;
        Node* next;
        explicit Node(const T& d) : data(d), next(nullptr) {}
    };

    Node* head_;
    Node* tail_;
    int   length_;

    void Clear() {
        Node* cur = head_;
        while (cur) { Node* n = cur->next; delete cur; cur = n; }
        head_ = tail_ = nullptr;
        length_ = 0;
    }

public:
    // конструкторы
    LinkedList(const T* items, int count) : head_(nullptr), tail_(nullptr), length_(0) {
        for (int i = 0; i < count; ++i) Append(items[i]);
    }

    LinkedList() : head_(nullptr), tail_(nullptr), length_(0) {}

    LinkedList(const LinkedList<T>& other) : head_(nullptr), tail_(nullptr), length_(0) {
        for (Node* cur = other.head_; cur; cur = cur->next) Append(cur->data);
    }

    LinkedList& operator=(const LinkedList<T>& other) {
        if (this == &other) return *this;
        Clear();
        for (Node* cur = other.head_; cur; cur = cur->next) Append(cur->data);
        return *this;
    }

    ~LinkedList() { Clear(); }

    // методы доступа
    const T& GetFirst() const {
        if (!head_) throw IndexOutOfRange("LinkedList::GetFirst: list is empty");
        return head_->data;
    }

    const T& GetLast() const {
        if (!tail_) throw IndexOutOfRange("LinkedList::GetLast: list is empty");
        return tail_->data;
    }

    const T& Get(int index) const {
        if (index < 0 || index >= length_)
            throw IndexOutOfRange(index, length_ - 1);
        Node* cur = head_;
        for (int i = 0; i < index; ++i) cur = cur->next;
        return cur->data;
    }

    T& GetRef(int index) {
        if (index < 0 || index >= length_)
            throw IndexOutOfRange(index, length_ - 1);
        Node* cur = head_;
        for (int i = 0; i < index; ++i) cur = cur->next;
        return cur->data;
    }

    LinkedList<T>* GetSubList(int startIndex, int endIndex) const {
        if (startIndex < 0 || endIndex >= length_ || startIndex > endIndex)
            throw IndexOutOfRange("LinkedList::GetSubList: invalid range [" +
                std::to_string(startIndex) + ", " + std::to_string(endIndex) + "]");
        auto* result = new LinkedList<T>();
        Node* cur = head_;
        for (int i = 0; i < startIndex; ++i) cur = cur->next;
        for (int i = startIndex; i <= endIndex; ++i, cur = cur->next)
            result->Append(cur->data);
        return result;
    }

    int GetLength() const { return length_; }

    // модифицирующие методы
    void Append(const T& item) {
        Node* node = new Node(item);
        if (!tail_) { head_ = tail_ = node; }
        else { tail_->next = node; tail_ = node; }
        ++length_;
    }

    void Prepend(const T& item) {
        Node* node = new Node(item);
        node->next = head_;
        head_ = node;
        if (!tail_) tail_ = head_;
        ++length_;
    }

    void InsertAt(const T& item, int index) {
        if (index < 0 || index > length_)
            throw IndexOutOfRange(index, length_);
        if (index == 0)        { Prepend(item); return; }
        if (index == length_)  { Append(item);  return; }
        Node* prev = head_;
        for (int i = 0; i < index - 1; ++i) prev = prev->next;
        Node* node = new Node(item);
        node->next = prev->next;
        prev->next = node;
        ++length_;
    }

    void RemoveFirst() {
        if (!head_) throw IndexOutOfRange("LinkedList::RemoveFirst: list is empty");
        Node* toDelete = head_;
        head_ = head_->next;
        if (!head_) tail_ = nullptr;
        delete toDelete;
        --length_;
    }

    void RemoveLast() {
        if (!tail_) throw IndexOutOfRange("LinkedList::RemoveLast: list is empty");
        if (head_ == tail_) {
            delete head_;
            head_ = tail_ = nullptr;
            --length_;
            return;
        }
        Node* prev = head_;
        while (prev->next != tail_) prev = prev->next;
        delete tail_;
        tail_       = prev;
        tail_->next = nullptr;
        --length_;
    }

    void RemoveAt(int index) {
        if (index < 0 || index >= length_)
            throw IndexOutOfRange(index, length_ - 1);
        if (index == 0)            { RemoveFirst(); return; }
        if (index == length_ - 1)  { RemoveLast();  return; }
        Node* prev = head_;
        for (int i = 0; i < index - 1; ++i) prev = prev->next;
        Node* toDelete = prev->next;
        prev->next     = toDelete->next;
        delete toDelete;
        --length_;
    }

    LinkedList<T>* Concat(const LinkedList<T>* other) const {
        auto* result = new LinkedList<T>(*this);
        for (Node* cur = other->head_; cur; cur = cur->next)
            result->Append(cur->data);
        return result;
    }

    // итератор
    class Enumerator : public IEnumerator<T> {
        Node* cur_;
        bool  started_;
        const LinkedList<T>* list_;
    public:
        explicit Enumerator(const LinkedList<T>* l)
            : cur_(nullptr), started_(false), list_(l) {}
        bool MoveNext() override {
            if (!started_) { cur_ = list_->head_; started_ = true; }
            else if (cur_) cur_ = cur_->next;
            return cur_ != nullptr;
        }
        T    Current() const override {
            if (!cur_) throw IndexOutOfRange("LinkedList::Enumerator: no current");
            return cur_->data;
        }
        void Reset() override { cur_ = nullptr; started_ = false; }
    };

    IEnumerator<T>* GetEnumerator() const override { return new Enumerator(this); }
};
