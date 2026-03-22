#pragma once

// Интерфейс итератора
template <class T>
class IEnumerator {
public:
    virtual ~IEnumerator() = default;
    virtual bool MoveNext() = 0;
    virtual T    Current()  const = 0;
    virtual void Reset()    = 0;
};

// Адаптер превращает IEnumerator в range-based итератор
// Владеет указателем на IEnumerator и удаляет его в деструкторе.
// Копирование запрещено, перемещение разрешено (нужно для возврата из begin()).
template <class T>
class EnumeratorAdapter {
    IEnumerator<T>* enumerator_;
    bool            valid_;

public:
    explicit EnumeratorAdapter(IEnumerator<T>* e)
        : enumerator_(e), valid_(e != nullptr && e->MoveNext()) {}

    EnumeratorAdapter()
        : enumerator_(nullptr), valid_(false) {}

    ~EnumeratorAdapter() { delete enumerator_; }

    EnumeratorAdapter(const EnumeratorAdapter&)            = delete;
    EnumeratorAdapter& operator=(const EnumeratorAdapter&) = delete;

    EnumeratorAdapter(EnumeratorAdapter&& o) noexcept
        : enumerator_(o.enumerator_), valid_(o.valid_) {
        o.enumerator_ = nullptr;
        o.valid_      = false;
    }

    EnumeratorAdapter& operator++() {
        if (enumerator_) valid_ = enumerator_->MoveNext();
        return *this;
    }

    T    operator*()                            const { return enumerator_->Current(); }
    bool operator!=(const EnumeratorAdapter& o) const { return valid_ != o.valid_;     }
};
