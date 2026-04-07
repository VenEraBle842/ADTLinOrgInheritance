#pragma once
#include "Exceptions.h"
#include "IEnumerable.h"
#include "DynamicArray.h"
#include "Complex.h"   // normSqOf, conjOf, absOf, innerProduct + struct Complex

// опережающие объявления
template <class T> class ArraySquareMatrix;
template <class T> class MutableSquareMatrix;
template <class T> class ImmutableSquareMatrix;

//  Абстрактный базовый класс квадратной матрицы
template <class T>
class SquareMatrix : public IEnumerable<T> {
public:
    virtual ~SquareMatrix() = default;

    // доступ к элементам
    virtual int      GetSize()                         const = 0;
    virtual const T& Get(int row, int col)             const = 0;
    virtual void     Set(int row, int col, const T& v)       = 0;

    // паттерн Mutable / Immutable
    virtual SquareMatrix<T>* Clone()            const = 0;
    virtual SquareMatrix<T>* Instance()               = 0;
    virtual SquareMatrix<T>* CreateEmpty(int n) const = 0;

    // арифметика (возвращают новый объект; владение — у вызывающего)
    SquareMatrix<T>* Add      (const SquareMatrix<T>* other) const;
    SquareMatrix<T>* MulScalar(const T& scalar)              const;
    double           Norm     ()                             const; // Фробениус

    // элементарные преобразования строк
    // Mutable: мутируют this и возвращают this.
    // Immutable: возвращают новый клон, this не трогают.
    SquareMatrix<T>* SwapRows(int i, int j);
    SquareMatrix<T>* MulRow  (int i, const T& scalar);
    SquareMatrix<T>* AddRow  (int i, int j, const T& scalar); // row[i] += scalar*row[j]

    // элементарные преобразования столбцов
    SquareMatrix<T>* SwapCols(int i, int j);
    SquareMatrix<T>* MulCol  (int i, const T& scalar);
    SquareMatrix<T>* AddCol  (int i, int j, const T& scalar); // col[i] += scalar*col[j]

    // операторы
    bool operator==(const SquareMatrix<T>& other) const {
        if (GetSize() != other.GetSize()) return false;
        int n = GetSize();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (Get(i, j) != other.Get(i, j)) return false;
        return true;
    }
    bool operator!=(const SquareMatrix<T>& other) const { return !(*this == other); }
};

//  Плоское хранение (row-major) на DynamicArray
template <class T>
class ArraySquareMatrix : public SquareMatrix<T> {
protected:
    int              size_;
    DynamicArray<T>* data_;   // (i, j)  ->  i * size_ + j

    void checkRC(int r, int c) const {
        if (r < 0 || r >= size_) throw RowIndexOutOfRange(r, size_);
        if (c < 0 || c >= size_) throw ColIndexOutOfRange(c, size_);
    }

public:
    // конструкторы
    explicit ArraySquareMatrix(int n)
    : size_(n >= 0 ? n : throw InvalidMatrixSize(n)),
      data_(new DynamicArray<T>(n * n))
    {}

    ArraySquareMatrix(const ArraySquareMatrix<T>& other)
        : size_(other.size_), data_(new DynamicArray<T>(*other.data_)) {}

    ArraySquareMatrix& operator=(const ArraySquareMatrix<T>& other) {
        if (this == &other) return *this;
        delete data_;
        size_ = other.size_;
        data_ = new DynamicArray<T>(*other.data_);
        return *this;
    }

    ~ArraySquareMatrix() override { delete data_; }

    // доступ
    int GetSize() const override { return size_; }

    const T& Get(int r, int c) const override {
        checkRC(r, c);
        return data_->Get(r * size_ + c);
    }

    void Set(int r, int c, const T& v) override {
        checkRC(r, c);
        data_->Set(r * size_ + c, v);
    }

    // IEnumerable: обход в порядке row-major
    class Enumerator : public IEnumerator<T> {
        const ArraySquareMatrix<T>* mat_;
        int pos_;                               // текущая позиция в плоском массиве
    public:
        explicit Enumerator(const ArraySquareMatrix<T>* m) : mat_(m), pos_(-1) {}

        bool MoveNext() override {
            return ++pos_ < mat_->size_ * mat_->size_;
        }

        T Current() const override {
            int total = mat_->size_ * mat_->size_;
            if (pos_ < 0 || pos_ >= total)
                throw IndexOutOfRange(
                    "SquareMatrix::Enumerator: no current element");
            return mat_->data_->Get(pos_);
        }

        void Reset() override { pos_ = -1; }
    };

    IEnumerator<T>* GetEnumerator() const override {
        return new Enumerator(this);
    }
};

//  Реализации методов SquareMatrix

template <class T>
SquareMatrix<T>* SquareMatrix<T>::Add(const SquareMatrix<T>* other) const {
    int n = GetSize();
    if (other->GetSize() != n)
        throw MatrixSizeMismatch(n, other->GetSize());
    SquareMatrix<T>* result = CreateEmpty(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result->Set(i, j, Get(i, j) + other->Get(i, j));
    return result;
}

template <class T>
SquareMatrix<T>* SquareMatrix<T>::MulScalar(const T& scalar) const {
    int n = GetSize();
    SquareMatrix<T>* result = CreateEmpty(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            result->Set(i, j, Get(i, j) * scalar);
    return result;
}

template <class T>
double SquareMatrix<T>::Norm() const {
    // Норма Фробениуса: sqrt( sum |a_ij|^2 )
    // normSqOf из Complex.h корректно обрабатывает int, double и Complex
    double sum = 0.0;
    int n = GetSize();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            sum += normSqOf(Get(i, j));
    return std::sqrt(sum);
}

// строки

template <class T>
SquareMatrix<T>* SquareMatrix<T>::SwapRows(int i, int j) {
    int n = GetSize();
    if (i < 0 || i >= n) throw RowIndexOutOfRange(i, n);
    if (j < 0 || j >= n) throw RowIndexOutOfRange(j, n);
    SquareMatrix<T>* inst = Instance();
    for (int k = 0; k < n; ++k) {
        T tmp = inst->Get(i, k);
        inst->Set(i, k, inst->Get(j, k));
        inst->Set(j, k, tmp);
    }
    return inst;
}

template <class T>
SquareMatrix<T>* SquareMatrix<T>::MulRow(int i, const T& scalar) {
    int n = GetSize();
    if (i < 0 || i >= n) throw RowIndexOutOfRange(i, n);
    SquareMatrix<T>* inst = Instance();
    for (int k = 0; k < n; ++k)
        inst->Set(i, k, inst->Get(i, k) * scalar);
    return inst;
}

template <class T>
SquareMatrix<T>* SquareMatrix<T>::AddRow(int i, int j, const T& scalar) {
    int n = GetSize();
    if (i < 0 || i >= n) throw RowIndexOutOfRange(i, n);
    if (j < 0 || j >= n) throw RowIndexOutOfRange(j, n);
    SquareMatrix<T>* inst = Instance();
    // Снимаем снимок строки j до мутации — защита от случая i == j
    DynamicArray<T> snap(n);
    for (int k = 0; k < n; ++k)
        snap.Set(k, inst->Get(j, k));
    for (int k = 0; k < n; ++k)
        inst->Set(i, k, inst->Get(i, k) + snap.Get(k) * scalar);
    return inst;
}

// столбцы

template <class T>
SquareMatrix<T>* SquareMatrix<T>::SwapCols(int i, int j) {
    int n = GetSize();
    if (i < 0 || i >= n) throw ColIndexOutOfRange(i, n);
    if (j < 0 || j >= n) throw ColIndexOutOfRange(j, n);
    SquareMatrix<T>* inst = Instance();
    for (int k = 0; k < n; ++k) {
        T tmp = inst->Get(k, i);
        inst->Set(k, i, inst->Get(k, j));
        inst->Set(k, j, tmp);
    }
    return inst;
}

template <class T>
SquareMatrix<T>* SquareMatrix<T>::MulCol(int i, const T& scalar) {
    int n = GetSize();
    if (i < 0 || i >= n) throw ColIndexOutOfRange(i, n);
    SquareMatrix<T>* inst = Instance();
    for (int k = 0; k < n; ++k)
        inst->Set(k, i, inst->Get(k, i) * scalar);
    return inst;
}

template <class T>
SquareMatrix<T>* SquareMatrix<T>::AddCol(int i, int j, const T& scalar) {
    int n = GetSize();
    if (i < 0 || i >= n) throw ColIndexOutOfRange(i, n);
    if (j < 0 || j >= n) throw ColIndexOutOfRange(j, n);
    SquareMatrix<T>* inst = Instance();
    // Снимок столбца j до мутации — защита от случая i == j
    DynamicArray<T> snap(n);
    for (int k = 0; k < n; ++k)
        snap.Set(k, inst->Get(k, j));
    for (int k = 0; k < n; ++k)
        inst->Set(k, i, inst->Get(k, i) + snap.Get(k) * scalar);
    return inst;
}

//  MutableSquareMatrix
template <class T>
class MutableSquareMatrix : public ArraySquareMatrix<T> {
public:
    explicit MutableSquareMatrix(int n)
        : ArraySquareMatrix<T>(n) {}

    explicit MutableSquareMatrix(const SquareMatrix<T>& other)
    : ArraySquareMatrix<T>(other.GetSize()) {
        int n = other.GetSize();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                this->Set(i, j, other.Get(i, j));
    }

    MutableSquareMatrix(const MutableSquareMatrix<T>& other)
        : ArraySquareMatrix<T>(other) {}

    SquareMatrix<T>* Clone()            const override {
        return new MutableSquareMatrix<T>(*this);
    }
    SquareMatrix<T>* Instance()               override {
        return this;
    }
    SquareMatrix<T>* CreateEmpty(int n) const override {
        return new MutableSquareMatrix<T>(n);
    }
};

//  ImmutableSquareMatrix
template <class T>
class ImmutableSquareMatrix : public ArraySquareMatrix<T> {
public:
    explicit ImmutableSquareMatrix(int n)
        : ArraySquareMatrix<T>(n) {}

    explicit ImmutableSquareMatrix(const SquareMatrix<T>& other)
    : ArraySquareMatrix<T>(other.GetSize()) {
        int n = other.GetSize();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                this->Set(i, j, other.Get(i, j));
    }

    ImmutableSquareMatrix(const ImmutableSquareMatrix<T>& other)
        : ArraySquareMatrix<T>(other) {}

    SquareMatrix<T>* Clone()            const override {
        return new ImmutableSquareMatrix<T>(*this);
    }
    SquareMatrix<T>* Instance()               override {
        return Clone();   // каждая мутирующая операция возвращает новый объект
    }
    SquareMatrix<T>* CreateEmpty(int n) const override {
        return new ImmutableSquareMatrix<T>(n);
    }
};
