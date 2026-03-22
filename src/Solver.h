#pragma once
#include <type_traits>
#include "Matrix.h"
#include "ArraySequence.h"

//  Скалярные вспомогательные функции

//  Complex имеет explicit-конструктор — static_cast<Complex>(double) не работает.
//  Эти два шаблона нужны только здесь, поэтому живут в Solver.h, а не Complex.h.

template <class T> inline T fromDouble(double x) { return static_cast<T>(x); }
template <>        inline Complex fromDouble<Complex>(double x) { return {x, 0.0}; }

template <class T> inline T scalarOne()  { return fromDouble<T>(1.0); }
template <class T> inline T scalarZero() { return fromDouble<T>(0.0); }

//  LUSolver<T>

//  Решает систему Ax = b методом LU-разложения с частичным выбором ведущего
//  элемента (алгоритм Дулиттла: PA = LU).

//  Жизненный цикл кэша:
//    - конструктор             -> dirty = true (разложение не вычислено)
//    - первый Solve / GetL ... -> decompose() -> dirty = false
//    - последующие Solve(b')   -> кэш валиден, только подстановки O(n^2)
//    - SetMatrix(A')           -> freeCache() -> dirty = true -> при следующем
//                                Solve снова decompose() O(n^3)
template <class T>
class LUSolver {
    static_assert(
        std::is_floating_point_v<T> || std::is_same_v<T, Complex>,
        "LUSolver: the type of elements must be real or Complex");

    // основное состояние
    MutableSquareMatrix<T>* A_;     // глубокая копия исходной матрицы
    double                  eps_;   // порог нулевого пивота

    // кэш (mutable -> Solve() объявлен const)
    mutable bool                    dirty_;
    mutable MutableSquareMatrix<T>* L_;
    mutable MutableSquareMatrix<T>* U_;
    mutable DynamicArray<int>*      perm_;  // perm_[k] = исходный номер строки k

    // управление памятью кэша

    void freeCache() const {
        delete L_;    L_    = nullptr;
        delete U_;    U_    = nullptr;
        delete perm_; perm_ = nullptr;
        dirty_ = true;
    }

    // Полиморфная копия: работает с любым подтипом SquareMatrix
    static MutableSquareMatrix<T>* copyAny(const SquareMatrix<T>& src) {
        int n = src.GetSize();
        auto* dst = new MutableSquareMatrix<T>(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                dst->Set(i, j, src.Get(i, j));
        return dst;
    }

    // PA = LU: алгоритм Дулиттла с частичным выбором пивота
    void decompose() const {
        int n = A_->GetSize();

        // W — рабочая копия A; будет модифицирована in-place
        auto* W = copyAny(*A_);

        // Инициализируем вектор перестановки как тождественный
        perm_ = new DynamicArray<int>(n);
        for (int i = 0; i < n; ++i) perm_->Set(i, i);

        for (int k = 0; k < n; ++k) {
            // Ищем строку с максимальным |W[i][k]| при i >= k
            double maxVal = absOf(W->Get(k, k));
            int    maxIdx = k;
            for (int i = k + 1; i < n; ++i) {
                double v = absOf(W->Get(i, k));
                if (v > maxVal) { maxVal = v; maxIdx = i; }
            }

            if (maxVal < eps_) {
                delete W;
                freeCache();   // освобождает perm_ и сбрасывает dirty_ = true
                throw SingularMatrixException(k, maxVal, eps_);
            }

            // Переставляем строки k и maxIdx
            if (maxIdx != k) {
                for (int j = 0; j < n; ++j) {
                    T tmp = W->Get(k, j);
                    W->Set(k, j, W->Get(maxIdx, j));
                    W->Set(maxIdx, j, tmp);
                }
                int tmpP = perm_->Get(k);
                perm_->Set(k, perm_->Get(maxIdx));
                perm_->Set(maxIdx, tmpP);
            }

            // Вычисляем множители и обновляем подматрицу Шура
            for (int i = k + 1; i < n; ++i) {
                T factor = W->Get(i, k) / W->Get(k, k);
                W->Set(i, k, factor);   // W[i][k] ← L[i][k] (ниже диагонали)
                for (int j = k + 1; j < n; ++j)
                    W->Set(i, j, W->Get(i, j) - factor * W->Get(k, j));
            }
        }

        // Извлекаем L и U из packed-матрицы W
        L_ = new MutableSquareMatrix<T>(n);
        U_ = new MutableSquareMatrix<T>(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > j) {
                    L_->Set(i, j, W->Get(i, j));     // нижний треугольник
                    U_->Set(i, j, scalarZero<T>());
                } else if (i == j) {
                    L_->Set(i, j, scalarOne<T>());        // диагональ L = 1 (Дулиттл)
                    U_->Set(i, j, W->Get(i, j));
                } else {
                    L_->Set(i, j, scalarZero<T>());
                    U_->Set(i, j, W->Get(i, j));     // верхний треугольник
                }
            }
        }
        delete W;
    }

    void ensureDecomposed() const {
        if (dirty_) {
            decompose();
            dirty_ = false;
        }
    }

    // прямая подстановка: Ly = Pb
    // L — единичная нижняя треугольная (L[i][i] = 1), поэтому деления нет.
    MutableArraySequence<T>* forwardSub(const Sequence<T>& b) const {
        int n = A_->GetSize();
        auto* y = new MutableArraySequence<T>();
        for (int i = 0; i < n; ++i) {
            T val = b.Get(perm_->Get(i));   // Pb: применяем перестановку
            for (int j = 0; j < i; ++j)
                val = val - L_->Get(i, j) * (*y)[j];
            y->Append(val);
        }
        return y;
    }

    // обратная подстановка: Ux = y
    MutableArraySequence<T>* backSub(const MutableArraySequence<T>* y) const {
        int n = A_->GetSize();
        auto* x = new MutableArraySequence<T>();
        for (int i = 0; i < n; ++i)
            x->Append(scalarZero<T>());
        for (int i = n - 1; i >= 0; --i) {
            T val = y->Get(i);
            for (int j = i + 1; j < n; ++j)
                val = val - U_->Get(i, j) * (*x)[j];
            (*x)[i] = val / U_->Get(i, i);
        }
        return x;
    }

public:
    // конструктор / деструктор

    explicit LUSolver(const SquareMatrix<T>& A, double eps = 1e-12)
        : A_(copyAny(A)), eps_(eps),
          dirty_(true), L_(nullptr), U_(nullptr), perm_(nullptr) {}

    ~LUSolver() { delete A_; freeCache(); }

    LUSolver(const LUSolver&)            = delete;
    LUSolver& operator=(const LUSolver&) = delete;

    // смена матрицы
    // Сбрасывает кэш. Следующий Solve пересчитает PA = LU.
    void SetMatrix(const SquareMatrix<T>& A) {
        freeCache();
        delete A_;
        A_ = copyAny(A);
    }

    // решение Ax = b
    // Первый вызов: O(n^3) разложение + O(n^2) подстановки.
    // Повторные вызовы с той же A (другим b): только O(n^2) подстановки.
    // Возвращает x; владение — у вызывающего (caller owns).
    MutableArraySequence<T>* Solve(const Sequence<T>& b) const {
        int n = A_->GetSize();
        if (b.GetLength() != n)
            throw RHSLengthMismatch(b.GetLength(), n);
        ensureDecomposed();
        auto* y = forwardSub(b);    // Ly = Pb
        auto* x = backSub(y);       // Ux = y
        delete y;
        return x;
    }

    // доступ к кэшированным множителям (ленивое вычисление)
    const SquareMatrix<T>&   GetL()    const { ensureDecomposed(); return *L_;    }
    const SquareMatrix<T>&   GetU()    const { ensureDecomposed(); return *U_;    }
    const DynamicArray<int>& GetP()    const { ensureDecomposed(); return *perm_; }
    int                      GetSize() const { return A_->GetSize();               }
};


//  QRSolver<T>

//  Решает систему Ax = b методом QR-разложения (A = QR).
//  Алгоритм: модифицированный метод Грама–Шмидта (численно устойчивее классического).
//  Решение: QRx = b  ->  Rx = Q^H * b  ->  back-substitution.
//  Q^H для вещественных типов, Q* (эрмитово сопряжение) для Complex.

//  Жизненный цикл кэша — идентичен LUSolver.
template <class T>
class QRSolver {
    static_assert(
        std::is_floating_point_v<T> || std::is_same_v<T, Complex>,
        "QRSolver: the type of elements must be real or Complex");

    // состояние
    MutableSquareMatrix<T>* A_;
    double                  eps_;

    // кэш
    mutable bool                    dirty_;
    mutable MutableSquareMatrix<T>* Q_;
    mutable MutableSquareMatrix<T>* R_;

    void freeCache() const {
        delete Q_; Q_ = nullptr;
        delete R_; R_ = nullptr;
        dirty_ = true;
    }

    static MutableSquareMatrix<T>* copyAny(const SquareMatrix<T>& src) {
        int n = src.GetSize();
        auto* dst = new MutableSquareMatrix<T>(n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                dst->Set(i, j, src.Get(i, j));
        return dst;
    }

    // A = QR: модифицированный метод Грама–Шмидта

    // Идея: Q начинается как копия A. На каждом шаге j нормируем j-й столбец Q
    // и вычитаем его проекцию из всех последующих столбцов.
    // Диагональ R[j][j] = норма j-го столбца до нормировки (всегда вещественна
    // и положительна; для Complex хранится как Complex с нулевой мнимой частью).
    // R[j][k] (k > j) = скалярное произведение <Q[:,j], Q[:,k]> до ортогонализации.
    void decompose() const {
        int n = A_->GetSize();

        Q_ = copyAny(*A_);                    // столбцы будут ортонормированы
        R_ = new MutableSquareMatrix<T>(n);   // инициализирован нулями

        for (int j = 0; j < n; ++j) {
            // Норма j-го столбца Q: ||Q[:,j]||
            double norm = 0.0;
            for (int i = 0; i < n; ++i)
                norm += normSqOf(Q_->Get(i, j));
            norm = std::sqrt(norm);

            if (norm < eps_) {
                freeCache();
                throw RankDeficientException(j, norm, eps_);
            }

            // R[j][j] = norm; нормируем столбец j
            R_->Set(j, j, fromDouble<T>(norm));
            T invNorm = fromDouble<T>(1.0 / norm);
            for (int i = 0; i < n; ++i)
                Q_->Set(i, j, Q_->Get(i, j) * invNorm);

            // Ортогонализируем столбцы k > j относительно нового Q[:,j]
            for (int k = j + 1; k < n; ++k) {
                // R[j][k] = <Q[:,j], Q[:,k]> = sum_i conj(Q[i][j]) * Q[i][k]
                // innerProduct(a,b) = conjOf(a) * b  (из Complex.h)
                T rjk = scalarZero<T>();
                for (int i = 0; i < n; ++i)
                    rjk = rjk + innerProduct(Q_->Get(i, j), Q_->Get(i, k));
                R_->Set(j, k, rjk);

                // Q[:,k] -= R[j][k] * Q[:,j]
                for (int i = 0; i < n; ++i)
                    Q_->Set(i, k, Q_->Get(i, k) - rjk * Q_->Get(i, j));
            }
        }
    }

    void ensureDecomposed() const {
        if (dirty_) {
            decompose();
            dirty_ = false;
        }
    }

    // умножение Q^H * b
    // c[j] = (Q^H * b)[j] = sum_i conj(Q[i][j]) * b[i]
    MutableArraySequence<T>* mulQdaggerB(const Sequence<T>& b) const {
        int n = A_->GetSize();
        auto* c = new MutableArraySequence<T>();
        for (int j = 0; j < n; ++j) {
            T val = scalarZero<T>();
            for (int i = 0; i < n; ++i)
                val = val + innerProduct(Q_->Get(i, j), b.Get(i));
            c->Append(val);
        }
        return c;
    }

    // обратная подстановка: Rx = c
    // R — верхняя треугольная.
    MutableArraySequence<T>* backSub(const MutableArraySequence<T>* c) const {
        int n = A_->GetSize();
        auto* x = new MutableArraySequence<T>();
        for (int i = 0; i < n; ++i)
            x->Append(scalarZero<T>());
        for (int i = n - 1; i >= 0; --i) {
            T val = c->Get(i);
            for (int j = i + 1; j < n; ++j)
                val = val - R_->Get(i, j) * (*x)[j];
            (*x)[i] = val / R_->Get(i, i);
        }
        return x;
    }

public:
    // конструктор / деструктор

    explicit QRSolver(const SquareMatrix<T>& A, double eps = 1e-12)
        : A_(copyAny(A)), eps_(eps),
          dirty_(true), Q_(nullptr), R_(nullptr) {}

    ~QRSolver() { delete A_; freeCache(); }

    QRSolver(const QRSolver&)            = delete;
    QRSolver& operator=(const QRSolver&) = delete;

    // смена матрицы
    void SetMatrix(const SquareMatrix<T>& A) {
        freeCache();
        delete A_;
        A_ = copyAny(A);
    }

    // решение Ax = b
    // Первый вызов: O(n^3) разложение + O(n^2) умножение Q^H * b + O(n^2) подстановка.
    // Повторные вызовы с той же A: только O(n^2).
    MutableArraySequence<T>* Solve(const Sequence<T>& b) const {
        int n = A_->GetSize();
        if (b.GetLength() != n)
            throw RHSLengthMismatch(b.GetLength(), n);
        ensureDecomposed();
        auto* c = mulQdaggerB(b);   // c = Q^H * b
        auto* x = backSub(c);       // Rx = c
        delete c;
        return x;
    }

    // доступ к кэшированным множителям
    const SquareMatrix<T>& GetQ()    const { ensureDecomposed(); return *Q_; }
    const SquareMatrix<T>& GetR()    const { ensureDecomposed(); return *R_; }
    int                    GetSize() const { return A_->GetSize();             }
};
