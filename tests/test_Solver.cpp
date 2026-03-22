#include <gtest/gtest.h>
#include <cmath>
#include "../src/Solver.h"

static constexpr double TOL = 1e-9;

// helpers

// Заполнить матрицу из плоского row-major массива
template <class T>
static void fillMatrix(SquareMatrix<T>& m, const T* data) {
    int n = m.GetSize();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m.Set(i, j, data[i * n + j]);
}

// Заполнить MutableArraySequence из массива
template <class T>
static MutableArraySequence<T>* makeVec(const T* data, int n) {
    auto* v = new MutableArraySequence<T>();
    for (int i = 0; i < n; ++i) v->Append(data[i]);
    return v;
}

// Сравнить два вектора с допуском
static void expectVecNear(const Sequence<double>& got,
                          const double* expected, int n, double tol = TOL) {
    ASSERT_EQ(got.GetLength(), n);
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(got.Get(i), expected[i], tol);
}

static void expectVecNearC(const Sequence<Complex>& got,
                           const Complex* expected, int n, double tol = TOL) {
    ASSERT_EQ(got.GetLength(), n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(got.Get(i).re, expected[i].re, tol);
        EXPECT_NEAR(got.Get(i).im, expected[i].im, tol);
    }
}

// Вычислить невязку: ||Ax - b||
static double residual(const SquareMatrix<double>& A,
                       const Sequence<double>& x,
                       const Sequence<double>& b) {
    int n = A.GetSize();
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double row = 0.0;
        for (int j = 0; j < n; ++j) row += A.Get(i,j) * x.Get(j);
        double diff = row - b.Get(i);
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static double residualC(const SquareMatrix<Complex>& A,
                        const Sequence<Complex>& x,
                        const Sequence<Complex>& b) {
    int n = A.GetSize();
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        Complex row{0,0};
        for (int j = 0; j < n; ++j) row += A.Get(i,j) * x.Get(j);
        sum += (row - b.Get(i)).normSq();
    }
    return std::sqrt(sum);
}

//  LUSolver — базовые случаи

TEST(LUSolver, Solve2x2) {
    // [[2,1],[5,3]] * x = [8,13] -> x = [11, -14]? нет: x=[1,6]
    // 2*1 + 1*6 = 8 ok   5*1 + 3*6 = 23 не ok!
    // Пересчет: [[2,1],[5,3]] x = [8,13]
    // det = 6-5 = 1, x1 = (3*8 - 1*13)/1 = 11, x2 = (2*13 - 5*8)/1 = -14
    MutableSquareMatrix<double> A(2);
    double dataA[] = {2.0, 1.0,
                      5.0, 3.0};
    fillMatrix(A, dataA);
    double dataB[] = {8.0, 13.0};
    auto* b = makeVec(dataB, 2);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    double expected[] = {11.0, -14.0};
    expectVecNear(*x, expected, 2);
    EXPECT_NEAR(residual(A, *x, *b), 0.0, TOL);
    delete x; delete b;
}

TEST(LUSolver, Solve3x3) {
    // [[1,2,3],[0,1,4],[5,6,0]] * x = [14, 11, 2] -> x = [-38, 27, -11]? проверяем невязкой
    MutableSquareMatrix<double> A(3);
    double dataA[] = {1,2,3,
                      0,1,4,
                      5,6,0};
    fillMatrix(A, dataA);
    double dataB[] = {14.0, 11.0, 2.0};
    auto* b = makeVec(dataB, 3);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

TEST(LUSolver, SolveIdentity) {
    // I * x = b -> x = b
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,1); A.Set(1,1,1); A.Set(2,2,1);
    double dataB[] = {3.0, -1.0, 7.0};
    auto* b = makeVec(dataB, 3);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    expectVecNear(*x, dataB, 3);
    delete x; delete b;
}

TEST(LUSolver, SolveDiagonal) {
    // diag(2, 3, 5) * x = [6, 9, 10] -> x = [3, 3, 2]
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,2); A.Set(1,1,3); A.Set(2,2,5);
    double dataB[]     = {6.0, 9.0, 10.0};
    double expected[]  = {3.0, 3.0,  2.0};
    auto* b = makeVec(dataB, 3);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    expectVecNear(*x, expected, 3);
    delete x; delete b;
}

TEST(LUSolver, SolveUpperTriangular) {
    // [[2,1,0],[0,3,0],[0,0,4]] * x = [5,6,8] -> x = [1.5, 2, 2]
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,2); A.Set(0,1,1);
    A.Set(1,1,3);
    A.Set(2,2,4);
    double dataB[]    = {5.0, 6.0, 8.0};
    double expected[] = {1.5, 2.0, 2.0};
    auto* b = makeVec(dataB, 3);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    expectVecNear(*x, expected, 3);
    delete x; delete b;
}

TEST(LUSolver, Solve4x4Residual) {
    MutableSquareMatrix<double> A(4);
    double dataA[] = { 4, 3, 2, 1,
                       3, 4, 3, 2,
                       2, 3, 4, 3,
                       1, 2, 3, 4 };
    fillMatrix(A, dataA);
    double dataB[] = {10.0, 12.0, 12.0, 10.0};
    auto* b = makeVec(dataB, 4);

    LUSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

//  LUSolver — повторное использование кэша

TEST(LUSolver, CacheReuseMultipleRHS) {
    // Одна матрица, три разных b — все решения должны давать нулевую невязку
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);

    LUSolver<double> solver(A);

    double b1[] = {1.0, 1.0, 1.0};
    double b2[] = {0.0, 1.0, 0.0};
    double b3[] = {3.0,-1.0, 2.0};

    for (auto* bArr : {b1, b2, b3}) {
        auto* b = makeVec(bArr, 3);
        auto* x = solver.Solve(*b);
        EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
        delete x; delete b;
    }
}

TEST(LUSolver, SetMatrixInvalidatesCache) {
    // Решаем с A1, затем меняем на A2, проверяем что ответ соответствует A2
    MutableSquareMatrix<double> A1(2), A2(2);
    A1.Set(0,0,2); A1.Set(0,1,0); A1.Set(1,0,0); A1.Set(1,1,3);
    A2.Set(0,0,1); A2.Set(0,1,2); A2.Set(1,0,2); A2.Set(1,1,1);

    double dataB[] = {2.0, 3.0};
    auto* b = makeVec(dataB, 2);

    LUSolver<double> solver(A1);
    auto* x1 = solver.Solve(*b);

    solver.SetMatrix(A2);
    auto* x2 = solver.Solve(*b);

    // x1 должен быть решением A1, x2 — решением A2
    EXPECT_NEAR(residual(A1, *x1, *b), 0.0, TOL);
    EXPECT_NEAR(residual(A2, *x2, *b), 0.0, TOL);
    // x1 и x2 должны различаться
    EXPECT_GT(std::fabs(x1->Get(0) - x2->Get(0)), 1e-6);

    delete x1; delete x2; delete b;
}

//  LUSolver — доступ к L, U, P

TEST(LUSolver, LowerTriangular) {
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);
    LUSolver<double> solver(A);

    const SquareMatrix<double>& L = solver.GetL();
    int n = L.GetSize();
    // L[i][j] = 0 при j > i
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            EXPECT_NEAR(L.Get(i,j), 0.0, TOL);
    // Диагональ L = 1 (Дулиттл)
    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(L.Get(i,i), 1.0, TOL);
}

TEST(LUSolver, UpperTriangular) {
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);
    LUSolver<double> solver(A);

    const SquareMatrix<double>& U = solver.GetU();
    int n = U.GetSize();
    // U[i][j] = 0 при i > j
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(U.Get(i,j), 0.0, TOL);
}

TEST(LUSolver, PermutationVectorLength) {
    MutableSquareMatrix<double> A(4);
    double dataA[] = {4,3,2,1,
                      3,4,3,2,
                      2,3,4,3,
                      1,2,3,4};
    fillMatrix(A, dataA);
    LUSolver<double> solver(A);
    EXPECT_EQ(solver.GetP().GetSize(), 4);
}

TEST(LUSolver, GetSizeMatchesMatrix) {
    MutableSquareMatrix<double> A(5);
    LUSolver<double> solver(A);
    EXPECT_EQ(solver.GetSize(), 5);
}

//  LUSolver — обработка ошибок

TEST(LUSolver, SingularMatrixThrows) {
    MutableSquareMatrix<double> A(2);
    // нулевая матрица — вырожденная
    LUSolver<double> solver(A);
    double dataB[] = {1.0, 1.0};
    auto* b = makeVec(dataB, 2);
    EXPECT_THROW(solver.Solve(*b), SingularMatrixException);
    delete b;
}

TEST(LUSolver, SingularByRankThrows) {
    // строки пропорциональны
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,1); A.Set(0,1,2); A.Set(0,2,3);
    A.Set(1,0,2); A.Set(1,1,4); A.Set(1,2,6);
    A.Set(2,0,3); A.Set(2,1,6); A.Set(2,2,9);
    LUSolver<double> solver(A);
    double dataB[] = {1.0, 2.0, 3.0};
    auto* b = makeVec(dataB, 3);
    EXPECT_THROW(solver.Solve(*b), SingularMatrixException);
    delete b;
}

TEST(LUSolver, WrongRHSLengthThrows) {
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,1); A.Set(1,1,1); A.Set(2,2,1);
    double dataB[] = {1.0, 2.0};       // длина 2, а матрица 3×3
    auto* b = makeVec(dataB, 2);
    LUSolver<double> solver(A);
    EXPECT_THROW(solver.Solve(*b), RHSLengthMismatch);
    delete b;
}

TEST(LUSolver, StaticAssertFloatOnly) {
    // Следующая строка не должна компилироваться:
    // LUSolver<int> bad(...);
    // Проверяем косвенно — тип double принимается без ошибок
    MutableSquareMatrix<double> A(1);
    A.Set(0,0,1.0);
    LUSolver<double> solver(A);
    SUCCEED();
}

//  LUSolver — тип Complex

TEST(LUSolver, SolveComplex2x2) {
    // [[1+i, 0],[0, 2]] * x = [2+2i, 4] -> x = [2, 2]
    MutableSquareMatrix<Complex> A(2);
    A.Set(0,0,{1,1}); A.Set(0,1,{0,0});
    A.Set(1,0,{0,0}); A.Set(1,1,{2,0});

    Complex dataB[] = {{2,2},{4,0}};
    auto* b = makeVec(dataB, 2);

    LUSolver<Complex> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residualC(A, *x, *b), 0.0, 1e-9);

    Complex expected[] = {{2,0},{2,0}};
    expectVecNearC(*x, expected, 2);
    delete x; delete b;
}

TEST(LUSolver, SolveComplex3x3Residual) {
    MutableSquareMatrix<Complex> A(3);
    A.Set(0,0,{2,1}); A.Set(0,1,{1,0}); A.Set(0,2,{0,0});
    A.Set(1,0,{1,0}); A.Set(1,1,{3,0}); A.Set(1,2,{1,1});
    A.Set(2,0,{0,0}); A.Set(2,1,{1,0}); A.Set(2,2,{2,0});

    Complex dataB[] = {{1,0},{2,0},{3,0}};
    auto* b = makeVec(dataB, 3);

    LUSolver<Complex> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residualC(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

//  QRSolver — базовые случаи

TEST(QRSolver, Solve2x2) {
    MutableSquareMatrix<double> A(2);
    double dataA[] = {2.0, 1.0,
                      5.0, 3.0};
    fillMatrix(A, dataA);
    double dataB[] = {8.0, 13.0};
    auto* b = makeVec(dataB, 2);

    QRSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    double expected[] = {11.0, -14.0};
    expectVecNear(*x, expected, 2);
    EXPECT_NEAR(residual(A, *x, *b), 0.0, TOL);
    delete x; delete b;
}

TEST(QRSolver, Solve3x3Residual) {
    MutableSquareMatrix<double> A(3);
    double dataA[] = {1,2,3,
                      0,1,4,
                      5,6,0};
    fillMatrix(A, dataA);
    double dataB[] = {14.0, 11.0, 2.0};
    auto* b = makeVec(dataB, 3);

    QRSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

TEST(QRSolver, SolveIdentity) {
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,1); A.Set(1,1,1); A.Set(2,2,1);
    double dataB[] = {3.0, -1.0, 7.0};
    auto* b = makeVec(dataB, 3);

    QRSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    expectVecNear(*x, dataB, 3);
    delete x; delete b;
}

TEST(QRSolver, SolveDiagonal) {
    MutableSquareMatrix<double> A(3);
    A.Set(0,0,2); A.Set(1,1,3); A.Set(2,2,5);
    double dataB[]    = {6.0, 9.0, 10.0};
    double expected[] = {3.0, 3.0,  2.0};
    auto* b = makeVec(dataB, 3);

    QRSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    expectVecNear(*x, expected, 3);
    delete x; delete b;
}

TEST(QRSolver, Solve4x4Residual) {
    MutableSquareMatrix<double> A(4);
    double dataA[] = { 4, 3, 2, 1,
                       3, 4, 3, 2,
                       2, 3, 4, 3,
                       1, 2, 3, 4 };
    fillMatrix(A, dataA);
    double dataB[] = {10.0, 12.0, 12.0, 10.0};
    auto* b = makeVec(dataB, 4);

    QRSolver<double> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

//  QRSolver — повторное использование кэша

TEST(QRSolver, CacheReuseMultipleRHS) {
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);

    QRSolver<double> solver(A);

    double b1[] = {1.0, 1.0, 1.0};
    double b2[] = {0.0, 1.0, 0.0};
    double b3[] = {3.0,-1.0, 2.0};

    for (auto* bArr : {b1, b2, b3}) {
        auto* b = makeVec(bArr, 3);
        auto* x = solver.Solve(*b);
        EXPECT_NEAR(residual(A, *x, *b), 0.0, 1e-9);
        delete x; delete b;
    }
}

TEST(QRSolver, SetMatrixInvalidatesCache) {
    MutableSquareMatrix<double> A1(2), A2(2);
    A1.Set(0,0,2); A1.Set(0,1,0); A1.Set(1,0,0); A1.Set(1,1,3);
    A2.Set(0,0,1); A2.Set(0,1,2); A2.Set(1,0,2); A2.Set(1,1,1);

    double dataB[] = {2.0, 3.0};
    auto* b = makeVec(dataB, 2);

    QRSolver<double> solver(A1);
    auto* x1 = solver.Solve(*b);

    solver.SetMatrix(A2);
    auto* x2 = solver.Solve(*b);

    EXPECT_NEAR(residual(A1, *x1, *b), 0.0, TOL);
    EXPECT_NEAR(residual(A2, *x2, *b), 0.0, TOL);
    EXPECT_GT(std::fabs(x1->Get(0) - x2->Get(0)), 1e-6);

    delete x1; delete x2; delete b;
}

//  QRSolver — доступ к Q и R

TEST(QRSolver, RisUpperTriangular) {
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);
    QRSolver<double> solver(A);

    const SquareMatrix<double>& R = solver.GetR();
    int n = R.GetSize();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            EXPECT_NEAR(R.Get(i,j), 0.0, TOL);
}

TEST(QRSolver, QisOrthogonal) {
    // Q^T * Q должна быть единичной матрицей
    MutableSquareMatrix<double> A(3);
    double dataA[] = {2,1,1,
                      4,3,3,
                      8,7,9};
    fillMatrix(A, dataA);
    QRSolver<double> solver(A);

    const SquareMatrix<double>& Q = solver.GetQ();
    int n = Q.GetSize();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double dot = 0.0;
            for (int k = 0; k < n; ++k) dot += Q.Get(k,i) * Q.Get(k,j);
            EXPECT_NEAR(dot, (i == j) ? 1.0 : 0.0, 1e-9);
        }
    }
}

TEST(QRSolver, QRProductEqualsA) {
    // Q * R должна восстанавливать A
    MutableSquareMatrix<double> A(3);
    double dataA[] = {1,2,3,
                      0,1,4,
                      5,6,0};
    fillMatrix(A, dataA);
    QRSolver<double> solver(A);

    const SquareMatrix<double>& Q = solver.GetQ();
    const SquareMatrix<double>& R = solver.GetR();
    int n = A.GetSize();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double qr = 0.0;
            for (int k = 0; k < n; ++k) qr += Q.Get(i,k) * R.Get(k,j);
            EXPECT_NEAR(qr, A.Get(i,j), 1e-9);
        }
    }
}

TEST(QRSolver, GetSizeMatchesMatrix) {
    MutableSquareMatrix<double> A(5);
    for (int i = 0; i < 5; ++i) A.Set(i,i,1.0);
    QRSolver<double> solver(A);
    EXPECT_EQ(solver.GetSize(), 5);
}

//  QRSolver — обработка ошибок

TEST(QRSolver, SingularMatrixThrows) {
    MutableSquareMatrix<double> A(2);   // нулевая матрица
    QRSolver<double> solver(A);
    double dataB[] = {1.0, 1.0};
    auto* b = makeVec(dataB, 2);
    EXPECT_THROW(solver.Solve(*b), RankDeficientException);
    delete b;
}

TEST(QRSolver, WrongRHSLengthThrows) {
    MutableSquareMatrix<double> A(3);
    for (int i = 0; i < 3; ++i) A.Set(i,i,1.0);
    double dataB[] = {1.0, 2.0};
    auto* b = makeVec(dataB, 2);
    QRSolver<double> solver(A);
    EXPECT_THROW(solver.Solve(*b), RHSLengthMismatch);
    delete b;
}

//  QRSolver — тип Complex

TEST(QRSolver, SolveComplex2x2) {
    MutableSquareMatrix<Complex> A(2);
    A.Set(0,0,{1,1}); A.Set(0,1,{0,0});
    A.Set(1,0,{0,0}); A.Set(1,1,{2,0});

    Complex dataB[] = {{2,2},{4,0}};
    auto* b = makeVec(dataB, 2);

    QRSolver<Complex> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residualC(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

TEST(QRSolver, SolveComplex3x3Residual) {
    MutableSquareMatrix<Complex> A(3);
    A.Set(0,0,{2,1}); A.Set(0,1,{1,0}); A.Set(0,2,{0,0});
    A.Set(1,0,{1,0}); A.Set(1,1,{3,0}); A.Set(1,2,{1,1});
    A.Set(2,0,{0,0}); A.Set(2,1,{1,0}); A.Set(2,2,{2,0});

    Complex dataB[] = {{1,0},{2,0},{3,0}};
    auto* b = makeVec(dataB, 3);

    QRSolver<Complex> solver(A);
    auto* x = solver.Solve(*b);

    EXPECT_NEAR(residualC(A, *x, *b), 0.0, 1e-9);
    delete x; delete b;
}

//  Сравнение LU и QR: оба дают одинаковый ответ

TEST(Solvers, LUandQRAgree2x2) {
    MutableSquareMatrix<double> A(2);
    double dataA[] = {3.0, 1.0,
                      1.0, 2.0};
    fillMatrix(A, dataA);
    double dataB[] = {9.0, 8.0};
    auto* b = makeVec(dataB, 2);

    LUSolver<double> lu(A);
    QRSolver<double> qr(A);
    auto* xlu = lu.Solve(*b);
    auto* xqr = qr.Solve(*b);

    ASSERT_EQ(xlu->GetLength(), xqr->GetLength());
    for (int i = 0; i < xlu->GetLength(); ++i)
        EXPECT_NEAR(xlu->Get(i), xqr->Get(i), 1e-9);

    delete xlu; delete xqr; delete b;
}

TEST(Solvers, LUandQRAgree4x4) {
    MutableSquareMatrix<double> A(4);
    double dataA[] = { 4, 3, 2, 1,
                       3, 4, 3, 2,
                       2, 3, 4, 3,
                       1, 2, 3, 4 };
    fillMatrix(A, dataA);
    double dataB[] = {10.0, 12.0, 12.0, 10.0};
    auto* b = makeVec(dataB, 4);

    LUSolver<double> lu(A);
    QRSolver<double> qr(A);
    auto* xlu = lu.Solve(*b);
    auto* xqr = qr.Solve(*b);

    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(xlu->Get(i), xqr->Get(i), 1e-9);

    delete xlu; delete xqr; delete b;
}

TEST(Solvers, LUandQRAgreeComplex) {
    MutableSquareMatrix<Complex> A(2);
    A.Set(0,0,{2,1}); A.Set(0,1,{1,0});
    A.Set(1,0,{1,0}); A.Set(1,1,{3,0});

    Complex dataB[] = {{1,0},{2,0}};
    auto* b = makeVec(dataB, 2);

    LUSolver<Complex> lu(A);
    QRSolver<Complex> qr(A);
    auto* xlu = lu.Solve(*b);
    auto* xqr = qr.Solve(*b);

    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(xlu->Get(i).re, xqr->Get(i).re, 1e-9);
        EXPECT_NEAR(xlu->Get(i).im, xqr->Get(i).im, 1e-9);
    }

    delete xlu; delete xqr; delete b;
}
