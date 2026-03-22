#include <gtest/gtest.h>
#include <cmath>
#include "../src/Matrix.h"

static constexpr double TOL = 1e-9;

// helpers

static void fill2x2i(SquareMatrix<int>& m) {
    m.Set(0,0,1); m.Set(0,1,2);
    m.Set(1,0,3); m.Set(1,1,4);
}

static void fill2x2d(SquareMatrix<double>& m) {
    m.Set(0,0,1.0); m.Set(0,1,2.0);
    m.Set(1,0,3.0); m.Set(1,1,4.0);
}

static void fill2x2c(SquareMatrix<Complex>& m) {
    m.Set(0,0,{1,0}); m.Set(0,1,{0,1});
    m.Set(1,0,{2,2}); m.Set(1,1,{3,0});
}

static void fill3x3d(SquareMatrix<double>& m) {
    m.Set(0,0,1); m.Set(0,1,2); m.Set(0,2,3);
    m.Set(1,0,4); m.Set(1,1,5); m.Set(1,2,6);
    m.Set(2,0,7); m.Set(2,1,8); m.Set(2,2,9);
}

//  Конструирование и базовый доступ

TEST(SquareMatrix, DefaultZeroInit) {
    MutableSquareMatrix<int> m(3);
    EXPECT_EQ(m.GetSize(), 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(m.Get(i, j), 0);
}

TEST(SquareMatrix, ZeroSize) {
    MutableSquareMatrix<double> m(0);
    EXPECT_EQ(m.GetSize(), 0);
}

TEST(SquareMatrix, NegativeSizeThrows) {
    EXPECT_THROW(MutableSquareMatrix<int>(-1),   InvalidMatrixSize);
    EXPECT_THROW(MutableSquareMatrix<double>(-5), InvalidMatrixSize);
}

TEST(SquareMatrix, SetAndGet) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    EXPECT_EQ(m.Get(0,0), 1);
    EXPECT_EQ(m.Get(0,1), 2);
    EXPECT_EQ(m.Get(1,0), 3);
    EXPECT_EQ(m.Get(1,1), 4);
}

TEST(SquareMatrix, OutOfRangeGet) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.Get(-1,  0), RowIndexOutOfRange);
    EXPECT_THROW(m.Get( 2, 0), RowIndexOutOfRange);
    EXPECT_THROW(m.Get( 0,  -1), ColIndexOutOfRange);
    EXPECT_THROW(m.Get( 0,  2), ColIndexOutOfRange);
}

TEST(SquareMatrix, OutOfRangeSet) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.Set(-1,  0, 1), RowIndexOutOfRange);
    EXPECT_THROW(m.Set( 2,  0, 1), RowIndexOutOfRange);
    EXPECT_THROW(m.Set( 0, -1, 1), ColIndexOutOfRange);
    EXPECT_THROW(m.Set( 0,  2, 1), ColIndexOutOfRange);
}

TEST(SquareMatrix, OverwriteValue) {
    MutableSquareMatrix<int> m(2);
    m.Set(0, 0, 10);
    m.Set(0, 0, 99);
    EXPECT_EQ(m.Get(0, 0), 99);
}

//  IEnumerable / IEnumerator: row-major обход

TEST(SquareMatrix, EnumeratorOrder) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    int expected[] = {1, 2, 3, 4};
    int idx = 0;
    for (int v : m) EXPECT_EQ(v, expected[idx++]);
    EXPECT_EQ(idx, 4);
}

TEST(SquareMatrix, EnumeratorSum) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    int sum = 0;
    for (int v : m) sum += v;
    EXPECT_EQ(sum, 10);
}

TEST(SquareMatrix, EnumeratorEmptyMatrix) {
    MutableSquareMatrix<int> m(0);
    int count = 0;
    for (int v : m) { (void)v; ++count; }
    EXPECT_EQ(count, 0);
}

TEST(SquareMatrix, EnumeratorReset) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    auto* en = m.GetEnumerator();
    ASSERT_TRUE(en->MoveNext());
    EXPECT_EQ(en->Current(), 1);
    en->Reset();
    ASSERT_TRUE(en->MoveNext());
    EXPECT_EQ(en->Current(), 1);   // после Reset начинаем заново
    delete en;
}

TEST(SquareMatrix, Enumerator3x3Count) {
    MutableSquareMatrix<double> m(3);
    fill3x3d(m);
    int count = 0;
    for (double v : m) { (void)v; ++count; }
    EXPECT_EQ(count, 9);
}

//  Арифметика

TEST(SquareMatrix, AddInt) {
    MutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a);
    b.Set(0,0,5); b.Set(0,1,6); b.Set(1,0,7); b.Set(1,1,8);
    auto* c = a.Add(&b);
    EXPECT_EQ(c->Get(0,0),  6);
    EXPECT_EQ(c->Get(0,1),  8);
    EXPECT_EQ(c->Get(1,0), 10);
    EXPECT_EQ(c->Get(1,1), 12);
    delete c;
}

TEST(SquareMatrix, AddDouble) {
    MutableSquareMatrix<double> a(2), b(2);
    fill2x2d(a); fill2x2d(b);
    auto* c = a.Add(&b);
    EXPECT_NEAR(c->Get(0,0), 2.0, TOL);
    EXPECT_NEAR(c->Get(1,1), 8.0, TOL);
    delete c;
}

TEST(SquareMatrix, AddComplex) {
    MutableSquareMatrix<Complex> a(2), b(2);
    fill2x2c(a);
    b.Set(0,0,{1,0}); b.Set(0,1,{0,-1});
    b.Set(1,0,{0,1}); b.Set(1,1,{-3,0});
    auto* c = a.Add(&b);
    EXPECT_EQ(c->Get(0,0), Complex(2,0));
    EXPECT_EQ(c->Get(0,1), Complex(0,0));
    EXPECT_EQ(c->Get(1,0), Complex(2,3));
    EXPECT_EQ(c->Get(1,1), Complex(0,0));
    delete c;
}

TEST(SquareMatrix, AddSizeMismatch) {
    MutableSquareMatrix<int> a(2), b(3);
    EXPECT_THROW(a.Add(&b), MatrixSizeMismatch);
}

TEST(SquareMatrix, AddDoesNotMutateOperands) {
    MutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a); fill2x2i(b);
    auto* c = a.Add(&b);
    EXPECT_EQ(a.Get(0,0), 1);   // a не изменился
    EXPECT_EQ(b.Get(0,0), 1);   // b не изменился
    delete c;
}

TEST(SquareMatrix, MulScalarInt) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    auto* r = m.MulScalar(3);
    EXPECT_EQ(r->Get(0,0),  3);
    EXPECT_EQ(r->Get(0,1),  6);
    EXPECT_EQ(r->Get(1,0),  9);
    EXPECT_EQ(r->Get(1,1), 12);
    delete r;
}

TEST(SquareMatrix, MulScalarDouble) {
    MutableSquareMatrix<double> m(2);
    fill2x2d(m);
    auto* r = m.MulScalar(0.5);
    EXPECT_NEAR(r->Get(0,0), 0.5, TOL);
    EXPECT_NEAR(r->Get(1,1), 2.0, TOL);
    delete r;
}

TEST(SquareMatrix, MulScalarComplex) {
    MutableSquareMatrix<Complex> m(1);
    m.Set(0, 0, {1, 2});
    auto* r = m.MulScalar({0, 1});   // (1+2i)*i = -2+i
    EXPECT_NEAR(r->Get(0,0).re, -2.0, TOL);
    EXPECT_NEAR(r->Get(0,0).im,  1.0, TOL);
    delete r;
}

TEST(SquareMatrix, MulScalarZero) {
    MutableSquareMatrix<double> m(2);
    fill2x2d(m);
    auto* r = m.MulScalar(0.0);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            EXPECT_NEAR(r->Get(i,j), 0.0, TOL);
    delete r;
}

// Норма Фробениуса

TEST(SquareMatrix, NormInt) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    EXPECT_NEAR(m.Norm(), std::sqrt(30.0), TOL);  // 1+4+9+16 = 30
}

TEST(SquareMatrix, NormDouble) {
    MutableSquareMatrix<double> m(2);
    fill2x2d(m);
    EXPECT_NEAR(m.Norm(), std::sqrt(30.0), TOL);
}

TEST(SquareMatrix, NormComplex) {
    // |1+0i|^2 + |0+1i|^2 + |2+2i|^2 + |3+0i|^2 = 1+1+8+9 = 19
    MutableSquareMatrix<Complex> m(2);
    fill2x2c(m);
    EXPECT_NEAR(m.Norm(), std::sqrt(19.0), TOL);
}

TEST(SquareMatrix, NormSingleComplex) {
    MutableSquareMatrix<Complex> m(1);
    m.Set(0, 0, {3.0, 4.0});   // |3+4i| = 5
    EXPECT_NEAR(m.Norm(), 5.0, TOL);
}

TEST(SquareMatrix, NormZeroMatrix) {
    MutableSquareMatrix<double> m(3);   // all zeros
    EXPECT_NEAR(m.Norm(), 0.0, TOL);
}

//  Элементарные преобразования строк

TEST(SquareMatrix, SwapRows) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.SwapRows(0, 1);
    EXPECT_EQ(m.Get(0,0), 3); EXPECT_EQ(m.Get(0,1), 4);
    EXPECT_EQ(m.Get(1,0), 1); EXPECT_EQ(m.Get(1,1), 2);
}

TEST(SquareMatrix, SwapRowsSelf) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.SwapRows(0, 0);
    EXPECT_EQ(m.Get(0,0), 1); EXPECT_EQ(m.Get(0,1), 2);
}

TEST(SquareMatrix, SwapRows3x3) {
    MutableSquareMatrix<double> m(3);
    fill3x3d(m);
    m.SwapRows(0, 2);
    EXPECT_NEAR(m.Get(0,0), 7.0, TOL);
    EXPECT_NEAR(m.Get(2,0), 1.0, TOL);
}

TEST(SquareMatrix, SwapRowsOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.SwapRows(-1, 0), RowIndexOutOfRange);
    EXPECT_THROW(m.SwapRows( 0, 2), RowIndexOutOfRange);
}

TEST(SquareMatrix, MulRow) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.MulRow(0, 3);
    EXPECT_EQ(m.Get(0,0), 3); EXPECT_EQ(m.Get(0,1), 6);
    EXPECT_EQ(m.Get(1,0), 3); EXPECT_EQ(m.Get(1,1), 4);
}

TEST(SquareMatrix, MulRowZero) {
    MutableSquareMatrix<double> m(2);
    fill2x2d(m);
    m.MulRow(1, 0.0);
    EXPECT_NEAR(m.Get(1,0), 0.0, TOL);
    EXPECT_NEAR(m.Get(1,1), 0.0, TOL);
}

TEST(SquareMatrix, MulRowOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.MulRow(-1, 2), RowIndexOutOfRange);
    EXPECT_THROW(m.MulRow( 2, 2), RowIndexOutOfRange);
}

TEST(SquareMatrix, AddRow) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    // row[0] += 2*row[1]: [1+6, 2+8] = [7, 10]
    m.AddRow(0, 1, 2);
    EXPECT_EQ(m.Get(0,0), 7); EXPECT_EQ(m.Get(0,1), 10);
    EXPECT_EQ(m.Get(1,0), 3); EXPECT_EQ(m.Get(1,1),  4);
}

TEST(SquareMatrix, AddRowSameIndex) {
    // row[0] += 1*row[0] -> row[0] *= 2
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.AddRow(0, 0, 1);
    EXPECT_EQ(m.Get(0,0), 2); EXPECT_EQ(m.Get(0,1), 4);
}

TEST(SquareMatrix, AddRowOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.AddRow(-1, 0, 1), RowIndexOutOfRange);
    EXPECT_THROW(m.AddRow( 0, 2, 1), RowIndexOutOfRange);
}

//  Элементарные преобразования столбцов

TEST(SquareMatrix, SwapCols) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.SwapCols(0, 1);
    EXPECT_EQ(m.Get(0,0), 2); EXPECT_EQ(m.Get(0,1), 1);
    EXPECT_EQ(m.Get(1,0), 4); EXPECT_EQ(m.Get(1,1), 3);
}

TEST(SquareMatrix, SwapColsSelf) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.SwapCols(1, 1);
    EXPECT_EQ(m.Get(0,1), 2);
}

TEST(SquareMatrix, SwapColsOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.SwapCols(-1, 0), ColIndexOutOfRange);
    EXPECT_THROW(m.SwapCols( 0, 2), ColIndexOutOfRange);
}

TEST(SquareMatrix, MulCol) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.MulCol(1, 2);
    EXPECT_EQ(m.Get(0,0), 1); EXPECT_EQ(m.Get(0,1), 4);
    EXPECT_EQ(m.Get(1,0), 3); EXPECT_EQ(m.Get(1,1), 8);
}

TEST(SquareMatrix, MulColOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.MulCol(-1, 2), ColIndexOutOfRange);
    EXPECT_THROW(m.MulCol( 2, 2), ColIndexOutOfRange);
}

TEST(SquareMatrix, AddCol) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    // col[0] += 2*col[1]: [1+4, 3+8] = [5, 11]
    m.AddCol(0, 1, 2);
    EXPECT_EQ(m.Get(0,0),  5); EXPECT_EQ(m.Get(0,1), 2);
    EXPECT_EQ(m.Get(1,0), 11); EXPECT_EQ(m.Get(1,1), 4);
}

TEST(SquareMatrix, AddColSameIndex) {
    // col[1] += 1*col[1] → col[1] *= 2
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.AddCol(1, 1, 1);
    EXPECT_EQ(m.Get(0,1), 4);
    EXPECT_EQ(m.Get(1,1), 8);
}

TEST(SquareMatrix, AddColOutOfRange) {
    MutableSquareMatrix<int> m(2);
    EXPECT_THROW(m.AddCol(-1, 0, 1), ColIndexOutOfRange);
    EXPECT_THROW(m.AddCol( 0, 2, 1), ColIndexOutOfRange);
}

//  Паттерн Mutable

TEST(MutablePattern, InstanceReturnsSelf) {
    MutableSquareMatrix<int> m(2);
    EXPECT_EQ(m.Instance(), static_cast<SquareMatrix<int>*>(&m));
}

TEST(MutablePattern, CloneIsDeepCopy) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    auto* c = m.Clone();
    c->Set(0, 0, 99);
    EXPECT_EQ(m.Get(0,0), 1);   // оригинал не изменился
    delete c;
}

TEST(MutablePattern, SwapRowsMutatesInPlace) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    SquareMatrix<int>* r = m.SwapRows(0, 1);
    EXPECT_EQ(r, static_cast<SquareMatrix<int>*>(&m));
    EXPECT_EQ(m.Get(0,0), 3);
}

TEST(MutablePattern, ChainedRowOps) {
    MutableSquareMatrix<int> m(2);
    fill2x2i(m);
    m.SwapRows(0, 1);
    m.MulRow(0, 2);
    EXPECT_EQ(m.Get(0,0), 6);
    EXPECT_EQ(m.Get(0,1), 8);
}

//  Паттерн Immutable

TEST(ImmutablePattern, InstanceReturnsClone) {
    ImmutableSquareMatrix<int> m(2);
    auto* inst = m.Instance();
    EXPECT_NE(inst, static_cast<const SquareMatrix<int>*>(&m));
    delete inst;
}

TEST(ImmutablePattern, SwapRowsLeavesOriginalUnchanged) {
    ImmutableSquareMatrix<int> m(2);
    fill2x2i(m);
    SquareMatrix<int>* r = m.SwapRows(0, 1);
    EXPECT_EQ(m.Get(0,0), 1); EXPECT_EQ(m.Get(0,1), 2);  // оригинал цел
    EXPECT_EQ(r->Get(0,0), 3); EXPECT_EQ(r->Get(0,1), 4); // клон изменён
    delete r;
}

TEST(ImmutablePattern, MulRowLeavesOriginalUnchanged) {
    ImmutableSquareMatrix<double> m(2);
    fill2x2d(m);
    SquareMatrix<double>* r = m.MulRow(0, 10.0);
    EXPECT_NEAR(m.Get(0,0),  1.0, TOL);
    EXPECT_NEAR(r->Get(0,0), 10.0, TOL);
    delete r;
}

TEST(ImmutablePattern, AddRowLeavesOriginalUnchanged) {
    ImmutableSquareMatrix<int> m(2);
    fill2x2i(m);
    SquareMatrix<int>* r = m.AddRow(0, 1, 1);
    EXPECT_EQ(m.Get(0,0), 1);
    EXPECT_EQ(r->Get(0,0), 4);
    delete r;
}

TEST(ImmutablePattern, ChainedOpsAllIndependent) {
    ImmutableSquareMatrix<int> m(2);
    fill2x2i(m);
    SquareMatrix<int>* r1 = m.SwapRows(0, 1);
    SquareMatrix<int>* r2 = m.MulRow(0, 5);
    SquareMatrix<int>* r3 = m.AddRow(0, 1, 1);
    EXPECT_EQ(m.Get(0,0), 1);    // оригинал всегда цел
    EXPECT_EQ(r1->Get(0,0), 3);
    EXPECT_EQ(r2->Get(0,0), 5);
    EXPECT_EQ(r3->Get(0,0), 4);
    delete r1; delete r2; delete r3;
}

TEST(ImmutablePattern, AddPreservesImmutability) {
    ImmutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a); fill2x2i(b);
    SquareMatrix<int>* c = a.Add(&b);
    // результат Add — ImmutableSquareMatrix (CreateEmpty) -> Instance() клонирует
    SquareMatrix<int>* inst = c->Instance();
    EXPECT_NE(inst, c);
    delete inst;
    delete c;
}

//  Равенство

TEST(SquareMatrix, EqualityTrue) {
    MutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a); fill2x2i(b);
    EXPECT_TRUE(a == b);
}

TEST(SquareMatrix, EqualityFalseValue) {
    MutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a); fill2x2i(b);
    b.Set(0, 0, 99);
    EXPECT_FALSE(a == b);
}

TEST(SquareMatrix, EqualityFalseSize) {
    MutableSquareMatrix<int> a(2), b(3);
    EXPECT_FALSE(a == b);
}

TEST(SquareMatrix, InequalityOperator) {
    MutableSquareMatrix<int> a(2), b(2);
    fill2x2i(a); fill2x2i(b);
    EXPECT_FALSE(a != b);
    b.Set(1, 1, 0);
    EXPECT_TRUE(a != b);
}

TEST(SquareMatrix, EqualityMutableVsImmutable) {
    MutableSquareMatrix<int>   a(2);
    ImmutableSquareMatrix<int> b(2);
    fill2x2i(a); fill2x2i(b);
    EXPECT_TRUE(a == b);
}
