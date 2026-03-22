#include <gtest/gtest.h>
#include "../src/ArraySequence.h"
#include "../src/ListSequence.h"

// вспомогательные функции
static int    doubleIt(int x)          { return x * 2; }
static bool   isEven(int x)            { return x % 2 == 0; }
static int    sum2(int acc, int x)     { return acc + x; }
static int    mul2(int acc, int x)     { return acc * x; }
static int    addIdx(int x, int idx)   { return x + idx; }

static Sequence<int>* repeat2(int x) {
    auto* s = new MutableArraySequence<int>();
    s->Append(x); s->Append(x);
    return s;
}

static bool isZero(int x) { return x == 0; }

// Map
TEST(MapReduce, Map) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    auto* res = seq.Map(doubleIt);
    EXPECT_EQ(res->Get(0), 2);
    EXPECT_EQ(res->Get(1), 4);
    EXPECT_EQ(res->Get(2), 6);
    delete res;
}

// Where
TEST(MapReduce, Where) {
    int arr[] = {1, 2, 3, 4, 5};
    MutableArraySequence<int> seq(arr, 5);
    auto* res = seq.Where(isEven);
    EXPECT_EQ(res->GetLength(), 2);
    EXPECT_EQ(res->Get(0), 2);
    EXPECT_EQ(res->Get(1), 4);
    delete res;
}

// Reduce
TEST(MapReduce, ReduceSum) {
    int arr[] = {1, 2, 3, 4, 5};
    MutableArraySequence<int> seq(arr, 5);
    EXPECT_EQ(seq.Reduce(sum2, 0), 15);
}

TEST(MapReduce, ReduceMul) {
    int arr[] = {1, 2, 3, 4};
    MutableArraySequence<int> seq(arr, 4);
    EXPECT_EQ(seq.Reduce(mul2, 1), 24);
}

// MapIndexed
TEST(MapReduce, MapIndexed) {
    int arr[] = {10, 20, 30};
    MutableArraySequence<int> seq(arr, 3);
    auto* res = seq.MapIndexed(addIdx);
    EXPECT_EQ(res->Get(0), 10); // 10 + 0
    EXPECT_EQ(res->Get(1), 21); // 20 + 1
    EXPECT_EQ(res->Get(2), 32); // 30 + 2
    delete res;
}

// Find
TEST(MapReduce, Find) {
    int arr[] = {1, 3, 4, 5};
    MutableArraySequence<int> seq(arr, 4);
    auto found = seq.Find(isEven);
    ASSERT_TRUE(found.HasValue());
    EXPECT_EQ(found.GetValue(), 4);
}

TEST(MapReduce, FindNotFound) {
    int arr[] = {1, 3, 5};
    MutableArraySequence<int> seq(arr, 3);
    EXPECT_FALSE(seq.Find(isEven).HasValue());
}

// FlatMap
TEST(MapReduce, FlatMap) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    auto* res = seq.FlatMap(repeat2);
    EXPECT_EQ(res->GetLength(), 6);
    EXPECT_EQ(res->Get(0), 1);
    EXPECT_EQ(res->Get(1), 1);
    EXPECT_EQ(res->Get(2), 2);
    delete res;
}

// Split
TEST(MapReduce, Split) {
    int arr[] = {1, 0, 2, 3, 0, 4};
    MutableArraySequence<int> seq(arr, 6);
    auto* parts = seq.Split(isZero);
    EXPECT_EQ(parts->GetLength(), 3);
    EXPECT_EQ(parts->Get(0)->GetLength(), 1); // [1]
    EXPECT_EQ(parts->Get(1)->GetLength(), 2); // [2, 3]
    EXPECT_EQ(parts->Get(2)->GetLength(), 1); // [4]
    for (int i = 0; i < parts->GetLength(); ++i) delete parts->Get(i);
    delete parts;
}

// Slice
TEST(MapReduce, Slice) {
    int arr[] = {1, 2, 3, 4, 5};
    MutableArraySequence<int> seq(arr, 5);
    auto* res = seq.Slice(1, 2); // удалить 2 элемента с позиции 1
    EXPECT_EQ(res->GetLength(), 3);
    EXPECT_EQ(res->Get(0), 1);
    EXPECT_EQ(res->Get(1), 4);
    delete res;
}

TEST(MapReduce, SliceNegativeIndex) {
    int arr[] = {1, 2, 3, 4, 5};
    MutableArraySequence<int> seq(arr, 5);
    auto* res = seq.Slice(-2, 2); // с конца: позиция 3
    EXPECT_EQ(res->GetLength(), 3);
    delete res;
}

TEST(MapReduce, SliceWithInsert) {
    int arr[] = {1, 2, 3, 4, 5};
    int ins[]  = {10, 20};
    MutableArraySequence<int> seq(arr, 5);
    MutableArraySequence<int> toIns(ins, 2);
    auto* res = seq.Slice(1, 2, &toIns); // заменить [2,3] на [10,20]
    EXPECT_EQ(res->GetLength(), 5);
    EXPECT_EQ(res->Get(1), 10);
    EXPECT_EQ(res->Get(2), 20);
    delete res;
}

// Zip / Unzip
TEST(MapReduce, Zip) {
    int a[] = {1, 2, 3}, b[] = {4, 5, 6};
    MutableArraySequence<int> sa(a, 3), sb(b, 3);
    auto* zipped = Zip<int, int>(&sa, &sb);
    EXPECT_EQ(zipped->GetLength(), 3);
    EXPECT_EQ(zipped->Get(0).first,  1);
    EXPECT_EQ(zipped->Get(0).second, 4);
    delete zipped;
}

TEST(MapReduce, Unzip) {
    int a[] = {1, 2}, b[] = {3, 4};
    MutableArraySequence<int> sa(a, 2), sb(b, 2);
    auto* zipped = Zip<int, int>(&sa, &sb);
    auto result = Unzip<int, int>(zipped);
    EXPECT_EQ(result.first->Get(0),  1);
    EXPECT_EQ(result.second->Get(0), 3);
    delete result.first;
    delete result.second;
    delete zipped;
}
