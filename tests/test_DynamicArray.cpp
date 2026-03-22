#include <gtest/gtest.h>
#include "../src/DynamicArray.h"

// Конструкторы
TEST(DynamicArray, ConstructFromArray) {
    int arr[] = {1, 2, 3};
    DynamicArray<int> da(arr, 3);
    EXPECT_EQ(da.GetSize(), 3);
    EXPECT_EQ(da.Get(0), 1);
    EXPECT_EQ(da.Get(2), 3);
}

TEST(DynamicArray, ConstructBySize) {
    DynamicArray<int> da(5);
    EXPECT_EQ(da.GetSize(), 5);
}

TEST(DynamicArray, CopyConstructor) {
    int arr[] = {10, 20};
    DynamicArray<int> original(arr, 2);
    DynamicArray<int> copy(original);
    copy.Set(0, 99);
    EXPECT_EQ(original.Get(0), 10); // оригинал не изменился
    EXPECT_EQ(copy.Get(0), 99);
}

// Методы
TEST(DynamicArray, SetAndGet) {
    DynamicArray<int> da(3);
    da.Set(1, 42);
    EXPECT_EQ(da.Get(1), 42);
}

TEST(DynamicArray, ResizeGrow) {
    int arr[] = {1, 2, 3};
    DynamicArray<int> da(arr, 3);
    da.Resize(5);
    EXPECT_EQ(da.GetSize(), 5);
    EXPECT_EQ(da.Get(0), 1);
    EXPECT_EQ(da.Get(2), 3);
}

TEST(DynamicArray, ResizeShrink) {
    int arr[] = {1, 2, 3, 4, 5};
    DynamicArray<int> da(arr, 5);
    da.Resize(3);
    EXPECT_EQ(da.GetSize(), 3);
    EXPECT_EQ(da.Get(2), 3);
}

// Граничные случаи
TEST(DynamicArray, ZeroSize) {
    DynamicArray<int> da(0);
    EXPECT_EQ(da.GetSize(), 0);
}

TEST(DynamicArray, SingleElement) {
    int arr[] = {7};
    DynamicArray<int> da(arr, 1);
    EXPECT_EQ(da.Get(0), 7);
}

// Некорректные случаи
TEST(DynamicArray, GetOutOfRange) {
    DynamicArray<int> da(3);
    EXPECT_THROW(da.Get(-1),  IndexOutOfRange);
    EXPECT_THROW(da.Get(3),   IndexOutOfRange);
}

TEST(DynamicArray, SetOutOfRange) {
    DynamicArray<int> da(3);
    EXPECT_THROW(da.Set(-1, 0), IndexOutOfRange);
    EXPECT_THROW(da.Set(3,  0), IndexOutOfRange);
}

TEST(DynamicArray, NegativeSize) {
    EXPECT_THROW(DynamicArray<int> da(-1), IndexOutOfRange);
}

// Итератор
TEST(DynamicArray, Enumerator) {
    int arr[] = {3, 1, 4};
    DynamicArray<int> da(arr, 3);
    auto* en = da.GetEnumerator();
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 3);
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 1);
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 4);
    EXPECT_FALSE(en->MoveNext());
    delete en;
}
