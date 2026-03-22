#include <gtest/gtest.h>
#include "../src/LinkedList.h"

TEST(LinkedList, ConstructFromArray) {
    int arr[] = {5, 6, 7};
    LinkedList<int> ll(arr, 3);
    EXPECT_EQ(ll.GetLength(), 3);
    EXPECT_EQ(ll.GetFirst(), 5);
    EXPECT_EQ(ll.GetLast(),  7);
}

TEST(LinkedList, EmptyList) {
    LinkedList<int> ll;
    EXPECT_EQ(ll.GetLength(), 0);
    EXPECT_THROW(ll.GetFirst(), IndexOutOfRange);
    EXPECT_THROW(ll.GetLast(),  IndexOutOfRange);
}

TEST(LinkedList, CopyConstructor) {
    int arr[] = {1, 2, 3};
    LinkedList<int> ll(arr, 3);
    LinkedList<int> copy(ll);
    copy.Prepend(0);
    EXPECT_EQ(ll.GetLength(), 3); // оригинал не изменился
    EXPECT_EQ(copy.GetFirst(), 0);
}

TEST(LinkedList, AppendPrepend) {
    LinkedList<int> ll;
    ll.Append(10);
    ll.Append(20);
    ll.Prepend(5);
    EXPECT_EQ(ll.GetFirst(), 5);
    EXPECT_EQ(ll.GetLast(),  20);
    EXPECT_EQ(ll.GetLength(), 3);
}

TEST(LinkedList, InsertAt) {
    int arr[] = {1, 3};
    LinkedList<int> ll(arr, 2);
    ll.InsertAt(2, 1);
    EXPECT_EQ(ll.Get(1), 2);
    EXPECT_EQ(ll.GetLength(), 3);
}

TEST(LinkedList, GetSubList) {
    int arr[] = {0, 1, 2, 3, 4};
    LinkedList<int> ll(arr, 5);
    auto* sub = ll.GetSubList(1, 3);
    EXPECT_EQ(sub->GetLength(), 3);
    EXPECT_EQ(sub->Get(0), 1);
    EXPECT_EQ(sub->Get(2), 3);
    delete sub;
}

TEST(LinkedList, Concat) {
    int a[] = {1, 2}, b[] = {3, 4};
    LinkedList<int> la(a, 2), lb(b, 2);
    auto* cat = la.Concat(&lb);
    EXPECT_EQ(cat->GetLength(), 4);
    EXPECT_EQ(cat->Get(3), 4);
    delete cat;
}

TEST(LinkedList, OutOfRange) {
    int arr[] = {1, 2};
    LinkedList<int> ll(arr, 2);
    EXPECT_THROW(ll.Get(-1), IndexOutOfRange);
    EXPECT_THROW(ll.Get(2),  IndexOutOfRange);
    EXPECT_THROW(ll.InsertAt(0, -1), IndexOutOfRange);
    EXPECT_THROW(ll.InsertAt(0,  3), IndexOutOfRange);
}

TEST(LinkedList, Enumerator) {
    int arr[] = {7, 8, 9};
    LinkedList<int> ll(arr, 3);
    auto* en = ll.GetEnumerator();
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 7);
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 8);
    ASSERT_TRUE(en->MoveNext()); EXPECT_EQ(en->Current(), 9);
    EXPECT_FALSE(en->MoveNext());
    delete en;
}

TEST(LinkedList, RemoveFirst) {
    int arr[] = {1, 2, 3};
    LinkedList<int> ll(arr, 3);
    ll.RemoveFirst();
    EXPECT_EQ(ll.GetFirst(),  2);
    EXPECT_EQ(ll.GetLength(), 2);
}

TEST(LinkedList, RemoveLast) {
    int arr[] = {1, 2, 3};
    LinkedList<int> ll(arr, 3);
    ll.RemoveLast();
    EXPECT_EQ(ll.GetLast(),   2);
    EXPECT_EQ(ll.GetLength(), 2);
}

TEST(LinkedList, RemoveAt) {
    int arr[] = {1, 2, 3};
    LinkedList<int> ll(arr, 3);
    ll.RemoveAt(1);
    EXPECT_EQ(ll.Get(1),      3);
    EXPECT_EQ(ll.GetLength(), 2);
}

TEST(LinkedList, RemoveSingleElement) {
    int arr[] = {42};
    LinkedList<int> ll(arr, 1);
    ll.RemoveFirst();
    EXPECT_EQ(ll.GetLength(), 0);
    EXPECT_THROW(ll.GetFirst(), IndexOutOfRange);
}

TEST(LinkedList, RemoveFirstEmpty) {
    LinkedList<int> ll;
    EXPECT_THROW(ll.RemoveFirst(), IndexOutOfRange);
}

TEST(LinkedList, RemoveLastEmpty) {
    LinkedList<int> ll;
    EXPECT_THROW(ll.RemoveLast(), IndexOutOfRange);
}

TEST(LinkedList, RemoveAtOutOfRange) {
    int arr[] = {1, 2};
    LinkedList<int> ll(arr, 2);
    EXPECT_THROW(ll.RemoveAt(-1), IndexOutOfRange);
    EXPECT_THROW(ll.RemoveAt(2),  IndexOutOfRange);
}
