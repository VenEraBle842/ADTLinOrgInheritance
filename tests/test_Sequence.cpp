#include <gtest/gtest.h>
#include "../src/ArraySequence.h"
#include "../src/ListSequence.h"

// MutableArraySequence
TEST(MutableArraySeq, BasicOps) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    EXPECT_EQ(seq.GetLength(), 3);
    EXPECT_EQ(seq.GetFirst(), 1);
    EXPECT_EQ(seq.GetLast(),  3);
    seq.Append(4);
    EXPECT_EQ(seq.GetLast(), 4);
    seq.Prepend(0);
    EXPECT_EQ(seq.GetFirst(), 0);
}

TEST(MutableArraySeq, InsertAt) {
    int arr[] = {1, 3};
    MutableArraySequence<int> seq(arr, 2);
    seq.InsertAt(2, 1);
    EXPECT_EQ(seq.Get(1), 2);
}

TEST(MutableArraySeq, GetSubsequence) {
    int arr[] = {10, 20, 30, 40};
    MutableArraySequence<int> seq(arr, 4);
    auto* sub = seq.GetSubsequence(1, 2);
    EXPECT_EQ(sub->GetLength(), 2);
    EXPECT_EQ(sub->Get(0), 20);
    delete sub;
}

TEST(MutableArraySeq, Concat) {
    int a[] = {1, 2}, b[] = {3, 4};
    MutableArraySequence<int> sa(a, 2), sb(b, 2);
    auto* cat = sa.Concat(&sb);
    EXPECT_EQ(cat->GetLength(), 4);
    EXPECT_EQ(cat->Get(2), 3);
    // sa изменена на месте, cat == &sa
}

TEST(MutableArraySeq, OperatorEquals) {
    int a[] = {1, 2, 3}, b[] = {1, 2, 3};
    MutableArraySequence<int> sa(a, 3), sb(b, 3);
    EXPECT_TRUE(sa == sb);
    sb.Append(4);
    EXPECT_FALSE(sa == sb);
}

// ImmutableArraySequence
TEST(ImmutableArraySeq, AppendDoesNotMutate) {
    int arr[] = {1, 2, 3};
    ImmutableArraySequence<int> seq(arr, 3);
    auto* newSeq = seq.Append(4);
    EXPECT_EQ(seq.GetLength(), 3);     // оригинал не изменился
    EXPECT_EQ(newSeq->GetLength(), 4); // новый объект с добавленным элементом
    delete newSeq;
}

TEST(ImmutableArraySeq, PrependDoesNotMutate) {
    int arr[] = {2, 3};
    ImmutableArraySequence<int> seq(arr, 2);
    auto* newSeq = seq.Prepend(1);
    EXPECT_EQ(seq.GetFirst(), 2);
    EXPECT_EQ(newSeq->GetFirst(), 1);
    delete newSeq;
}

// MutableListSequence
TEST(MutableListSeq, BasicOps) {
    int arr[] = {5, 10, 15};
    MutableListSequence<int> seq(arr, 3);
    EXPECT_EQ(seq.GetLength(), 3);
    EXPECT_EQ(seq.GetFirst(), 5);
    seq.Append(20);
    EXPECT_EQ(seq.GetLast(), 20);
}

TEST(MutableListSeq, Concat) {
    int a[] = {1, 2}, b[] = {3, 4};
    MutableListSequence<int> sa(a, 2), sb(b, 2);
    sa.Concat(&sb);
    EXPECT_EQ(sa.GetLength(), 4);
}

// ImmutableListSequence
TEST(ImmutableListSeq, AppendDoesNotMutate) {
    int arr[] = {1, 2};
    ImmutableListSequence<int> seq(arr, 2);
    auto* newSeq = seq.Append(3);
    EXPECT_EQ(seq.GetLength(), 2);
    EXPECT_EQ(newSeq->GetLength(), 3);
    delete newSeq;
}

// граничные случаи
TEST(Sequence, EmptySeq) {
    MutableArraySequence<int> seq;
    EXPECT_EQ(seq.GetLength(), 0);
    EXPECT_THROW(seq.GetFirst(), IndexOutOfRange);
    EXPECT_THROW(seq.GetLast(),  IndexOutOfRange);
    EXPECT_THROW(seq.Get(0),     IndexOutOfRange);
}

TEST(Sequence, SingleElement) {
    MutableArraySequence<int> seq;
    seq.Append(42);
    EXPECT_EQ(seq.GetFirst(), 42);
    EXPECT_EQ(seq.GetLast(),  42);
    EXPECT_EQ(seq.GetLength(), 1);
}

TEST(Sequence, OutOfRange) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    EXPECT_THROW(seq.Get(-1), IndexOutOfRange);
    EXPECT_THROW(seq.Get(3),  IndexOutOfRange);
    EXPECT_THROW(seq.GetSubsequence(-1, 1), IndexOutOfRange);
    EXPECT_THROW(seq.GetSubsequence(0, 5),  IndexOutOfRange);
}

// TryGet / TryGetFirst / TryGetLast
TEST(Sequence, TrySemantic) {
    MutableArraySequence<int> seq;
    EXPECT_FALSE(seq.TryGetFirst().HasValue());
    EXPECT_FALSE(seq.TryGetLast().HasValue());
    EXPECT_FALSE(seq.TryGet(0).HasValue());
    seq.Append(7);
    EXPECT_TRUE(seq.TryGetFirst().HasValue());
    EXPECT_EQ(seq.TryGet(0).GetValue(), 7);
}

// RemoveAt / RemoveFirst / RemoveLast
TEST(MutableArraySeq, RemoveAt) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    seq.RemoveAt(1);
    EXPECT_EQ(seq.GetLength(), 2);
    EXPECT_EQ(seq.Get(1),      3);
}

TEST(MutableArraySeq, RemoveFirst) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    seq.RemoveFirst();
    EXPECT_EQ(seq.GetFirst(),  2);
    EXPECT_EQ(seq.GetLength(), 2);
}

TEST(MutableArraySeq, RemoveLast) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    seq.RemoveLast();
    EXPECT_EQ(seq.GetLast(),   2);
    EXPECT_EQ(seq.GetLength(), 2);
}

TEST(ImmutableArraySeq, RemoveAtDoesNotMutate) {
    int arr[] = {1, 2, 3};
    ImmutableArraySequence<int> seq(arr, 3);
    auto* newSeq = seq.RemoveAt(1);
    EXPECT_EQ(seq.GetLength(),    3);  // оригинал не изменился
    EXPECT_EQ(newSeq->GetLength(), 2);
    EXPECT_EQ(newSeq->Get(1),      3);
    delete newSeq;
}

TEST(MutableListSeq, RemoveAt) {
    int arr[] = {5, 10, 15};
    MutableListSequence<int> seq(arr, 3);
    seq.RemoveAt(1);
    EXPECT_EQ(seq.GetLength(), 2);
    EXPECT_EQ(seq.Get(1),      15);
}

TEST(ImmutableListSeq, RemoveAtDoesNotMutate) {
    int arr[] = {1, 2, 3};
    ImmutableListSequence<int> seq(arr, 3);
    auto* newSeq = seq.RemoveAt(1);
    EXPECT_EQ(seq.GetLength(),    3);  // оригинал не изменился
    EXPECT_EQ(newSeq->GetLength(), 2);
    delete newSeq;
}

TEST(Sequence, RemoveOutOfRange) {
    int arr[] = {1, 2, 3};
    MutableArraySequence<int> seq(arr, 3);
    EXPECT_THROW(seq.RemoveAt(-1), IndexOutOfRange);
    EXPECT_THROW(seq.RemoveAt(3),  IndexOutOfRange);
}

TEST(Sequence, RemoveFromEmpty) {
    MutableArraySequence<int> seq;
    EXPECT_THROW(seq.RemoveFirst(), IndexOutOfRange);
    EXPECT_THROW(seq.RemoveLast(),  IndexOutOfRange);
}
