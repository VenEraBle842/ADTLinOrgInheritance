#pragma once
#include <stdexcept>
#include <string>

// Исключение выхода за границы индекса
class IndexOutOfRange : public std::out_of_range {
public:
    IndexOutOfRange(int idx, int maxIdx)
        : std::out_of_range(
              "IndexOutOfRange: got " + std::to_string(idx) +
              ", valid range [0, " + std::to_string(maxIdx) + "]") {}

    explicit IndexOutOfRange(const std::string& msg)
        : std::out_of_range(msg) {}
};

//  Matrix

// Отрицательный или иным образом недопустимый размер матрицы
class InvalidMatrixSize : public std::invalid_argument {
public:
    explicit InvalidMatrixSize(int n)
        : std::invalid_argument(
              "InvalidMatrixSize: size must be >= 0, got " +
              std::to_string(n)) {}

    explicit InvalidMatrixSize(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// Для работы требуются две матрицы одинакового размера
class MatrixSizeMismatch : public std::invalid_argument {
public:
    MatrixSizeMismatch(int lhs, int rhs)
        : std::invalid_argument(
              "MatrixSizeMismatch: left " + std::to_string(lhs) +
              "x" + std::to_string(lhs) +
              " vs right " + std::to_string(rhs) +
              "x" + std::to_string(rhs)) {}

    explicit MatrixSizeMismatch(const std::string& msg)
        : std::invalid_argument(msg) {}
};

// Индекс строки выходит за допустимый диапазон [0, n-1]
class RowIndexOutOfRange : public IndexOutOfRange {
public:
    RowIndexOutOfRange(int row, int n)
        : IndexOutOfRange(
              "RowIndexOutOfRange: got row " + std::to_string(row) +
              ", valid range [0, " + std::to_string(n - 1) + "]") {}
};

// Индекс столбца выходит за допустимый диапазон [0, n-1]
class ColIndexOutOfRange : public IndexOutOfRange {
public:
    ColIndexOutOfRange(int col, int n)
        : IndexOutOfRange(
              "ColIndexOutOfRange: got col " + std::to_string(col) +
              ", valid range [0, " + std::to_string(n - 1) + "]") {}
};

//  Solver

class SolverException : public std::runtime_error {
public:
    explicit SolverException(const std::string& msg)
        : std::runtime_error(msg) {}
};

// Матрица является сингулярной или почти сингулярной (пивот < эпсилона).
class SingularMatrixException : public SolverException {
public:
    // LU: пивот опустился ниже эпсилон на шаге k
    SingularMatrixException(int step, double pivot, double eps)
        : SolverException(
              "SingularMatrixException: pivot " +
              std::to_string(pivot) +
              " < eps " + std::to_string(eps) +
              " at elimination step " + std::to_string(step)) {}

    explicit SingularMatrixException(const std::string& msg)
        : SolverException(msg) {}
};

// Матрица не соответствует рангу (норма столбца упала до нуля во время QR)
class RankDeficientException : public SolverException {
public:
    // QR: в столбце j норма ниже эпсилон
    RankDeficientException(int col, double norm, double eps)
        : SolverException(
              "RankDeficientException: column " + std::to_string(col) +
              " norm " + std::to_string(norm) +
              " < eps " + std::to_string(eps) +
              " during QR decomposition") {}

    explicit RankDeficientException(const std::string& msg)
        : SolverException(msg) {}
};

// Длина вектора b не соответствует размеру матрицы
class RHSLengthMismatch : public SolverException {
public:
    RHSLengthMismatch(int bLen, int matrixSize)
        : SolverException(
              "RHSLengthMismatch: b has " + std::to_string(bLen) +
              " element(s), matrix is " +
              std::to_string(matrixSize) + "x" +
              std::to_string(matrixSize)) {}
};

// Solver используется перед предоставлением матрицы
class SolverNotInitialized : public SolverException {
public:
    explicit SolverNotInitialized(const std::string& solverName)
        : SolverException(
              "SolverNotInitialized: " + solverName +
              " has no matrix — call SetMatrix() first") {}
};