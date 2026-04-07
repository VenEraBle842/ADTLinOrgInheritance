#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <random>
#include <limits>
#include <algorithm>

#include "src/Matrix.h"
#include "src/Solver.h"
#include "src/Output.h"

using Clock = std::chrono::high_resolution_clock;
using NS    = std::chrono::nanoseconds;


static void printTiming(NS ns) {
    std::cout << "  Time: " << ns.count() << " ns\n";
}


template <class T> static std::string typeName()          { return "unknown"; }
template <>        std::string typeName<double>()         { return "double";  }
template <>        std::string typeName<Complex>()        { return "complex"; }
template <>        std::string typeName<int>()            { return "int";     }

//  Ввод скаляра (из istringstream или из std::cin)

template <class T>
static T readScalarISS(std::istringstream& iss) {
    T v{}; iss >> v; return v;
}
template <>
Complex readScalarISS<Complex>(std::istringstream& iss) {
    double re, im; iss >> re >> im; return {re, im};
}

template <class T>
static T readScalarCin() {
    T v{}; std::cin >> v; return v;
}
template <>
Complex readScalarCin<Complex>() {
    double re, im; std::cin >> re >> im; return {re, im};
}

static void cinFlush() {
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

//  Генератор случайных чисел (один на всю программу)

static std::mt19937 rng = []() {
    std::random_device rd;
    return std::mt19937(rd());
}();

template <class T>
static T randomScalar(double lo, double hi);

template <>
double randomScalar<double>(double lo, double hi) {
    return std::uniform_real_distribution<double>(lo, hi)(rng);
}
template <>
int randomScalar<int>(double lo, double hi) {
    return std::uniform_int_distribution<int>(
        static_cast<int>(lo), static_cast<int>(hi))(rng);
}
template <>
Complex randomScalar<Complex>(double lo, double hi) {
    std::uniform_real_distribution<double> d(lo, hi);
    return {d(rng), d(rng)};
}

//  Фабрики матриц и векторов

template <class T>
static MutableSquareMatrix<T>* makeRandomMatrix(int n, double lo, double hi) {
    auto* m = new MutableSquareMatrix<T>(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m->Set(i, j, randomScalar<T>(lo, hi));
    return m;
}

// Диагонально-доминирующая матрица — гарантированно невырожденная
template <class T>
static MutableSquareMatrix<T>* makeRandomDiagDom(int n, double lo, double hi,
                                                  double delta = 1.0) {
    auto* m = makeRandomMatrix<T>(n, lo, hi);
    for (int i = 0; i < n; ++i) {
        double rowSum = 0.0;
        for (int j = 0; j < n; ++j)
            if (j != i) rowSum += absOf(m->Get(i, j));
        m->Set(i, i, fromDouble<T>(rowSum + delta));
    }
    return m;
}

static MutableSquareMatrix<double>* makeHilbert(int n) {
    auto* m = new MutableSquareMatrix<double>(n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m->Set(i, j, 1.0 / (i + j + 1));
    return m;
}

template <class T>
static MutableArraySequence<T>* makeRandomVec(int n, double lo, double hi) {
    auto* v = new MutableArraySequence<T>();
    for (int i = 0; i < n; ++i) v->Append(randomScalar<T>(lo, hi));
    return v;
}

// интерактивный ввод

template <class T>
static MutableArraySequence<T>* inputVec(int n) {
    auto* v = new MutableArraySequence<T>();
    std::cout << "Enter " << n << " value(s)";
    if (std::is_same_v<T, Complex>) std::cout << " (each as 're im')";
    std::cout << ":\n";
    for (int i = 0; i < n; ++i) v->Append(readScalarCin<T>());
    cinFlush();
    return v;
}

template <class T>
static MutableSquareMatrix<T>* inputMatrix(int n) {
    auto* m = new MutableSquareMatrix<T>(n);
    std::cout << "Enter " << n * n << " element(s) row by row";
    if (std::is_same_v<T, Complex>) std::cout << " (each as 're im')";
    std::cout << ":\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            m->Set(i, j, readScalarCin<T>());
    cinFlush();
    return m;
}

// единый помощник выбора заполнения

template <class T>
static MutableSquareMatrix<T>* acquireMatrix(int n) {
    std::cout << "Fill: 0=manual  1=random  2=random diag-dominant\n> ";
    int method; std::cin >> method; cinFlush();
    if (method == 0) {
        return inputMatrix<T>(n);
    }
    double lo, hi;
    if (method == 1) {
        std::cout << "Min Max: "; std::cin >> lo >> hi; cinFlush();
        return makeRandomMatrix<T>(n, lo, hi);
    }
    std::cout << "Off-diagonal Min Max (delta=1.0): ";
    std::cin >> lo >> hi; cinFlush();
    return makeRandomDiagDom<T>(n, lo, hi);
}

//  Вывод

template <class T>
static void printMatrix(const SquareMatrix<T>& m, int maxRows = -1) {
    int n     = m.GetSize();
    int limit = (maxRows > 0 && maxRows < n) ? maxRows : n;
    std::cout << std::fixed << std::setprecision(6);
    for (int i = 0; i < limit; ++i) {
        std::cout << "  [ ";
        for (int j = 0; j < n; ++j) {
            if (j > 0) std::cout << "  ";
            std::cout << std::setw(14) << m.Get(i, j);
        }
        std::cout << " ]\n";
    }
    if (limit < n)
        std::cout << "  ... (" << (n - limit) << " rows hidden)\n";
    std::cout << std::defaultfloat;
}

template <class T>
static void printVec(const Sequence<T>& v) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  [";
    for (int i = 0; i < v.GetLength(); ++i) {
        if (i > 0) std::cout << "  ";
        std::cout << " " << v.Get(i);
    }
    std::cout << " ]\n" << std::defaultfloat;
}

//  Невязка  ||Ax - b||

template <class T>
static double computeResidual(const SquareMatrix<T>& A,
                               const Sequence<T>&     x,
                               const Sequence<T>&     b) {
    int n = A.GetSize();
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        T row = scalarZero<T>();
        for (int j = 0; j < n; ++j)
            row = row + A.Get(i, j) * x.Get(j);
        sum += normSqOf(row - b.Get(i));
    }
    return std::sqrt(sum);
}

//  Справка

static void printMatrixHelp() {
    std::cout << R"(
=== Matrix Shell Commands ===
  new               - create new matrix
  show              - print current matrix
  get <r> <c>       - get element at (r,c)
  set <r> <c> <v>   - set element  [Complex: <r> <c> <re> <im>]
  add               - add another matrix
  scale <v>         - multiply by scalar  [Complex: <re> <im>]
  norm              - Frobenius norm
  swaprows <i> <j>  - swap rows
  mulrow  <i> <v>   - multiply row by scalar
  addrow  <i> <j> <v> - row[i] += v*row[j]
  swapcols <i> <j>  - swap columns
  mulcol  <i> <v>   - multiply column by scalar
  addcol  <i> <j> <v> - col[i] += v*col[j]
  iter              - iterate all elements (row-major)
  lu                - show LU decomposition
  qr                - show QR decomposition
  help / back
)";
}

static void printSysHelp() {
    std::cout << R"(
=== SLAE Shell Commands ===
  new               - create new system (A and b)
  show              - print A and b
  showx             - print last solution + residual
  seta              - replace A (solver caches invalidated)
  setb              - replace b (solver caches remain valid)
  solvelu           - solve with LUSolver
  solveqr           - solve with QRSolver
  compare           - solve with both, print timing and residuals
  multisolve <k>    - solve k random RHS with both solvers (cache demo)
  residual          - ||Ax - b|| for last solution
  help / back
)";
}

//  matrixShell<T>

template <class T>
static void matrixShell() {
    std::cout << "Mode: 0=Mutable  1=Immutable\n> ";
    int mut; std::cin >> mut; cinFlush();
    bool immutable = (mut == 1);

    SquareMatrix<T>* cur = nullptr;
    std::string line;

    while (true) {
        std::cout << "mat[" << typeName<T>() << (immutable ? "/imm" : "/mut") << "]> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd; iss >> cmd;

        try {
            if (cmd == "back") {
                break;

            } if (cmd == "help") {
                printMatrixHelp();

            } else if (cmd == "new") {
                int n;
                std::cout << "Size n: "; std::cin >> n; cinFlush();
                auto t0 = Clock::now();
                MutableSquareMatrix<T>* tmp = acquireMatrix<T>(n);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                delete cur;
                if (immutable) {
                    cur = new ImmutableSquareMatrix<T>(*tmp);
                    delete tmp;
                } else {
                    cur = tmp;
                }
                std::cout << "Created " << (immutable ? "Immutable" : "Mutable")
                          << " " << n << "x" << n << ":\n";
                printMatrix(*cur);
                printTiming(elapsed);

            } else if (cmd == "show") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                auto t0 = Clock::now();
                printMatrix(*cur);
                printTiming(std::chrono::duration_cast<NS>(Clock::now() - t0));

            } else if (cmd == "get") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int r, c; iss >> r >> c;
                auto t0 = Clock::now();
                T val = cur->Get(r, c);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                std::cout << "  (" << r << "," << c << ") = " << val << "\n";
                printTiming(elapsed);

            } else if (cmd == "set") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int r, c; iss >> r >> c;
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                cur->Set(r, c, v);
                printTiming(std::chrono::duration_cast<NS>(Clock::now() - t0));

            } else if (cmd == "add") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int n = cur->GetSize();
                std::cout << "Second matrix (" << n << "x" << n << "):\n";
                MutableSquareMatrix<T>* other = acquireMatrix<T>(n);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->Add(other);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                delete other;
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur);
                printTiming(elapsed);

            } else if (cmd == "scale") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->MulScalar(v);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur);
                printTiming(elapsed);

            } else if (cmd == "norm") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                auto t0 = Clock::now();
                double nrm = cur->Norm();
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                std::cout << "  Frobenius norm = " << nrm << "\n";
                printTiming(elapsed);

            } else if (cmd == "swaprows") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i, j; iss >> i >> j;
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->SwapRows(i, j);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "mulrow") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i; iss >> i;
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->MulRow(i, v);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "addrow") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i, j; iss >> i >> j;
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->AddRow(i, j, v);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "swapcols") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i, j; iss >> i >> j;
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->SwapCols(i, j);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "mulcol") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i; iss >> i;
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->MulCol(i, v);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "addcol") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                int i, j; iss >> i >> j;
                T v = readScalarISS<T>(iss);
                auto t0 = Clock::now();
                SquareMatrix<T>* res = cur->AddCol(i, j, v);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                if (res != cur) { delete cur; }
                cur = res;
                printMatrix(*cur); printTiming(elapsed);

            } else if (cmd == "iter") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                auto t0 = Clock::now();
                int n = cur->GetSize(), idx = 0;
                for (T val : *cur) {
                    int r = idx / n, c = idx % n;
                    std::cout << "  [" << r << "," << c << "] = " << val << "\n";
                    ++idx;
                }
                printTiming(std::chrono::duration_cast<NS>(Clock::now() - t0));

            } else if (cmd == "lu") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                if constexpr (std::is_same_v<T, int>) {
                    std::cout << "LU requires double or complex type.\n";
                } else {
                    auto t0 = Clock::now();
                    LUSolver<T> solver(*cur);
                    const SquareMatrix<T>&   L = solver.GetL();
                    const SquareMatrix<T>&   U = solver.GetU();
                    const DynamicArray<int>& P = solver.GetP();
                    auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                    std::cout << "  L:\n"; printMatrix(L);
                    std::cout << "  U:\n"; printMatrix(U);
                    std::cout << "  P: [";
                    for (int i = 0; i < solver.GetSize(); ++i)
                        std::cout << " " << P.Get(i);
                    std::cout << " ]\n";
                    printTiming(elapsed);
                }

            } else if (cmd == "qr") {
                if (!cur) { std::cout << "No matrix.\n"; continue; }
                if constexpr (std::is_same_v<T, int>) {
                    std::cout << "QR requires double or complex type.\n";
                } else {
                    auto t0 = Clock::now();
                    QRSolver<T> solver(*cur);
                    const SquareMatrix<T>& Q = solver.GetQ();
                    const SquareMatrix<T>& R = solver.GetR();
                    auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                    std::cout << "  Q:\n"; printMatrix(Q);
                    std::cout << "  R:\n"; printMatrix(R);
                    printTiming(elapsed);
                }

            } else {
                std::cout << "Unknown command. Type 'help'.\n";
            }

        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }
    delete cur;
}

//  slaShell<T>

template <class T>
static void slaShell() {
    MutableSquareMatrix<T>*  A      = nullptr;
    MutableArraySequence<T>* b      = nullptr;
    MutableArraySequence<T>* lastX  = nullptr;
    LUSolver<T>*             luSolv = nullptr;
    QRSolver<T>*             qrSolv = nullptr;

    auto resetSolvers = [&]() {
        delete luSolv; luSolv = nullptr;
        delete qrSolv; qrSolv = nullptr;
    };

    auto ensureSolvers = [&]() {
        if (!luSolv) luSolv = new LUSolver<T>(*A);
        if (!qrSolv) qrSolv = new QRSolver<T>(*A);
    };

    std::string line;
    while (true) {
        std::cout << "sys[" << typeName<T>() << "]> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd; iss >> cmd;

        try {
            if (cmd == "back") {
                break;

            } if (cmd == "help") {
                printSysHelp();

            } else if (cmd == "new") {
                int n;
                std::cout << "Size n: "; std::cin >> n; cinFlush();

                std::cout << "--- Matrix A ---\n";
                MutableSquareMatrix<T>* newA = acquireMatrix<T>(n);

                std::cout << "--- Vector b ---\nFill: 0=manual  1=random\n> ";
                int bm; std::cin >> bm; cinFlush();
                MutableArraySequence<T>* newB;
                if (bm == 0) {
                    newB = inputVec<T>(n);
                } else {
                    double lo, hi;
                    std::cout << "Min Max: "; std::cin >> lo >> hi; cinFlush();
                    newB = makeRandomVec<T>(n, lo, hi);
                }

                delete A; delete b; delete lastX;
                resetSolvers();
                A = newA; b = newB; lastX = nullptr;
                std::cout << "A:\n"; printMatrix(*A);
                std::cout << "b:\n"; printVec(*b);

            } else if (cmd == "show") {
                if (!A) { std::cout << "No system. Use 'new'.\n"; continue; }
                std::cout << "A:\n"; printMatrix(*A);
                std::cout << "b:\n"; printVec(*b);

            } else if (cmd == "showx") {
                if (!lastX) { std::cout << "No solution yet.\n"; continue; }
                std::cout << "x:\n"; printVec(*lastX);
                std::cout << "  ||Ax - b|| = "
                          << computeResidual(*A, *lastX, *b) << "\n";

            } else if (cmd == "seta") {
                if (!A) { std::cout << "No system. Use 'new' first.\n"; continue; }
                int n = A->GetSize();
                std::cout << "--- New Matrix A (" << n << "x" << n << ") ---\n";
                MutableSquareMatrix<T>* newA = acquireMatrix<T>(n);
                delete A; A = newA;
                // SetMatrix сбрасывает кэш разложения, сами объекты солверов живут
                if (luSolv) luSolv->SetMatrix(*A);
                if (qrSolv) qrSolv->SetMatrix(*A);
                delete lastX; lastX = nullptr;
                std::cout << "Matrix A updated. Solver caches invalidated.\nA:\n";
                printMatrix(*A);

            } else if (cmd == "setb") {
                if (!A) { std::cout << "No system. Use 'new' first.\n"; continue; }
                int n = A->GetSize();
                std::cout << "--- New vector b ---\nFill: 0=manual  1=random\n> ";
                int bm; std::cin >> bm; cinFlush();
                MutableArraySequence<T>* newB;
                if (bm == 0) {
                    newB = inputVec<T>(n);
                } else {
                    double lo, hi;
                    std::cout << "Min Max: "; std::cin >> lo >> hi; cinFlush();
                    newB = makeRandomVec<T>(n, lo, hi);
                }
                delete b; b = newB;
                std::cout << "Vector b updated. Solver caches remain valid.\nb:\n";
                printVec(*b);

            } else if (cmd == "solvelu") {
                if (!A) { std::cout << "No system.\n"; continue; }
                ensureSolvers();
                auto t0 = Clock::now();
                auto* x = luSolv->Solve(*b);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                delete lastX; lastX = x;
                std::cout << "x:\n"; printVec(*lastX);
                std::cout << "  ||Ax - b|| = "
                          << computeResidual(*A, *lastX, *b) << "\n";
                printTiming(elapsed);

            } else if (cmd == "solveqr") {
                if (!A) { std::cout << "No system.\n"; continue; }
                ensureSolvers();
                auto t0 = Clock::now();
                auto* x = qrSolv->Solve(*b);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                delete lastX; lastX = x;
                std::cout << "x:\n"; printVec(*lastX);
                std::cout << "  ||Ax - b|| = "
                          << computeResidual(*A, *lastX, *b) << "\n";
                printTiming(elapsed);

            } else if (cmd == "compare") {
                if (!A) { std::cout << "No system.\n"; continue; }
                ensureSolvers();
                int n = A->GetSize();

                auto t0 = Clock::now();
                auto* xlu = luSolv->Solve(*b);
                NS tlu = std::chrono::duration_cast<NS>(Clock::now() - t0);

                t0 = Clock::now();
                auto* xqr = qrSolv->Solve(*b);
                NS tqr = std::chrono::duration_cast<NS>(Clock::now() - t0);

                double resLU = computeResidual(*A, *xlu, *b);
                double resQR = computeResidual(*A, *xqr, *b);

                double diff = 0.0;
                for (int i = 0; i < n; ++i)
                    diff += normSqOf(xlu->Get(i) - xqr->Get(i));
                diff = std::sqrt(diff);

                std::cout << "\n=== SLAE n=" << n
                          << ", " << typeName<T>() << " ===\n";
                std::cout << "  -- LU ---------------------------------------\n";
                std::cout << "  x:        "; printVec(*xlu);
                std::cout << "  ||Ax-b||:  " << std::scientific << resLU << "\n";
                std::cout << "  Time:      " << tlu.count() << " ns\n";
                std::cout << "  -- QR ---------------------------------------\n";
                std::cout << "  x:        "; printVec(*xqr);
                std::cout << "  ||Ax-b||:  " << resQR << "\n";
                std::cout << "  Time:      " << tqr.count() << " ns\n";
                std::cout << "  -- Total ------------------------------------\n";
                std::cout << "  ||x_LU - x_QR||: " << diff << "\n"
                          << std::defaultfloat << "\n";

                delete lastX; lastX = xlu;
                delete xqr;

            } else if (cmd == "multisolve") {
                if (!A) { std::cout << "No system.\n"; continue; }
                int k = 5; iss >> k;
                if (k <= 0) k = 5;
                ensureSolvers();
                int n = A->GetSize();

                DynamicArray<MutableArraySequence<T>*> bVecs(k);
                for (int i = 0; i < k; ++i)
                    bVecs[i] = makeRandomVec<T>(n, -10.0, 10.0);

                std::cout << "\n=== multisolve  k=" << k
                          << "  n=" << n << "  " << typeName<T>() << " ===\n";
                std::cout << std::setw(4)  << "i"
                          << std::setw(16) << "LU time (ns)"
                          << std::setw(16) << "QR time (ns)"
                          << std::setw(16) << "||Ax-b|| LU"
                          << std::setw(16) << "||Ax-b|| QR"
                          << "\n"
                          << std::string(68, '-') << "\n";

                long long sumLU = 0, sumQR = 0;
                for (int i = 0; i < k; ++i) {
                    auto t0 = Clock::now();
                    auto* xlu = luSolv->Solve(*bVecs[i]);
                    long long tlu = std::chrono::duration_cast<NS>(
                                        Clock::now() - t0).count();
                    t0 = Clock::now();
                    auto* xqr = qrSolv->Solve(*bVecs[i]);
                    long long tqr = std::chrono::duration_cast<NS>(
                                        Clock::now() - t0).count();

                    double rlu = computeResidual(*A, *xlu, *bVecs[i]);
                    double rqr = computeResidual(*A, *xqr, *bVecs[i]);

                    std::cout << std::setw(4)  << (i + 1)
                              << std::setw(16) << tlu
                              << std::setw(16) << tqr
                              << std::setw(16) << std::scientific << rlu
                              << std::setw(16) << rqr
                              << std::defaultfloat << "\n";

                    sumLU += tlu; sumQR += tqr;
                    delete xlu; delete xqr;
                }

                std::cout << std::string(68, '-') << "\n";
                std::cout << "  Total LU: " << sumLU << " ns  |  "
                          << "Total QR: " << sumQR << " ns\n";
                if (k > 1)
                    std::cout << "  (Call #1 includes decomposition; "
                                 "#2..#" << k << " reuse cached factors)\n";
                std::cout << "\n";

                for (auto* v : bVecs) delete v;

            } else if (cmd == "residual") {
                if (!lastX) { std::cout << "No solution yet.\n"; continue; }
                auto t0 = Clock::now();
                double r = computeResidual(*A, *lastX, *b);
                auto elapsed = std::chrono::duration_cast<NS>(Clock::now() - t0);
                std::cout << "  ||Ax - b|| = " << r << "\n";
                printTiming(elapsed);

            } else {
                std::cout << "Unknown command. Type 'help'.\n";
            }

        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }
    }

    delete A; delete b; delete lastX;
    delete luSolv; delete qrSolv;
}

//  Структура результатов одного запуска demoHilbert

struct HilbertResult {
    int       n;
    long long tDecompLU,  tDecompQR;
    long long tSolveLU[3], tSolveQR[3];
    double    resLU[3],    resQR[3];
};

//  demoHilbert

static HilbertResult demoHilbert(int n) {
    HilbertResult hr{}; hr.n = n;
    static constexpr int K = 3;

    const std::string SEP(60, '=');
    std::cout << "\n+" << std::string(58, '=') << "+\n"
          << "|  Demo: Hilbert matrix H_"
          << std::setw(2) << n
          << std::string(18, ' ') << "|\n"
          << "+" << std::string(58, '=') << "+\n\n";

    // 1. Генерация
    auto t0 = Clock::now();
    MutableSquareMatrix<double>* H = makeHilbert(n);
    struct Guard { MutableSquareMatrix<double>*& p; ~Guard() { delete p; } } hGuard{H};
    long long tGen = std::chrono::duration_cast<NS>(Clock::now() - t0).count();

    std::cout << "1. H_" << n << "  (H[i][j] = 1/(i+j+1)):\n";
    printMatrix(*H, (n <= 7) ? n : 4);
    std::cout << "   Generation time: " << tGen << " ns\n\n";

    // 2. PLU-разложение
    std::cout << "2. PLU decomposition (PA = LU, Doolittle algorithm + partial pivoting):\n";
    // Используем минимально возможный eps, чтобы отвергать только
    // точно нулевые пивоты. H_11 невырожденная, но близка к этому;
    // позволяем солверам продолжить работу и наглядно демонстрируем
    // рост невязки из-за плохой обусловленности матрицы.
    static constexpr double DEMO_EPS = std::numeric_limits<double>::min();
    auto* luSolv = new LUSolver<double>(*H, DEMO_EPS);

    t0 = Clock::now();
    const SquareMatrix<double>& L = luSolv->GetL();
    const SquareMatrix<double>& U = luSolv->GetU();
    hr.tDecompLU = std::chrono::duration_cast<NS>(Clock::now() - t0).count();

    std::cout << "   L:\n"; printMatrix(L, (n <= 7) ? n : 4);
    std::cout << "   U:\n"; printMatrix(U, (n <= 7) ? n : 4);
    std::cout << "   Decomposition time: " << hr.tDecompLU << " ns\n\n";
    if (n >= 10)
        std::cout << "   WARNING: H_" << n
                  << " cond ~1.6e13 — results are affected by rounding error.\n";


    // 3. QR-разложение
    std::cout << "3. QR decomposition (modified Gram-Schmidt):\n";
    auto* qrSolv = new QRSolver<double>(*H, DEMO_EPS);

    t0 = Clock::now();
    const SquareMatrix<double>& Q = qrSolv->GetQ();
    const SquareMatrix<double>& R = qrSolv->GetR();
    hr.tDecompQR = std::chrono::duration_cast<NS>(Clock::now() - t0).count();

    std::cout << "   Q:\n"; printMatrix(Q, (n <= 7) ? n : 4);
    std::cout << "   R:\n"; printMatrix(R, (n <= 7) ? n : 4);
    std::cout << "   Decomposition time: " << hr.tDecompQR << " ns\n\n";

    // 4. Генерация правых частей
    std::cout << "4. Right-hand sides b1, b2, b3 (random in [-1, 1]):\n";
    MutableArraySequence<double>* bVecs[K];
    for (int i = 0; i < K; ++i) {
        bVecs[i] = makeRandomVec<double>(n, -1.0, 1.0);
        std::cout << "   b" << (i + 1) << ": "; printVec(*bVecs[i]);
    }
    std::cout << "\n";

    // 5. Решение LUSolver
    std::cout << "5. LUSolver  (cached factors from step 2 — substitution only, O(n^2)):\n";
    std::cout << "   " << std::setw(4)  << "b"
              << std::setw(18) << "Time (ns)"
              << std::setw(20) << "||Hx - b||"
              << "\n   " << SEP << "\n";
    for (int i = 0; i < K; ++i) {
        t0 = Clock::now();
        auto* x = luSolv->Solve(*bVecs[i]);
        hr.tSolveLU[i] = std::chrono::duration_cast<NS>(Clock::now() - t0).count();
        hr.resLU[i]    = computeResidual(*H, *x, *bVecs[i]);
        std::cout << "   " << std::setw(4)  << (i + 1)
                  << std::setw(18) << hr.tSolveLU[i]
                  << std::setw(20) << std::scientific << hr.resLU[i]
                  << std::defaultfloat << "\n";
        delete x;
    }
    std::cout << "\n";

    // 6. Решение QRSolver
    std::cout << "6. QRSolver  (cached factors from step 3 — Q^H * b and back-sub, O(n^2)):\n";
    std::cout << "   " << std::setw(4)  << "b"
              << std::setw(18) << "Time (ns)"
              << std::setw(20) << "||Hx - b||"
              << "\n   " << SEP << "\n";
    for (int i = 0; i < K; ++i) {
        t0 = Clock::now();
        auto* x = qrSolv->Solve(*bVecs[i]);
        hr.tSolveQR[i] = std::chrono::duration_cast<NS>(Clock::now() - t0).count();
        hr.resQR[i]    = computeResidual(*H, *x, *bVecs[i]);
        std::cout << "   " << std::setw(4)  << (i + 1)
                  << std::setw(18) << hr.tSolveQR[i]
                  << std::setw(20) << std::scientific << hr.resQR[i]
                  << std::defaultfloat << "\n";
        delete x;
    }
    std::cout << "\n";

    // 7. Сводная таблица
    std::cout << "7. Summary H_" << n << ":\n";
    auto prRow = [&](const std::string& name, long long vl, long long vq) {
        std::cout << "   " << std::left  << std::setw(22) << name
                  << std::right << std::setw(14) << vl << " ns"
                  << std::setw(14) << vq << " ns\n";
    };
    auto prRowD = [&](const std::string& name, double vl, double vq) {
        std::cout << "   " << std::left << std::setw(22) << name
                  << std::right << std::setw(16) << std::scientific << vl
                  << std::setw(16) << vq
                  << std::defaultfloat << "\n";
    };

    std::cout << "   " << std::left  << std::setw(22) << "Parameter"
              << std::right << std::setw(16) << "LU"
              << std::setw(16) << "QR"
              << "\n   " << SEP << "\n";
    prRow("Decomposition",   hr.tDecompLU, hr.tDecompQR);
    for (int i = 0; i < K; ++i)
        prRow("Solve #" + std::to_string(i + 1), hr.tSolveLU[i], hr.tSolveQR[i]);
    prRowD("Max ||Hx-b||",
           *std::max_element(hr.resLU, hr.resLU + K),
           *std::max_element(hr.resQR, hr.resQR + K));
    prRowD("Min  ||Hx-b||",
           *std::min_element(hr.resLU, hr.resLU + K),
           *std::min_element(hr.resQR, hr.resQR + K));
    std::cout << "\n";

    for (auto* v : bVecs) delete v;
    delete luSolv; delete qrSolv;
    return hr;
}

int main() {
    std::cout << "=== Matrix & Solver Shell ===\n"
              << "Commands: mat | sys | demo | quit\n\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string cmd; iss >> cmd;

        if (cmd == "quit" || cmd == "exit") {
            break;

        } if (cmd == "mat") {
            std::cout << "Element type: 0=double  1=complex  2=int\n> ";
            int t; std::cin >> t; cinFlush();
            if      (t == 0) matrixShell<double>();
            else if (t == 1) matrixShell<Complex>();
            else             matrixShell<int>();

        } else if (cmd == "sys") {
            std::cout << "Element type: 0=double  1=complex\n> ";
            int t; std::cin >> t; cinFlush();
            if (t == 0) slaShell<double>();
            else        slaShell<Complex>();

        } else if (cmd == "demo") {
            HilbertResult r7  = demoHilbert(7);
            HilbertResult r11 = demoHilbert(11);

            // Итоговая таблица сравнения
            std::cout
                << "\n+" << std::string(66, '=') << "+\n"
                << std::string(31, ' ') << "|\n"
                << "+" << std::string(66, '=') << "+\n\n";

            std::cout << std::left  << std::setw(22) << "Parameter"
                      << std::right << std::setw(14) << "LU n=7"
                      << std::setw(14) << "QR n=7"
                      << std::setw(14) << "LU n=11"
                      << std::setw(14) << "QR n=11"
                      << "\n" << std::string(78, '-') << "\n";

            auto row4 = [&](const std::string& label,
                            long long a, long long b,
                            long long c, long long d) {
                std::cout << std::left  << std::setw(22) << label
                          << std::right
                          << std::setw(11) << a << " ns"
                          << std::setw(11) << b << " ns"
                          << std::setw(11) << c << " ns"
                          << std::setw(11) << d << " ns\n";
            };
            auto row4d = [&](const std::string& label,
                             double a, double b, double c, double d) {
                std::cout << std::left  << std::setw(22) << label
                          << std::right << std::scientific
                          << std::setw(14) << a
                          << std::setw(14) << b
                          << std::setw(14) << c
                          << std::setw(14) << d
                          << std::defaultfloat << "\n";
            };

            row4("Decomposition",
                 r7.tDecompLU,  r7.tDecompQR,
                 r11.tDecompLU, r11.tDecompQR);
            for (int i = 0; i < 3; ++i)
                row4("Solve #" + std::to_string(i + 1),
                     r7.tSolveLU[i],  r7.tSolveQR[i],
                     r11.tSolveLU[i], r11.tSolveQR[i]);

            row4d("Max ||Hx-b||",
                  *std::max_element(r7.resLU,  r7.resLU  + 3),
                  *std::max_element(r7.resQR,  r7.resQR  + 3),
                  *std::max_element(r11.resLU, r11.resLU + 3),
                  *std::max_element(r11.resQR, r11.resQR + 3));

            // Ускорение кэша: время разложения / среднее время solve #2..#3
            auto cacheSpeedup = [](long long tDecomp, long long t1, long long t2) {
                double avgSolve = static_cast<double>(t1 + t2) / 2.0;
                if (avgSolve < 1.0) return 0.0;
                return static_cast<double>(tDecomp) / avgSolve;
            };

            auto row4f = [&](const std::string& label,
                             double a, double b, double c, double d) {
                std::cout << std::left  << std::setw(22) << label
                          << std::right << std::fixed << std::setprecision(1)
                          << std::setw(14) << a
                          << std::setw(14) << b
                          << std::setw(14) << c
                          << std::setw(14) << d
                          << std::defaultfloat << "\n";
            };

            row4f("Cache speedup (x)",
                  cacheSpeedup(r7.tDecompLU,  r7.tSolveLU[1],  r7.tSolveLU[2]),
                  cacheSpeedup(r7.tDecompQR,  r7.tSolveQR[1],  r7.tSolveQR[2]),
                  cacheSpeedup(r11.tDecompLU, r11.tSolveLU[1], r11.tSolveLU[2]),
                  cacheSpeedup(r11.tDecompQR, r11.tSolveQR[1], r11.tSolveQR[2]));

            std::cout << "\n"
                      << "  Note: H_7 cond ~4.7e8, H_11 cond ~1.6e13.\n"
                      << "  Larger residuals for H_11 reflect accumulated rounding error\n"
                      << "  due to high condition number (~1.6e13).\n\n";

        } else {
            std::cout << "Unknown command. Use: mat | sys | demo | quit\n";
        }
    }
    return 0;
}
