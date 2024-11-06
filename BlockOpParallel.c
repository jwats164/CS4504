#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 2048
#define FactorIntToDouble 1.1

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

// Parallel initialization of matrices
void matrixInit() {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            unsigned int seed = row + col;
            firstMatrix[row][col] = (rand_r(&seed) % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand_r(&seed) % 10) * FactorIntToDouble;
        }
    }
}

// Multiplies smaller matrix blocks (used by matrixMulti)
void smallMatrixMult(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight) {
    #pragma omp parallel for collapse(2)
    for (int row = upperOfRow; row <= bottomOfRow; row++) {
        for (int col = leftOfCol; col <= rightOfCol; col++) {
            double resultValue = 0.0;
            for (int transNum = transLeft; transNum <= transRight; transNum++) {
                resultValue += firstMatrix[row][transNum] * secondMatrix[transNum][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
    }
}

// Divides and conquers for large matrices
void matrixMulti(int upperOfRow, int bottomOfRow, int leftOfCol, int rightOfCol, int transLeft, int transRight) {
    if ((bottomOfRow - upperOfRow) < 512) {
        smallMatrixMult(upperOfRow, bottomOfRow, leftOfCol, rightOfCol, transLeft, transRight);
    } else {
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task
                matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2);

                #pragma omp task
                matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight);

                #pragma omp task
                matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2);

                #pragma omp task
                matrixMulti(upperOfRow, (upperOfRow + bottomOfRow) / 2, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight);

                #pragma omp task
                matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, transLeft, (transLeft + transRight) / 2);

                #pragma omp task
                matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, leftOfCol, (leftOfCol + rightOfCol) / 2, (transLeft + transRight) / 2 + 1, transRight);

                #pragma omp task
                matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, transLeft, (transLeft + transRight) / 2);

                #pragma omp task
                matrixMulti((upperOfRow + bottomOfRow) / 2 + 1, bottomOfRow, (leftOfCol + rightOfCol) / 2 + 1, rightOfCol, (transLeft + transRight) / 2 + 1, transRight);

                #pragma omp taskwait
            }
        }
    }
}

int main() {
    matrixInit();

    double t1 = omp_get_wtime();
    matrixMulti(0, N - 1, 0, N - 1, 0, N - 1);
    double t2 = omp_get_wtime();
    printf("Execution time: %f seconds\n", t2 - t1);

    return 0;
}
