#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 64 // Adjust based on your system’s cache size for optimal performance
#define FactorIntToDouble 1.1

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

// Function to initialize matrices with random values
void matrixInit() {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            srand(row + col); // Use row + col as a seed for reproducibility
            firstMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand() % 10) * FactorIntToDouble;
        }
    }
}

// Block-optimized matrix multiplication
void matrixMulti() {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                // Multiply individual blocks
                for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj++) {
                        double sum = 0.0;
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk++) {
                            sum += firstMatrix[ii][kk] * secondMatrix[kk][jj];
                        }
                        matrixMultiResult[ii][jj] += sum;
                    }
                }
            }
        }
    }
}

int main() {
    // Initialize matrices
    matrixInit();

    // Measure execution time for sequential block-optimized multiplication
    clock_t t1 = clock();
    matrixMulti();
    clock_t t2 = clock();

    printf("Sequential block-optimized execution time: %f seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

    return 0;
}
