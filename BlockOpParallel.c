#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 2048
#define BLOCK_SIZE 64 // Adjust block size based on your system's cache size for optimal performance
#define FactorIntToDouble 1.1

double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};
double matrixMultiResult[N][N] = {0.0};

// Function to initialize matrices with random values
void matrixInit() {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            unsigned int seed = row + col; // create a unique seed for each element
            firstMatrix[row][col] = (rand_r(&seed) % 10) * FactorIntToDouble;
            secondMatrix[row][col] = (rand_r(&seed) % 10) * FactorIntToDouble;
        }
    }
}

// Block optimized matrix multiplication using OpenMP
void matrixMulti() {
    #pragma omp parallel for collapse(2)
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
                        #pragma omp atomic
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

    // Measure execution time for block optimized parallel multiplication
    double t1 = omp_get_wtime();
    matrixMulti();
    double t2 = omp_get_wtime();
    printf("Block-optimized parallel execution time: %f seconds\n", t2 - t1);

    return 0;
}
