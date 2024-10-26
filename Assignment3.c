#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NUMSTEPS 1000000

int main(int argc, char** argv) {
    int i, rank, numprocs;
    double x, pi, sum = 0.0, global_sum = 0.0;
    double step = 1.0 / (double)NUMSTEPS;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs); // Get number of processes

    // Calculate the range of steps for each process
    int start = rank * (NUMSTEPS / numprocs);
    int end = (rank + 1) * (NUMSTEPS / numprocs);

    // Each process calculates its portion of the sum
    for (i = start; i < end; i++) {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    // Multiply by step size to get the partial pi value
    sum = sum * step;

    // Reduce all partial sums to obtain the final result in rank 0
    MPI_Reduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the result
    if (rank == 0) {
        printf("Calculated value of Pi: %.20f\n", global_sum);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
