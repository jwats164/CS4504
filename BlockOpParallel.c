#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>

#define N 2048
#define FactorIntToDouble 1.1;

double firstMatrix [N] [N] = {0.0};
double secondMatrix [N] [N] = {0.0};
double matrixMultiResult [N] [N] = {0.0};


void matrixInit()
{
//    #pragma omp parallel for num_threads(4)
    for(int row = 0 ; row < N ; row++ ) {
        for(int col = 0 ; col < N ;col++){
            srand(row+col);
            firstMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
            secondMatrix [row] [col] = ( rand() % 10 ) * FactorIntToDouble;
        }
    }
}



//for small block
void smallMatrixMult (int upperOfRow , int bottomOfRow , int leftOfCol , int rightOfCol , int transLeft ,int transRight )
{
//implement your code here using openMP
}




//if the matrix is too big, just seperate it into smaller ones
void matrixMulti(int upperOfRow , int bottomOfRow ,
                 int leftOfCol , int rightOfCol ,
                 int transLeft ,int transRight )
{
    //If you think 512*512 is not small enough, you can adjust the number
    if ( ( bottomOfRow - upperOfRow ) < 512 ) {
        smallMatrixMult ( upperOfRow , bottomOfRow , leftOfCol , rightOfCol , transLeft , transRight );
    } else {
        #pragma omp task
        {
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp task
        {
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( upperOfRow , ( upperOfRow + bottomOfRow ) / 2 ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }


        #pragma omp task
        {
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         leftOfCol , ( leftOfCol + rightOfCol ) / 2 ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp task
        {
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         transLeft , ( transLeft + transRight ) / 2 );
            matrixMulti( ( upperOfRow + bottomOfRow ) / 2 + 1 , bottomOfRow ,
                         ( leftOfCol + rightOfCol ) / 2 + 1 , rightOfCol ,
                         ( transLeft + transRight ) / 2 + 1 , transRight );
        }

        #pragma omp taskwait
    }
}






int main()
{
    matrixInit();

    //clock_t t1 = clock();
    //matrixMulti(0, N-1, 0, N-1, 0, N-1);
    //clock_t t2 = clock();
    //printf("time: %ld", t2-t1);

    double t1 = omp_get_wtime();
    matrixMulti(0, N-1, 0, N-1, 0, N-1);
    double t2 = omp_get_wtime();
    printf("execution   time: %3f\n", ((double)t2 - t1) / CLOCKS_PER_SEC * 1000000.0);

    return 0;
}
