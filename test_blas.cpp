#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <stdio.h>
#include <sys/wait.h>
#include <stdlib.h>

// g++ -o testC testC.cpp  -llapack -lblas

extern "C" int dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" int dsyrk_(char *uplo, char *trans, int *n, int *k,
                      double *alpha, double *a, int *lda, double *beta, double *c, int *ldc);

int main()
{
    pid_t childpid; /* variable to store the child's pid */
    int retval;     /* child process: user-provided return code */
    int status;     /* parent process: child's exit status */
    int size = 3;

    int info = 0;
    char uplo = 'U';
    char trans = 'N';
    double alpha = 1.0;
    double beta = 0.0;
    double *x = new double[size * size];
    double *C = new double[size * size];
    for (int i = 0; i < size * size; i++)
    {
        x[i] = rand();
        C[i] = 0.0;
    }

    childpid = fork();

    if (childpid >= 0) /* fork succeeded */
    {
        if (childpid == 0) /* fork() returns 0 to the child process */
        {
            printf("CHILD: I am the child process!\n");
            printf("CHILD: Here's my PID: %d\n", getpid());
            printf("CHILD: My parent's PID is: %d\n", getppid());
            dsyrk_(&uplo, &trans, &size, &size, &alpha, x, &size, &beta, C, &size);
            printf("CHILD: done crossprod.\n");
            dpotrf_(&uplo, &size, C, &size, &info);
            printf("CHILD: done Cholesky.\n");
            printf("CHILD: Goodbye!\n");
            exit(0);
        }
        else /* fork() returns new pid to the parent process */
        {
            printf("PARENT: I am the parent process!\n");
            printf("PARENT: Here's my PID: %d\n", getpid());
            printf("PARENT: The value of my copy of childpid is %d\n", childpid);
            printf("PARENT: I will now wait for my child to exit.\n");
            wait(&status); /* wait for child to exit, and store its status */
            printf("PARENT: Child's exit code is: %d\n", WEXITSTATUS(status));
            printf("PARENT: Goodbye!\n");
            exit(0); /* parent exits */
        }
    }
    else /* fork returns -1 on failure */
    {
        perror("fork"); /* display error message */
        exit(0);
    }
}
