/*Trail C++ program*/

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define N 512 

__global__ void add(int *a, int *b, int *c)
{
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];		//Each block performs the addition command separately on its contents 		
}

int main(void)
{	
	printf("Hello! This is my first cuda C program with Ubuntu 11.10\n");
	
	/* Do something more if you want */
	int *a, *b, *c;		//host or cpu copies 
	int *d_a, *d_b, *d_c;	//device or GPU copies
	
	int size = N*sizeof(int);
	
	//Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);
	
	//Input values: Allocate space for host copies of a,b,c and setup input values
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	for (int i=0; i<N; i++)
	{	
		srand(time(NULL));
		a[i] = rand();
		b[i] = rand();
	}		
	c = (int *)malloc(size);

	//setup input values
	//a=2;
	//b=7;

	//copy inputs to device(GPU) memory
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//launch add() kernel on N GPU
	add<<<N,1>>>(d_a, d_b, d_c);
	
	//copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
		
	//int result = c;
	printf("Result=%p \n", c);

	//cleanup 
	free(a);
	free(b);
	free(c);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	return 0;
}
