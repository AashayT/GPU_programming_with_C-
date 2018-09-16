#include <stdio.h>

__global__ void vector_add(long int *a, long int *b, long int *c)
{
    /* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}

/* experiment with N */
/* how large can it be? */
#define N (2*2048*2048*2)
#define THREADS_PER_BLOCK 512

int main()
{
    	long int *a, *b, *c;
	long int *d_a, *d_b, *d_c;
	long int size = N * sizeof(long int );
	
	printf("Value of N=%d \n", N);
	/* allocate space for device copies of a, b, c */
	
	printf("Memory allocation for GPU device\n");
	cudaMalloc( (void **) &d_a, size );
	cudaMalloc( (void **) &d_b, size );
	cudaMalloc( (void **) &d_c, size );

	/* allocate space for host copies of a, b, c and setup input values */
	printf("Memory allocation for CPU\n");
	a = (long int *)malloc( size );
	b = (long int *)malloc( size );
	c = (long int *)malloc( size );
	
	printf("Defining the numbers\n");
	for( long int i = 0; i < N; i++ )
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	cudaMemcpy( d_a, a, size, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, b, size, cudaMemcpyHostToDevice );

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */ 
	vector_add<<<(N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( d_a, d_b, d_c );

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	cudaMemcpy( c, d_c, size, cudaMemcpyDeviceToHost );


	printf( "c[0] = %d\n",c[0] );
	printf( "c[%d] = %d\n",N-1, c[N-1] );

	/* clean up */

	free(a);
	free(b);
	free(c);
	cudaFree( d_a );
	cudaFree( d_b );
	cudaFree( d_c );
	
	return 0;
} /* end main */
