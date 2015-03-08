#include <stdio.h>
#include <stdlib.h>

#define N 320000

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__global__ void add(int *a, int *b, int *c) {
	int tid = blockIdx.x; // handle the data at this index
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

void vector_add(int *a, int *b, int *c){
	for(int i=0; i <N; i++){
		c[i] = a[i]+b[i];
	}
}


void cpuImplementation(){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	//allocate and initialize host cpu memory
	int a[N];
	int b[N];
	int c[N];

	//fill the arrays 'h_a' and 'h_b' on the CPU
	for(int i=0; i< N; i++){
		a[i]=-i;
		b[i]=i*i;
	}

	cudaEventRecord(start);
	vector_add(a,b,c);
	cudaEventRecord(stop);
	// display the results
//	for (int i = 0; i < N; i++) {
//		printf("%d + %d = %d\n", a[i], b[i], c[i]);
//	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Elapsed Time in CPU %fms\n", milliseconds);
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// allocate the memory on the GPU
	CUDA_CHECK_RETURN(cudaMalloc( (void**)&dev_a, N * sizeof(int) ));
	CUDA_CHECK_RETURN(cudaMalloc( (void**)&dev_b, N * sizeof(int) ));
	CUDA_CHECK_RETURN(cudaMalloc( (void**)&dev_c, N * sizeof(int) ));

	// fill the arrays 'a' and 'b' on the CPU
	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	// copy the arrays 'a' and 'b' to the GPU
	CUDA_CHECK_RETURN(cudaMemcpy( dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice ));
	CUDA_CHECK_RETURN(cudaMemcpy( dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice ));
	cudaEventRecord(start);
	add<<<N, 1>>>(dev_a, dev_b, dev_c);
	cudaEventRecord(stop);
	// copy the array 'c' back from the GPU to the CPU
	CUDA_CHECK_RETURN(cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost ));

	// display the results
//	for (int i = 0; i < N; i++) {
//		printf("%d + %d = %d\n", a[i], b[i], c[i]);
//	}

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Elapsed Time in GPU %fms\n", milliseconds);

	// free the memory allocated on the GPU
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cpuImplementation();

	return 0;
}
