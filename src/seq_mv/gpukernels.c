#include "seq_mv.h"

//#include <stdio.h>
//#include <cuda_runtime.h>
//#include <cublas_v2.h>

#if defined(HYPRE_USING_GPU)
static HYPRE_Int FirstCall=1; 
static cublasHandle_t myHandle;
#define BlockSize 64
#define MaxSpace 100
#define maxk 80
#define Tv5 1024
#define NB 1024
#define MAX_SIZE 512

extern "C"{
__global__
void VecScaleKernelText(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex *__restrict__ l1_norm, HYPRE_Int num_rows){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		u[i]+=__ldg(v+i)/__ldg(l1_norm+i);
	}
}
}

extern "C"{
__global__
void VecScaleKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex * __restrict__ l1_norm, HYPRE_Int num_rows){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		u[i]+=v[i]/l1_norm[i];
	}
}
}

extern "C"{
void VecScale(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, HYPRE_Int num_rows,cudaStream_t s){


#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
	PUSH_RANGE_PAYLOAD("VECSCALE",1,num_rows);
	const HYPRE_Int tpb=64;
	HYPRE_Int num_blocks=num_rows/tpb+1;
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	MemPrefetchSized(l1_norm,num_rows*sizeof(HYPRE_Complex),HYPRE_DEVICE,s);
	VecScaleKernel<<<num_blocks,tpb,0,s>>>(u,v,l1_norm,num_rows);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	//  hypre_CheckErrorDevice(cudaStreamSynchronize(s));
	POP_RANGE;
#else

	const HYPRE_Int tpb=64;
	HYPRE_Int num_blocks=num_rows/tpb+1;
	VecScaleKernel<<<num_blocks,tpb>>>(u,v,l1_norm,num_rows);
#endif
}
}


extern "C"{

__global__
void VecCopyKernel(HYPRE_Complex* __restrict__ tgt, const HYPRE_Complex* __restrict__ src, HYPRE_Int size){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<size) tgt[i]=src[i];
}
void VecCopy(HYPRE_Complex* tgt, const HYPRE_Complex* src, HYPRE_Int size,cudaStream_t s){
	HYPRE_Int tpb=64;
	HYPRE_Int num_blocks=size/tpb+1;
	PUSH_RANGE_PAYLOAD("VecCopy",5,size);
	//MemPrefetch(tgt,0,s);
	//MemPrefetch(src,0,s);
	VecCopyKernel<<<num_blocks,tpb,0,s>>>(tgt,src,size);
	//hypre_CheckErrorDevice(cudaStreamSynchronize(s));
	POP_RANGE;
}
}
extern "C"{

__global__
void VecSetKernel(HYPRE_Complex* __restrict__ tgt, const HYPRE_Complex value,HYPRE_Int size){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<size) tgt[i]=value;
}
void VecSet(HYPRE_Complex* tgt, HYPRE_Int size, HYPRE_Complex value, cudaStream_t s){
	HYPRE_Int tpb=64;
	//cudaDeviceSynchronize();
	//printf("this is vec set, size %d \n", size);
	MemPrefetchSized(tgt,size*sizeof(HYPRE_Complex),HYPRE_DEVICE,s);
	HYPRE_Int num_blocks=size/tpb+1;
	VecSetKernel<<<num_blocks,tpb,0,s>>>(tgt,value,size);
	//cudaStreamSynchronize(s);
	//cudaDeviceSynchronize();
}
}
extern "C"{
__global__
void  PackOnDeviceKernel(HYPRE_Complex* __restrict__ send_data,const HYPRE_Complex* __restrict__ x_local_data, const HYPRE_Int* __restrict__ send_map, HYPRE_Int begin,HYPRE_Int end){
	HYPRE_Int i = begin+blockIdx.x * blockDim.x + threadIdx.x;
	if (i<end){
		send_data[i-begin]=x_local_data[send_map[i]];
	}
}
void PackOnDevice(HYPRE_Complex *send_data,HYPRE_Complex *x_local_data, HYPRE_Int *send_map, HYPRE_Int begin,HYPRE_Int end,cudaStream_t s){
	if ((end-begin)<=0) return;
	HYPRE_Int tpb=64;
	HYPRE_Int num_blocks=(end-begin)/tpb+1;
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	PackOnDeviceKernel<<<num_blocks,tpb,0,s>>>(send_data,x_local_data,send_map,begin,end);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	PUSH_RANGE("PACK_PREFETCH",1);
#ifndef HYPRE_GPU_USE_PINNED
	MemPrefetchSized((void*)send_data,(end-begin)*sizeof(HYPRE_Complex),cudaCpuDeviceId,s);
#endif
	POP_RANGE;
	//hypre_CheckErrorDevice(cudaStreamSynchronize(s));
}
}

//PackOnDeviceGPUONLY

//for GPU only data
extern "C"{
__global__
void  PackOnDeviceKernelGPUonly(HYPRE_Complex* __restrict__ send_data,const HYPRE_Complex* __restrict__ x_local_data, const HYPRE_Int* __restrict__ send_map, HYPRE_Int begin,HYPRE_Int end){
	HYPRE_Int i = begin+blockIdx.x * blockDim.x + threadIdx.x;
	if (i<end){
		// printf("putting %f in place %d \n",x_local_data[send_map[i]], i-begin);
		send_data[i-begin]=x_local_data[send_map[i]];
	}
}
void PackOnDeviceGPUonly(HYPRE_Complex *send_data,HYPRE_Complex *x_local_data, HYPRE_Int *send_map, HYPRE_Int begin,HYPRE_Int end){
	//printf("all right, inside GPUonly pack on device, begin %d end %d \n", begin, end);
	if ((end-begin)<=0) return;
	HYPRE_Int tpb=64;
	HYPRE_Int num_blocks=(end-begin)/tpb+1;
	PackOnDeviceKernelGPUonly<<<num_blocks,tpb>>>(send_data,x_local_data,send_map,begin,end);
	cudaDeviceSynchronize();
}

}




// Scale vector by scalar

extern "C"{
__global__
void VecScaleScalarKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex alpha ,HYPRE_Int num_rows){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	//if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
	if (i<num_rows){
		u[i]*=alpha;
		//if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
	}
}
}
extern "C"{
HYPRE_Int VecScaleScalar(HYPRE_Complex *u, const HYPRE_Complex alpha,  HYPRE_Int num_rows,cudaStream_t s){
	PUSH_RANGE("SEQVECSCALE",4);
	HYPRE_Int num_blocks=num_rows/64+1;

#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	VecScaleScalarKernel<<<num_blocks,64,0,s>>>(u,alpha,num_rows);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	// hypre_CheckErrorDevice(cudaStreamSynchronize(s));
	POP_RANGE;
	return 0;
}
}
//vec scale GPU only


extern "C"{
__global__
void VecScaleScalarGPUonlyKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex alpha ,HYPRE_Int num_rows){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		u[i]*=alpha;
	}
}
}

extern "C"{
HYPRE_Int VecScaleScalarGPUonly(HYPRE_Complex *u, const HYPRE_Complex alpha,  HYPRE_Int num_rows,cudaStream_t s){
	HYPRE_Int num_blocks=num_rows/64+1;

	VecScaleScalarGPUonlyKernel<<<num_blocks,64>>>(u,alpha,num_rows);
	return 0;
}
}


extern "C"{
	__global__
void SpMVCudaKernel(HYPRE_Complex* __restrict__ y,HYPRE_Complex alpha, const HYPRE_Complex* __restrict__ A_data, const HYPRE_Int* __restrict__ A_i, const HYPRE_Int* __restrict__ A_j, const HYPRE_Complex* __restrict__ x, HYPRE_Complex beta, HYPRE_Int num_rows)
{
	HYPRE_Int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		HYPRE_Complex temp = 0.0;
		HYPRE_Int jj;
		for (jj = A_i[i]; jj < A_i[i+1]; jj++){
			HYPRE_Int ajj=A_j[jj];
			temp += A_data[jj] * x[ajj];
		}
		y[i] =y[i]*beta+alpha*temp;
	}
}

	__global__
void SpMVCudaKernelZB(HYPRE_Complex* __restrict__ y,HYPRE_Complex alpha, const HYPRE_Complex* __restrict__ A_data, const HYPRE_Int* __restrict__ A_i, const HYPRE_Int* __restrict__ A_j, const HYPRE_Complex* __restrict__ x, HYPRE_Int num_rows)
{
	HYPRE_Int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		HYPRE_Complex temp = 0.0;
		HYPRE_Int jj;
		for (jj = A_i[i]; jj < A_i[i+1]; jj++){
			HYPRE_Int ajj=A_j[jj];
			temp += A_data[jj] * x[ajj];
		}
		y[i] = alpha*temp;
	}
}
void SpMVCuda(HYPRE_Int num_rows,HYPRE_Complex alpha, HYPRE_Complex *A_data,HYPRE_Int *A_i, HYPRE_Int *A_j, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y){
	HYPRE_Int num_threads=64;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	if (beta==0.0)
		SpMVCudaKernelZB<<<num_blocks,num_threads>>>(y,alpha,A_data,A_i,A_j,x,num_rows);
	else
		SpMVCudaKernel<<<num_blocks,num_threads>>>(y,alpha,A_data,A_i,A_j,x,beta,num_rows);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif

}
}
extern "C"{
__global__
void CompileFlagSafetyCheck(HYPRE_Int actual){
#ifdef __CUDA_ARCH__
	HYPRE_Int cudarch=__CUDA_ARCH__;
	if (cudarch!=actual){
		//printf("WARNING :: nvcc -arch flag does not match actual device architecture\nWARNING :: The code can fail silently and produce wrong results\n");
		//printf("Arch specified at compile = sm_%d Actual device = sm_%d\n",cudarch/10,actual/10);
	}
#else
	hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: CUDA_ ARCH is not defined \n This should not be happening\n");
#endif
}
}
extern "C"{
void CudaCompileFlagCheck(){
	HYPRE_Int devCount;
	cudaGetDeviceCount(&devCount);
	HYPRE_Int i;
	HYPRE_Int cudarch_actual;
	for(i = 0; i < devCount; ++i)
	{
		struct cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		cudarch_actual=props.major*100+props.minor*10;
	}
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
	CompileFlagSafetyCheck<<<1,1,0,0>>>(cudarch_actual);
	cudaError_t code=cudaPeekAtLastError();
	if (code != cudaSuccess)
	{
		hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR in CudaCompileFlagCheck \nERROR :: Check if compile arch flags match actual device arch = sm_\n");
		//fprintf(stderr,"ERROR in CudaCompileFlagCheck%s \n", cudaGetErrorString(code));
		//fprintf(stderr,"ERROR :: Check if compile arch flags match actual device arch = sm_%d\n",cudarch_actual/10);
		exit(2);
	}
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
}
}
//written by KS
//naive version

extern "C"{
__global__
void MassInnerProdKernel(HYPRE_Real * __restrict__ u,  HYPRE_Real ** __restrict__ v, HYPRE_Real * result, HYPRE_Int k, HYPRE_Int n){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<n){
		// KS we should fetch u to shared or to registers 
		int j;
#pragma unroll                
		for (j =0; j<k; ++j){
			//if ( blockIdx.x < 100000){printf("adding %f * %f  about to requrest v[%d][%d]\n", u[i], v[j][i], j, i);}
			//sum += u[i]*v[j][i];
			atomicAdd_system(&result[j], u[i]*v[j][i]);
		}
	}
}
}
//v2
extern "C"{
__global__
void MassInnerProdKernel_v1(HYPRE_Real * __restrict__ u,  HYPRE_Real ** __restrict__ v, HYPRE_Real * result, HYPRE_Int k, HYPRE_Int n){
	HYPRE_Int i = blockIdx.x * blockDim.x + threadIdx.x;
	//HYPRE_Int tid = threadIdx.x;

	int j;
	/*HYPRE_Real r_uk[150];
		volatile __shared__ HYPRE_Real s_u [BlockSize];
	// __shared__ HYPRE_Real  s_v [BlockSize][MaxSpace];
	s_u[tid] = u[i];
	for (j=0; j<k; j++){
	r_uk[j] = v[j][i];
	}
	__syncthreads();*/


	if (i<n){

#pragma unroll                
		for (j =0; j<k; ++j){
			atomicAdd_system(&result[j], u[i]*v[j][i]);
		}
	}
}
}
//`MassInnerProd(int, int, double**, double*, double*)
extern "C"{
void MassInnerProd(HYPRE_Int n, HYPRE_Int k, HYPRE_Real **v, HYPRE_Real *u, HYPRE_Real * result){

	HYPRE_Int num_threads=64;
	HYPRE_Int num_blocks=n/num_threads+1;
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
	printf("about to launch on (%d, %d), k = %d n = %d \n", num_blocks, num_threads, k, n);

	MassInnerProdKernel_v1<<<num_blocks, num_threads>>>(u, v, result, k,n);


	hypre_CheckErrorDevice(cudaDeviceSynchronize());

}
}



extern "C"{
__global__
void CSRMatvecTKernel_v1(HYPRE_Int num_rows, const HYPRE_Real * __restrict__ a, const HYPRE_Int * __restrict__ ia,const __restrict__  HYPRE_Int  * ja,const  HYPRE_Real * x, HYPRE_Real * y){

	/*
		 i = 0; num_rows-1
		 for (jj = A_i[i]; jj < A_i[i+1]; jj++)
		 {
		 j = A_j[jj];
		 y_data[j] += A_data[jj] * x_data[i];
		 }


*/

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;

	if (i<num_rows) {
		const double xx = x[i];
		for (j=ia[i]; j< ia[i+1]; j++){
			//    y[ja[j]] += a[j]*xx;

			if (abs(xx*a[j]) >  1e-16){
				//  atomicAdd(&y[ja[j]], 0.1f); 

				atomicAdd_system(&y[ja[j]], a[j]*xx);

			}
		}
	}

}
}
//v2 shared memory for x


extern "C"{
__global__
void CSRMatvecTKernel_v2(HYPRE_Int num_rows, const HYPRE_Real * __restrict__ a, const HYPRE_Int * __restrict__ ia,const __restrict__  HYPRE_Int  * ja,const  HYPRE_Real * x, HYPRE_Real * y){


	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j=threadIdx.x;
	__shared__ HYPRE_Real s_x[64];


	if (i<num_rows) {
		s_x[j] = x[i];
		__syncthreads();
		const double xx = s_x[j];
		for (j=ia[i]; j< ia[i+1]; j++){

			if (abs(xx*a[j]) >  1e-16){

				atomicAdd_system(&y[ja[j]], a[j]*xx);

			}
		}
	}

}
}
extern "C"{
void MatvecTCSR(HYPRE_Int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y){
	HYPRE_Int num_threads=64;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
	//  printf("blocks: %d threads %d \n", num_blocks, num_threads);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif

	CSRMatvecTKernel_v1<<<num_blocks,num_threads>>>(num_rows, a, ia, ja, x, y);
	cudaDeviceSynchronize();
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif

}


__global__ void massIPV7part1(const double * __restrict__ u,
		const double * __restrict__ v,
		double * result,
		const int k,
		const int N){

	//  int b = blockIdx.x;
	int t = threadIdx.x;
	int bsize = blockDim.x;

	// assume T threads per thread block (and k reductions to be performed)
	volatile __shared__ HYPRE_Real s_tmp[Tv5];

	// map between thread index space and the problem index space
	int j = blockIdx.x;
	s_tmp[t] = 0.0f;
	int nn =t;

	//printf ("nn = %d bsize = %d N = %d gridDim = %d j = %d\n", nn, bsize, N, gridDim.x, j);      
	while (nn<N){
		double can =  u[nn];

		double cbn = v[N*j+nn];
		//double can2, cbn2;
		s_tmp[t] += can*cbn;
		if ((nn+bsize)<N){
			can = u[nn+bsize];
			cbn = v[N*j +(nn+bsize)];
			s_tmp[t] += can*cbn;
		}
		//  else {can2 = 0.0f; cbn2=0.0f;}
		//    s_tmp[t] += (can*cbn + can2*cbn2);

		nn+=2*bsize;
	}


	__syncthreads();
	//if (j == 0) printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if(Tv5>=1024) {if (t<512) {s_tmp[t] += s_tmp[t+512];} __syncthreads(); }
	if (Tv5>=512) { if (t < 256) { s_tmp[t] += s_tmp[t + 256]; } __syncthreads(); }
	{ if (t < 128) { s_tmp[t] += s_tmp[t + 128]; } __syncthreads(); }
	{ if (t < 64) { s_tmp[t] += s_tmp[t + 64]; } __syncthreads(); }


	//if (t==0) printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if (t < 32)
	{
		s_tmp[t] += s_tmp[t+32];
		s_tmp[t] += s_tmp[t+16];

		s_tmp[t] += s_tmp[t+8];
		s_tmp[t] += s_tmp[t+4];
		s_tmp[t] += s_tmp[t+2];
		s_tmp[t] += s_tmp[t+1];
	}
	if (t == 0) {
		result[blockIdx.x] = s_tmp[0];
		//printf("putting %f in place %d \n", s_tmp[0], blockIdx.x+j*gridDim.x);
	}

}
//for two vectorsi

__global__ void MassIPTwoVec(const double * __restrict__ u1,const double * __restrict__ u2,   
		const double * __restrict__ v, 
		double * result,  
		const int k, 
		const int N){

	int b = blockIdx.x;
	int t = threadIdx.x;
	int bsize = blockDim.x;

	// assume T threads per thread block (and k reductions to be performed)
	volatile __shared__ HYPRE_Real s_tmp1[Tv5];

	volatile __shared__ HYPRE_Real s_tmp2[Tv5];
	// map between thread index space and the problem index space
	int j = blockIdx.x;
	s_tmp1[t] = 0.0f;
	s_tmp2[t] = 0.0f;
	int nn =t;
	double can1,can2, cbn;
	//printf ("nn = %d bsize = %d N = %d gridDim = %d j = %d\n", nn, bsize, N, gridDim.x, j);      
	while (nn<N){
		can1 =  u1[nn];
		can2 =  u2[nn];

		cbn = v[N*j+nn];
		s_tmp1[t] += can1*cbn;
		s_tmp2[t] += can2*cbn;


		nn+=bsize;
	}


	__syncthreads();
	//if (j == 0)	printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if(Tv5>=1024) {if (t<512)    { s_tmp1[t] += s_tmp1[t+512];    s_tmp2[t] += s_tmp2[t + 512];} __syncthreads(); }	
	if (Tv5>=512) { if (t < 256) { s_tmp1[t] += s_tmp1[t + 256];  s_tmp2[t] += s_tmp2[t + 256];} __syncthreads(); }
	{ if (t < 128)               { s_tmp1[t] += s_tmp1[t + 128];  s_tmp2[t] += s_tmp2[t + 128];} __syncthreads(); }
	{ if (t < 64)                { s_tmp1[t] += s_tmp1[t + 64];   s_tmp2[t] += s_tmp2[t + 64];} __syncthreads(); }


	//if (t==0) printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if (t < 32)
	{
		s_tmp1[t] += s_tmp1[t+32];
		s_tmp2[t] += s_tmp2[t+32];

		s_tmp1[t] += s_tmp1[t+16];
		s_tmp2[t] += s_tmp2[t+16];

		s_tmp1[t] += s_tmp1[t+8];
		s_tmp2[t] += s_tmp2[t+8];

		s_tmp1[t] += s_tmp1[t+4];
		s_tmp2[t] += s_tmp2[t+4];

		s_tmp1[t] += s_tmp1[t+2];
		s_tmp2[t] += s_tmp2[t+2];

		s_tmp1[t] += s_tmp1[t+1];
		s_tmp2[t] += s_tmp2[t+1];
	}
	if (t == 0) {
		result[blockIdx.x] = s_tmp1[0];
		result[blockIdx.x+k] = s_tmp2[0];
		//printf("putting %f in place %d \n", s_tmp[0], blockIdx.x+j*gridDim.x);
	} 

}
//end of code for two vectors

__global__ void MassIPV7part1withScaling(const double * __restrict__ u,
		const double * __restrict__ v,
		const double * scaleFactors,
		double * result,
		const int k,
		const int N){

	//  int b = blockIdx.x;
	int t = threadIdx.x;
	int bsize = blockDim.x;

	// assume T threads per thread block (and k reductions to be performed)
	volatile __shared__ HYPRE_Real s_tmp[Tv5];
	double s;
	// map between thread index space and the problem index space
	int j = blockIdx.x;
	s_tmp[t] = 0.0f;
	int nn =t;
	//printf ("nn = %d bsize = %d N = %d gridDim = %d j = %d\n", nn, bsize, N, gridDim.x, j);      
	while (nn<N){
		double can =  u[nn];

		double cbn = v[N*j+nn];
		//double can2, cbn2;
		s_tmp[t] += can*cbn;
		if ((nn+bsize)<N){
			can = u[nn+bsize];
			cbn = v[N*j +(nn+bsize)];
			s_tmp[t] += can*cbn;
		}
		//  else {can2 = 0.0f; cbn2=0.0f;}
		//    s_tmp[t] += (can*cbn + can2*cbn2);

		nn+=2*bsize;
	}


	__syncthreads();
	//if (j == 0) printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if(Tv5>=1024) {if (t<512) {s_tmp[t] += s_tmp[t+512];} __syncthreads(); }
	if (Tv5>=512) { if (t < 256) { s_tmp[t] += s_tmp[t + 256]; } __syncthreads(); }
	{ if (t < 128) { s_tmp[t] += s_tmp[t + 128]; } __syncthreads(); }
	{ if (t < 64) { s_tmp[t] += s_tmp[t + 64]; } __syncthreads(); }


	//if (t==0) printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
	if (t < 32)
	{
		s_tmp[t] += s_tmp[t+32];
		s_tmp[t] += s_tmp[t+16];

		s_tmp[t] += s_tmp[t+8];
		s_tmp[t] += s_tmp[t+4];
		s_tmp[t] += s_tmp[t+2];
		s_tmp[t] += s_tmp[t+1];
	}
	if (t == 0) {
		s = scaleFactors[blockIdx.x];
		//printf("scaling by %f \n", s);
		result[blockIdx.x] = s*s_tmp[0];
		//  printf("putting %f in place %d and mult by %f \n", s_tmp[0], blockIdx.x,s);
	}

}
__global__ void massAxpy3(int N,
		int k,
		const  double  * x_data,
		double *y_data,
		const  double   * alpha) {

	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	__shared__ double s_alpha[maxk];
	if (t<k) {s_alpha[t] = alpha[t];}
	__syncthreads();


	if (i < N) {
		double temp = 0.0f;
		for (int j=0; j<k; ++j){
			temp += x_data[j*N +i]*s_alpha[j];
		}
		y_data[i] -= temp;
	}
}

void InnerProdGPUonly(const double * __restrict__ u,
		const double * __restrict__ v,
		double *result,
		const int N) {
	//static cublasHandle_t myHandle;
	if (FirstCall){
		cublasCreate(&myHandle);
		FirstCall = 0;
	}
	cublasDdot (myHandle, N,
			u, 1,
			v, 1,
			&result[0]);

}
void AxpyGPUonly(const double * __restrict__ u,
		double * __restrict__ v,
		const double alpha,
		const int N) {
	//cublasHandle_t myHandle;
	// cublasCreate(&myHandle);
	if (FirstCall){
		//cublasHandle_t myHandle;
		cublasCreate(&myHandle);
		FirstCall = 0;
	}
	cublasDaxpy(myHandle, N,
			&alpha,
			u, 1,
			v, 1);

}
void ScaleGPUonly(double * __restrict__ u,
		const double alpha,
		const int N) {
	//cublasHandle_t myHandle;
	//cublasCreate(&myHandle);
	//static cublasHandle_t myHandle;
	if (FirstCall){
		//cublasHandle_t myHandle;
		cublasCreate(&myHandle);
		FirstCall = 0;
	}
	//int test;
	cublasDscal(myHandle, N,
			&alpha,
			u, 1);
	//printf("GPU returned %d \n", test);
}
void MassInnerProdGPUonly(const double * __restrict__ u,
		const double * __restrict__ v,
		double * result,
		const int k,
		const int N) {
	massIPV7part1<<<k, 1024>>>(u, v, result, k, N);

	//hypre_CheckErrorDevice(cudaDeviceSynchronize());
}

void MassInnerProdWithScalingGPUonly(const double * __restrict__ u,
		const double * __restrict__ v,
		const double * __restrict__ scaleFactors,
		double * result,
		const int k,
		const int N) {
	MassIPV7part1withScaling<<<k, 1024>>>(u, v,scaleFactors, result, k, N);

	//hypre_CheckErrorDevice(cudaDeviceSynchronize());
}

void MassInnerProdTwoVectorsGPUonly(const double * __restrict__ u1,const double * __restrict__ u2,
		const double * __restrict__ v,
		double * result,
		const int k,
		const int N) {
	MassIPTwoVec<<<k, 1024>>>(u1,u2, v, result, k, N);

	//hypre_CheckErrorDevice(cudaDeviceSynchronize());
}
void MassAxpyGPUonly(int N,
		int k,
		const  double  * x_data,
		double *y_data,
		const  double   * alpha){
	int  B = (N+384-1)/384;
	//  hypre_CheckErrorDevice(cudaDeviceSynchronize());
	massAxpy3<<<B, 384>>>( N,
			k,
			x_data,
			y_data,
			alpha);

}

__global__ void  GivensRotRightKernel(int N,
		int k1,int k2,
		double * q_data1,
		double * q_data2,
		double a1, double a2,double a3, double a4){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	while (i<N){
		double q1 = q_data1[i];
		q_data1[i] = q1*a1+q_data2[i]*a2;
		q_data2[i] = q1*a3+q_data2[i]*a4;
		i+= blockDim.x*gridDim.x;
	}

}

/* computes q1 = a1*q1+a2*q2; q2 = a3*q3+a4*q2*/
void GivensRotRight(int N,
		int k1,
		int k2,
		double  * q_data1,
		double  * q_data2,
		const  double   a1, const double a2, const double a3, const double a4){
	int  B = (N+384-1)/384;
	//  hypre_CheckErrorDevice(cudaDeviceSynchronize());
	GivensRotRightKernel<<<B, 384>>>( N,
			k1,k2,
			q_data1,
			q_data2,
			a1, a2, a3, a4);

}



}



//new code

extern "C"{
//y = alpha *A *x + beta*y;
__global__
void CSRMatvecAsynchKernel_v1(HYPRE_Int num_rows, const HYPRE_Real alpha, const HYPRE_Real * __restrict__ a, const HYPRE_Int * __restrict__ ia,const __restrict__  HYPRE_Int  * ja, HYPRE_Real * x, const HYPRE_Real beta, HYPRE_Real * y){

	/*
		 i = 0; num_rows-1
		 for (jj = A_i[i]; jj < A_i[i+1]; jj++)
		 {
		 j = A_j[jj];
		 y_data[j] += A_data[jj] * x_data[i];
		 }


*/

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;

	if (i<num_rows) {
		if (a[ia[i]]!=0.0){
			HYPRE_Real sum = 0.0f;
			for (j=ia[i]; j< ia[i+1]; j++){
				//if (i<5)
				//printf("I am thread %d, adding a[%d]*x[%d] =  %f *%f to the sum, result = %f \n", i,j, ja[j], a[j],x[ja[j]], sum+a[j]*x[ja[j]]);
				sum += a[j]*x[ja[j]];
			}
			// u_data[i] = res / A_diag_data[A_diag_i[i]];
			//if(i<5) printf("this is tread %d, sum is %f  dividing by %f and multiplying by %f \n", i,sum, a[ia[i]], alpha );
			sum*=alpha;
			sum/=a[ia[i]];
			if (abs(sum) >  1e-16){
				//if(i<5) printf("this is tread %d, sum is %f  adding to %f*%f+%f, result %f old: %f new %f \n", i,sum, beta,y[i],x[i],  sum+beta*y[i]+x[i], x[i], sum+beta*y[i]+x[i]);
				atomicAdd_system(&x[i], sum+beta*y[i]);
				//x[i] += sum+beta*y[i];
			}
		}
	}

}





void MatvecCSRAsynch(HYPRE_Int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,HYPRE_Int *ia, HYPRE_Int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y){
	HYPRE_Int num_threads=1024;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
	// printf("blocks: %d threads %d alpha %f beta %f \n", num_blocks, num_threads, alpha, beta);

	CSRMatvecAsynchKernel_v1<<<num_blocks,num_threads>>>(num_rows,alpha, a, ia, ja, x,beta, y);

}



__global__
void CSRMatvecAsynchTwoInOneKernel_v1(HYPRE_Int num_rows, const HYPRE_Real alpha, const HYPRE_Real * __restrict__ a1, const HYPRE_Int * __restrict__ ia1,const __restrict__  HYPRE_Int  * ja1, const HYPRE_Real * __restrict__ a2, const HYPRE_Int * __restrict__ ia2,const __restrict__  HYPRE_Int  * ja2,HYPRE_Real * x1,HYPRE_Real * x2, const HYPRE_Real beta, HYPRE_Real * y){

	/*
		 i = 0; num_rows-1
		 for (jj = A_i[i]; jj < A_i[i+1]; jj++)
		 {
		 j = A_j[jj];
		 y_data[j] += A_data[jj] * x_data[i];
		 }


*/

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;

	if (i<num_rows) { 
		HYPRE_Real sum = 0.0f;

		if (a1!=NULL){

			for (j=ia1[i]; j< ia1[i+1]; j++){
				//if (i<5)
				//printf("OFFD: I am thread %d, adding a[%d]*x[%d] =  %f *%f to the sum, result = %f \n", i,j, ja1[j], a1[j],x1[ja1[j]], sum+a1[j]*x1[ja1[j]]);
				sum += a1[j]*x1[ja1[j]];
			}

		}
		if (a2[ia2[i]]!=0.0){
			for (j=ia2[i]; j< ia2[i+1]; j++){
				//if (i<5)
				//printf("DIAG: I am thread %d, adding a[%d]*x[%d] =  %f *%f to the sum, result = %f \n", i,j, ja2[j], a2[j],x2[ja2[j]], sum+a2[j]*x2[ja2[j]]);
				sum += a2[j]*x2[ja2[j]];
			}
			// u_data[i] = res / A_diag_data[A_diag_i[i]];
			//if(i<5) printf("this is tread %d, sum is %f  dividing by %f and multiplying by %f \n", i,sum, a2[ia2[i]], alpha );
			sum*=alpha;
			sum += beta*y[i];
			sum/=a2[ia2[i]];
			if (abs(sum) >  1e-16){
				atomicAdd_system(&x2[i], sum);
				//x2[i] += sum;
				//+beta*y[i];
			}
		}
	}

}

void MatvecCSRAsynchTwoInOne(HYPRE_Int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a1,HYPRE_Int *ia1, HYPRE_Int *ja1,  HYPRE_Complex *a2,HYPRE_Int *ia2, HYPRE_Int *ja2,HYPRE_Complex *x1,HYPRE_Complex *x2, HYPRE_Complex beta, HYPRE_Complex *y){
	HYPRE_Int num_threads=1024;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
	// printf("blocks: %d threads %d alpha %f beta %f \n", num_blocks, num_threads, alpha, beta);

	CSRMatvecAsynchTwoInOneKernel_v1<<<num_blocks,num_threads>>>(num_rows,alpha, a1, ia1, ja1,a2,ia2,ja2, x1,x2,beta, y);

}


__global__
void CSRMatvecTwoInOneKernel_v1(HYPRE_Int num_rows, const HYPRE_Real alpha, const HYPRE_Real * __restrict__ a1, const HYPRE_Int * __restrict__ ia1,const __restrict__  HYPRE_Int  * ja1, const HYPRE_Real * __restrict__ a2, const HYPRE_Int * __restrict__ ia2,const __restrict__  HYPRE_Int  * ja2,HYPRE_Real * x1,HYPRE_Real * x2, const HYPRE_Real beta, HYPRE_Real * y, HYPRE_Real *z){


	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;

	if (i<num_rows) { 
		HYPRE_Real sum = 0.0f;

		if (a1!=NULL){

			for (j=ia1[i]; j< ia1[i+1]; j++){
				sum += a1[j]*x1[ja1[j]];
			}

		}
		if (a2!=NULL){      
			for (j=ia2[i]; j< ia2[i+1]; j++){
				sum += a2[j]*x2[ja2[j]];
			}
			sum*=alpha;
			sum += beta*y[i];
			y[i] = sum;
			if (a2[ia2[i]]!=0.0){sum/=a2[ia2[i]];}
			z[i] = sum;
		}//if
	}//if num rows

}
//second version - one block processes multiple rows


__global__
void CSRMatvecTwoInOneKernel_v2(HYPRE_Int num_rows, const HYPRE_Real alpha, const HYPRE_Real * __restrict__ a1, const HYPRE_Int * __restrict__ ia1,const __restrict__  HYPRE_Int  * ja1, const HYPRE_Real * __restrict__ a2, const HYPRE_Int * __restrict__ ia2,const __restrict__  HYPRE_Int  * ja2,HYPRE_Real * x1,HYPRE_Real * x2, const HYPRE_Real beta, HYPRE_Real * y, HYPRE_Real *z){


	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int j;

	int sz = floorf(num_rows/gridDim.x);
	int rows_per_block = sz + (num_rows%gridDim.x > bid);
	int first_row = bid*rows_per_block;
	int last_row = (bid+1)*rows_per_block;

	int block_start_diag = ia1[first_row]; 
	int block_end_diag = ia1[last_row];
	int nnz_diag = block_end_diag-block_start_diag;

	int block_start_offd = ia2[first_row]; 
	int block_end_offd = ia2[last_row];
	int nnz_offd = block_end_offd-block_start_offd;

	//load values to shared
	__shared__ double s_partSums1[MAX_SIZE];
	__shared__ double s_partSums2[MAX_SIZE];
	int j1=tid;
	//	int idx1 = ja1[tid];
	while (j1<nnz_diag){
		s_partSums1[j1] = a1[ia1[block_start_diag]+j1]*x1[ja1[block_start_diag+j1]];
		j1+= blockDim.x;
	}

	int j2=tid;
	//		int idx2 = ja2[i];
	while (j2<nnz_offd){
		s_partSums2[j2] = a2[ia2[block_start_offd]+j2]*x2[ja2[block_start_offd+j2]];
		j2+= blockDim.x;
	}
	__syncthreads();
	//reduction
	//doesnt matter, the matrices have the same # of rows

	if (tid<rows_per_block){
		int myRowStart_diag = ia1[first_row+tid]-block_start_diag ;  
		int myRowStart_offd = ia1[last_row+tid] - block_start_diag;
		int myRowEnd_diag   = ia2[first_row+tid]-block_start_offd ;  
		int myRowEnd_offd   = ia2[last_row+tid]-block_start_offd ;
		double sum = 0.0f;
		for (j=myRowStart_diag; j<= myRowEnd_diag; ++j)
		{
			sum += s_partSums1[j];
		}
		for (j=myRowStart_offd; j<= myRowEnd_offd; ++j)
		{
			sum += s_partSums2[j];
		}

		sum*=alpha;
		sum += beta*y[first_row+tid];
		y[first_row+tid] = sum;
		if (a2[ia2[first_row+tid]]!=0.0){sum/=a2[ia2[first_row+tid]];}
		z[first_row+tid] = sum;
	}
}

void MatvecCSRTwoInOne(HYPRE_Int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a1,HYPRE_Int *ia1, HYPRE_Int *ja1,  HYPRE_Complex *a2,HYPRE_Int *ia2, HYPRE_Int *ja2,HYPRE_Complex *x1,HYPRE_Complex *x2, HYPRE_Complex beta, HYPRE_Complex *y, HYPRE_Complex *z){
	HYPRE_Int num_threads=64;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
	// printf("blocks: %d threads %d alpha %f beta %f \n", num_blocks, num_threads, alpha, beta);
//printf("running v1\n");
	CSRMatvecTwoInOneKernel_v1<<<num_blocks,num_threads>>>(num_rows,alpha, a1, ia1, ja1,a2,ia2,ja2, x1,x2,beta, y, z);

}

//end of new code

__global__
void CSRMatvecAMGKernel_v1(HYPRE_Int num_rows, 
		const HYPRE_Real alpha, 
		const HYPRE_Real * __restrict__ a1, 
		const HYPRE_Int * __restrict__ ia1,
		const __restrict__  HYPRE_Int  * ja1,
		HYPRE_Real * x1,HYPRE_Real * x2, 
		const HYPRE_Real beta, 
		HYPRE_Real * y){
	//two inputs,one output

	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j;

	if (i<num_rows) { 
		HYPRE_Real sum = 0.0f;

		if (a1!=NULL){

			for (j=ia1[i]; j< ia1[i+1]; j++){
				sum += a1[j]*x1[ja1[j]];
			}

		}
		sum*=alpha;
		sum += beta*x2[i];
		//y_data[i] += 0.4*res / A_diag_data[A_diag_i[i]]; 
		sum = y[i] + x1[i]+0.8*sum/a1[ia1[i]]; 
		//  if (abs(sum) >  1e-16){
		y[i] = sum;
		// }
	}
}



void MatvecCSRAMG(HYPRE_Int num_rows,
		HYPRE_Complex alpha, 
		HYPRE_Complex *a1,
		HYPRE_Int *ia1, 
		HYPRE_Int *ja1,
		HYPRE_Complex *x1,
		HYPRE_Complex *x2, 
		HYPRE_Complex beta, 
		HYPRE_Complex *y){
	HYPRE_Int num_threads=1024;
	HYPRE_Int num_blocks=num_rows/num_threads+1;
	// printf("blocks: %d threads %d alpha %f beta %f \n", num_blocks, num_threads, alpha, beta);

	CSRMatvecAMGKernel_v1<<<num_blocks,num_threads>>>(num_rows,alpha, a1, ia1, ja1, x1,x2,beta, y);

}
}//extern C
//end of new code

#endif
