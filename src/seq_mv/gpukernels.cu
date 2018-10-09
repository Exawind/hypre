#if defined(HYPRE_USE_GPU)
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "_hypre_utilities.h"
#define BlockSize 64
#define MaxSpace 100
#define maxk 80
#define Tv5 1024

  static HYPRE_Int FirstCall=1; 
static cublasHandle_t myHandle;
extern "C"{
__global__
void VecScaleKernelText(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex *__restrict__ l1_norm, hypre_int num_rows){
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		u[i]+=__ldg(v+i)/__ldg(l1_norm+i);
	}
}
}

extern "C"{
__global__
void VecScaleKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex *__restrict__ v, const HYPRE_Complex * __restrict__ l1_norm, hypre_int num_rows){
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		u[i]+=v[i]/l1_norm[i];
	}
}
}

extern "C"{
void VecScale(HYPRE_Complex *u, HYPRE_Complex *v, HYPRE_Complex *l1_norm, hypre_int num_rows,cudaStream_t s){
	PUSH_RANGE_PAYLOAD("VECSCALE",1,num_rows);
	const hypre_int tpb=64;
	hypre_int num_blocks=num_rows/tpb+1;
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
	hypre_CheckErrorDevice(cudaStreamSynchronize(s));
	POP_RANGE;
}
}


extern "C"{

__global__
void VecCopyKernel(HYPRE_Complex* __restrict__ tgt, const HYPRE_Complex* __restrict__ src, hypre_int size){
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<size) tgt[i]=src[i];
}
void VecCopy(HYPRE_Complex* tgt, const HYPRE_Complex* src, hypre_int size,cudaStream_t s){
	hypre_int tpb=64;
	hypre_int num_blocks=size/tpb+1;
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
void VecSetKernel(HYPRE_Complex* __restrict__ tgt, const HYPRE_Complex value,hypre_int size){
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<size) tgt[i]=value;
}
void VecSet(HYPRE_Complex* tgt, hypre_int size, HYPRE_Complex value, cudaStream_t s){
	hypre_int tpb=64;
	//cudaDeviceSynchronize();
#if defined(HYPRE_USE_MANAGED)
printf("using managed fucking memory! %d\n", HYPRE_USE_MANAGED);
	MemPrefetchSized(tgt,size*sizeof(HYPRE_Complex),HYPRE_DEVICE,s);
	hypre_int num_blocks=size/tpb+1;
	VecSetKernel<<<num_blocks,tpb,0,s>>>(tgt,value,size);
	cudaStreamSynchronize(s);
	//cudaDeviceSynchronize();
#endif
#if defined(HYPRE_USE_GPU) && !defined(HYPRE_USE_MANAGED)
// not using unified
cudaMemset(tgt,value,size*sizeof(HYPRE_Complex));		
#endif

}
}
extern "C"{
__global__
void  PackOnDeviceKernel(HYPRE_Complex* __restrict__ send_data,const HYPRE_Complex* __restrict__ x_local_data, const hypre_int* __restrict__ send_map, hypre_int begin,hypre_int end){
	hypre_int i = begin+blockIdx.x * blockDim.x + threadIdx.x;
	if (i<end){
		send_data[i-begin]=x_local_data[send_map[i]];
	}
}
void PackOnDevice(HYPRE_Complex *send_data,HYPRE_Complex *x_local_data, hypre_int *send_map, hypre_int begin,hypre_int end,cudaStream_t s){
	if ((end-begin)<=0) return;
	hypre_int tpb=64;
	hypre_int num_blocks=(end-begin)/tpb+1;
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

// Scale vector by scalar

extern "C"{
__global__
void VecScaleScalarKernel(HYPRE_Complex *__restrict__ u, const HYPRE_Complex alpha ,hypre_int num_rows){
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	//if (i<5) printf("DEVICE %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
	if (i<num_rows){
		u[i]*=alpha;
		//if (i==0) printf("Diff Device %d %lf %lf %lf\n",i,u[i],v[i],l1_norm[i]);
	}
}
}
extern "C"{
hypre_int VecScaleScalar(HYPRE_Complex *u, const HYPRE_Complex alpha,  hypre_int num_rows,cudaStream_t s){
	PUSH_RANGE("SEQVECSCALE",4);
	hypre_int num_blocks=num_rows/64+1;

#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	VecScaleScalarKernel<<<num_blocks,64,0,s>>>(u,alpha,num_rows);
#ifdef CATCH_LAUNCH_ERRORS
	hypre_CheckErrorDevice(cudaPeekAtLastError());
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
	hypre_CheckErrorDevice(cudaStreamSynchronize(s));
	POP_RANGE;
	return 0;
}
}


extern "C"{
	__global__
void SpMVCudaKernel(HYPRE_Complex* __restrict__ y,HYPRE_Complex alpha, const HYPRE_Complex* __restrict__ A_data, const hypre_int* __restrict__ A_i, const hypre_int* __restrict__ A_j, const HYPRE_Complex* __restrict__ x, HYPRE_Complex beta, hypre_int num_rows)
{
	hypre_int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		HYPRE_Complex temp = 0.0;
		hypre_int jj;
		for (jj = A_i[i]; jj < A_i[i+1]; jj++){
			hypre_int ajj=A_j[jj];
			temp += A_data[jj] * x[ajj];
		}
		y[i] =y[i]*beta+alpha*temp;
	}
}

	__global__
void SpMVCudaKernelZB(HYPRE_Complex* __restrict__ y,HYPRE_Complex alpha, const HYPRE_Complex* __restrict__ A_data, const hypre_int* __restrict__ A_i, const hypre_int* __restrict__ A_j, const HYPRE_Complex* __restrict__ x, hypre_int num_rows)
{
	hypre_int i= blockIdx.x * blockDim.x + threadIdx.x;
	if (i<num_rows){
		HYPRE_Complex temp = 0.0;
		hypre_int jj;
		for (jj = A_i[i]; jj < A_i[i+1]; jj++){
			hypre_int ajj=A_j[jj];
			temp += A_data[jj] * x[ajj];
		}
		y[i] = alpha*temp;
	}
}
void SpMVCuda(hypre_int num_rows,HYPRE_Complex alpha, HYPRE_Complex *A_data,hypre_int *A_i, hypre_int *A_j, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y){
	hypre_int num_threads=64;
	hypre_int num_blocks=num_rows/num_threads+1;
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
void CompileFlagSafetyCheck(hypre_int actual){
#ifdef __CUDA_ARCH__
	hypre_int cudarch=__CUDA_ARCH__;
	if (cudarch!=actual){
		printf("WARNING :: nvcc -arch flag does not match actual device architecture\nWARNING :: The code can fail silently and produce wrong results\n");
		printf("Arch specified at compile = sm_%d Actual device = sm_%d\n",cudarch/10,actual/10);
	} 
#else
	printf("ERROR:: CUDA_ ARCH is not defined \n This should not be happening\n");
#endif
}
}
extern "C"{
void CudaCompileFlagCheck(){
	hypre_int devCount;
	cudaGetDeviceCount(&devCount);
	hypre_int i;
	hypre_int cudarch_actual;
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
		fprintf(stderr,"ERROR in CudaCompileFlagCheck%s \n", cudaGetErrorString(code));
		fprintf(stderr,"ERROR :: Check if compile arch flags match actual device arch = sm_%d\n",cudarch_actual/10);
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
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
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
	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;
	hypre_int tid = threadIdx.x;

	int j;	
	HYPRE_Real r_uk[150];
	volatile __shared__ HYPRE_Real s_u [BlockSize];
	// __shared__ HYPRE_Real  s_v [BlockSize][MaxSpace];
	s_u[tid] = u[i];
	for (j=0; j<k; j++){
		r_uk[j] = v[j][i];
	}
	__syncthreads();  


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

	hypre_int num_threads=64;
	hypre_int num_blocks=n/num_threads+1;
	hypre_CheckErrorDevice(cudaDeviceSynchronize());
	printf("about to launch on (%d, %d), k = %d n = %d \n", num_blocks, num_threads, k, n);

	MassInnerProdKernel_v1<<<num_blocks, num_threads>>>(u, v, result, k,n);


	hypre_CheckErrorDevice(cudaDeviceSynchronize());

}
}
/**hypre_int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,hypre_int *ia, hypre_int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y */
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
				//	atomicAdd(&y[ja[j]], 0.1f); 

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
void MatvecTCSR(hypre_int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,hypre_int *ia, hypre_int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y){
	hypre_int num_threads=64;
	hypre_int num_blocks=num_rows/num_threads+1;
	//	printf("blocks: %d threads %d \n", num_blocks, num_threads);
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
}

extern "C"{
__global__
void ParRelaxKernel(
		HYPRE_Int n,
		HYPRE_Int relax_points,
		HYPRE_Int *__restrict__ cf_marker,
		HYPRE_Int *__restrict__ A_diag_i,
		HYPRE_Int *__restrict__ A_diag_j,
		HYPRE_Real *__restrict__ A_diag_data,
		HYPRE_Int *__restrict__ A_offd_i,
		HYPRE_Int *__restrict__ A_offd_j,
		HYPRE_Real *__restrict__ A_offd_data,
		HYPRE_Real *__restrict__ Vext_data,
		HYPRE_Real *__restrict__ f_data,
		HYPRE_Real *__restrict__ u_data){

	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n &&
			cf_marker[i] == relax_points && 
			A_diag_data[A_diag_i[i]] != 0.0)
	{
		HYPRE_Real res = f_data[i];
		for (int jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
		{
			int ii = A_diag_j[jj];
			if (ii>=i){          
				res -= A_diag_data[jj] * u_data[ii];
			}}
		for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
		{
			int ii = A_offd_j[jj];
			res -= A_offd_data[jj] * Vext_data[ii];
		}
		u_data[i] = res / A_diag_data[A_diag_i[i]];
	}
}

void ParRelax(
		HYPRE_Int n,
		HYPRE_Int relax_points,
		HYPRE_Int * __restrict__ cf_marker,
		HYPRE_Int * __restrict__ A_diag_i,
		HYPRE_Int * __restrict__ A_diag_j,
		HYPRE_Real * __restrict__ A_diag_data,
		HYPRE_Int * __restrict__ A_offd_i,
		HYPRE_Int * __restrict__ A_offd_j,
		HYPRE_Real * __restrict__ A_offd_data,
		HYPRE_Real * __restrict__ Vext_data,
		HYPRE_Real * __restrict__ f_data,
		HYPRE_Real * u_data) {

	hypre_int num_threads=128;
	hypre_int num_blocks=n / num_threads + 1;

	/*     HYPRE_Real * d_u_data_out = NULL;
				 cudaMalloc(&d_u_data_out, n * sizeof(HYPRE_Real));

				 cudaMemset(d_u_data_out, 0, n * sizeof(HYPRE_Real));
	 */
	ParRelaxKernel<<<num_blocks, num_threads>>>(n, relax_points, cf_marker, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data, Vext_data, f_data, u_data);

	/*
		 cudaMemcpy(u_data, d_u_data_out, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);

		 cudaFree(d_u_data_out);
	 */
}
}

//L1Jacobi

extern "C"{
__global__
void ParRelaxL1JacobiKernel(HYPRE_Int n,
		HYPRE_Real * l1_norms,	
		HYPRE_Real  relax_weight,
		HYPRE_Int * A_diag_i,
		HYPRE_Int * A_diag_j,
		HYPRE_Real * A_diag_data,
		HYPRE_Int * A_offd_i,
		HYPRE_Int * A_offd_j,
		HYPRE_Real *  A_offd_data,
		HYPRE_Real *  Vtemp_data,
		HYPRE_Real * Vext_data,
		HYPRE_Real * f_data,
		HYPRE_Real *  u_data){

	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n){
		int ii, jj;
		HYPRE_Real res;
		if (A_diag_data[A_diag_i[i]] != 0.0)
		{
			res = f_data[i];
			for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
			{
				ii = A_diag_j[jj];
				res -= A_diag_data[jj] * Vtemp_data[ii];
			}
			for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
			{
				ii = A_offd_j[jj];
				res -= A_offd_data[jj] * Vext_data[ii];
			}
			u_data[i] += (relax_weight*res)/l1_norms[i];
		}


	}
}

/*
	 void ParRelaxL1Jacobi(
	 HYPRE_Int n,
	 HYPRE_Real * __restrict__ l1_data,
	 HYPRE_Real __restrict__ relax_weight,
	 HYPRE_Int *__restrict__ A_diag_i,
	 HYPRE_Int *__restrict__ A_diag_j,
	 HYPRE_Real *__restrict__ A_diag_data,
	 HYPRE_Int *__restrict__ A_offd_i,
	 HYPRE_Int *__restrict__ A_offd_j,
	 HYPRE_Real *__restrict__ A_offd_data,
	 HYPRE_Real *__restrict__ Vtemp_data,
	 HYPRE_Real *__restrict__ f_data,
	 HYPRE_Real *__restrict__ u_data);
 */
/*(int, double*, double, int*, int*, double*, int*, int*, double*, double*, double*, double*)*/
void ParRelaxL1Jacobi(HYPRE_Int n,
		HYPRE_Real *  l1_data,	
		HYPRE_Real relax_weight,
		HYPRE_Int * A_diag_i,
		HYPRE_Int * A_diag_j,
		HYPRE_Real * A_diag_data,
		HYPRE_Int * A_offd_i,
		HYPRE_Int *  A_offd_j,
		HYPRE_Real *  A_offd_data,
		HYPRE_Real *  Vtemp_data,
		HYPRE_Real * Vext_data,
		HYPRE_Real * f_data,
		HYPRE_Real * u_data) {

	hypre_int num_threads=128;
	hypre_int num_blocks=n / num_threads + 1;

	/*     HYPRE_Real * d_u_data_out = NULL;
				 cudaMalloc(&d_u_data_out, n * sizeof(HYPRE_Real));

				 cudaMemset(d_u_data_out, 0, n * sizeof(HYPRE_Real));
	 */
	ParRelaxL1JacobiKernel<<<num_blocks, num_threads>>>(n,l1_data,relax_weight, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data, Vtemp_data, Vext_data, f_data, u_data);

	/*
		 cudaMemcpy(u_data, d_u_data_out, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);

		 cudaFree(d_u_data_out);
	 */
}


__global__
void ParRelaxL1JacobiCFKernel(HYPRE_Int n,
		HYPRE_Int * cf_marker,	
		HYPRE_Int  relax_points,
		HYPRE_Real relax_weight,
		HYPRE_Real * l1_norms,
		HYPRE_Int * A_diag_i,
		HYPRE_Int * A_diag_j,
		HYPRE_Real * A_diag_data,
		HYPRE_Int * A_offd_i,
		HYPRE_Int * A_offd_j,
		HYPRE_Real *  A_offd_data,
		HYPRE_Real *  Vtemp_data,
		HYPRE_Real * Vext_data,
		HYPRE_Real * f_data,
		HYPRE_Real *  u_data){

	hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;

	/*-----------------------------------------------------------
	 * If i is of the right type ( C or F ) and diagonal is
	 * nonzero, relax point i; otherwise, skip it.
	 *-----------------------------------------------------------*/
	HYPRE_Int ii,jj;
	HYPRE_Real res;
	if (i<n && cf_marker[i] == relax_points
			&& A_diag_data[A_diag_i[i]] != 0.0f)
	{
		res = f_data[i];
		for (jj = A_diag_i[i]; jj < A_diag_i[i+1]; jj++)
		{
			ii = A_diag_j[jj];
			res -= A_diag_data[jj] * Vtemp_data[ii];
		}
		for (jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
		{
			ii = A_offd_j[jj];
			res -= A_offd_data[jj] * Vext_data[ii];
			//printf("i=%d, Vext_data[%d] =  %f \n", i, ii, Vext_data[ii]);
		}
		u_data[i] += (relax_weight * res)/l1_norms[i];
	}


}//kernel

void ParRelaxL1JacobiCF(HYPRE_Int n,
		HYPRE_Int *  cf_marker,	
		HYPRE_Int relax_points,
		HYPRE_Real  relax_weight,
		HYPRE_Real * l1_norms,
		HYPRE_Int * A_diag_i,
		HYPRE_Int * A_diag_j,
		HYPRE_Real * A_diag_data,
		HYPRE_Int * A_offd_i,
		HYPRE_Int *  A_offd_j,
		HYPRE_Real *  A_offd_data,
		HYPRE_Real *  Vtemp_data,
		HYPRE_Real * Vext_data,
		HYPRE_Real * f_data,
		HYPRE_Real * u_data) {

	hypre_int num_threads=128;
	hypre_int num_blocks=n / num_threads + 1;

	/*     HYPRE_Real * d_u_data_out = NULL;
				 cudaMalloc(&d_u_data_out, n * sizeof(HYPRE_Real));

				 cudaMemset(d_u_data_out, 0, n * sizeof(HYPRE_Real));
	 */
	ParRelaxL1JacobiCFKernel<<<num_blocks, num_threads>>>(n,cf_marker,relax_points,relax_weight,l1_norms, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data, Vtemp_data, Vext_data, f_data, u_data);

	/*
		 cudaMemcpy(u_data, d_u_data_out, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToDevice);

		 cudaFree(d_u_data_out);
	 */
}//kernel





__global__ void massIPV7part1(const double * __restrict__ u,  
		const double * __restrict__ v, 
		double * result,  
		const int k, 
		const int N){

	int b = blockIdx.x;
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
		//	else {can2 = 0.0f; cbn2=0.0f;}
		//		s_tmp[t] += (can*cbn + can2*cbn2);

		nn+=2*bsize;
	}


	__syncthreads();
	//if (j == 0)	printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
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
// get IPs AND SCALE


__global__ void MassIPV7part1withScaling(const double * __restrict__ u,  
		const double * __restrict__ v,
    const double * scaleFactors, 
		double * result,  
		const int k, 
		const int N){

	int b = blockIdx.x;
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
		//	else {can2 = 0.0f; cbn2=0.0f;}
		//		s_tmp[t] += (can*cbn + can2*cbn2);

		nn+=2*bsize;
	}


	__syncthreads();
	//if (j == 0)	printf("I am block %d, t=%d, s_tmp[%d] = %f\n", j, t,t, s_tmp[t]);
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
	//	printf("putting %f in place %d and mult by %f \n", s_tmp[0], blockIdx.x,s);
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

/*
const double * __restrict__ u,  
		const double * __restrict__ v, 
		double * result,  
		const int k, 
		const int N)

*/


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
int test;
test = cublasDscal(myHandle, N,
                            &alpha,
                            u, 1);
//printf("GPU returned %d \n", test);
}

/*
void ScaleGPUonly(const double * __restrict__ u, 
		const double alpha, 
		const int N);
void AxpyGPUonly(const double * __restrict__ u,  
		 double * __restrict__ v,
		const double alpha, 
		const int N); 
void InnerProdGPUonly(const double * __restrict__ u,  
		const double * __restrict__ v, 
		double result, 
		const int N);
*/
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
/*
void MassAxpyGPUonly(int N,    int k,
    const  double  * x_data,
    double *y_data,
    const  double   * alpha);

*/

void MassAxpyGPUonly(int N, 
		int k, 
		const  double  * x_data, 
		double *y_data,
		const  double   * alpha){
int  B = (N+384-1)/384;
//	hypre_CheckErrorDevice(cudaDeviceSynchronize());
massAxpy3<<<B, 384>>>( N, 
		k, 
		x_data, 
		y_data,
		alpha);

}



//end of KS code
}//externn C
#endif
