#ifdef HYPRE_USE_GPU
#include <cuda_runtime_api.h>
int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s);
void VecCopy(double* tgt, const double* src, int size,cudaStream_t s);
void VecSet(double* tgt, int size, double value, cudaStream_t s);
void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
void MatvecTCSR(int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,hypre_int *ia, hypre_int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y);
void MassInnerProd(HYPRE_Int n, HYPRE_Int k, HYPRE_Real **v, HYPRE_Real *u, HYPRE_Real *result);
void MassAxpyGPUonly(int N,    int k,
    const  double  * x_data,
    double *y_data,
    const  double   * alpha);
void MassInnerProdGPUonly(const double * __restrict__ u,  
		const double * __restrict__ v, 
		double * result,  
		const int k, 
		const int N);

void MassInnerProdWithScalingGPUonly(const double * __restrict__ u,  
		const double * __restrict__ v,
    const double * __restrict__ scaleFactors, 
		double * result,  
		const int k, 
		const int N);


void ScaleGPUonly(double * __restrict__ u, 
		const double alpha, 
		const int N);
void AxpyGPUonly(const double * __restrict__ u,  
		 double * __restrict__ v,
		const double alpha, 
		const int N); 
void InnerProdGPUonly(const double * __restrict__ u,  
		const double * __restrict__ v, 
		double *result, 
		const int N);


void ParRelaxL1Jacobi(
		HYPRE_Int n,
		HYPRE_Real * __restrict__ l1_data,
		HYPRE_Real  relax_weight,
		HYPRE_Int *__restrict__ A_diag_i,
		HYPRE_Int *__restrict__ A_diag_j,
		HYPRE_Real *__restrict__ A_diag_data,
		HYPRE_Int *__restrict__ A_offd_i,
		HYPRE_Int *__restrict__ A_offd_j,
		HYPRE_Real *__restrict__ A_offd_data,
		HYPRE_Real *__restrict__ Vtemp_data,
HYPRE_Real * __restrict__ Vext_data,
		HYPRE_Real *__restrict__ f_data,
		HYPRE_Real *__restrict__ u_data);

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
                HYPRE_Real * u_data);
#endif
