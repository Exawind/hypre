#ifdef HYPRE_USING_GPU
#include <cuda_runtime_api.h>
int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s);

int VecScaleScalarGPUonly(double *u, const double alpha,  int num_rows,cudaStream_t s);
void VecCopy(double* tgt, const double* src, int size,cudaStream_t s);
void VecSet(double* tgt, int size, double value, cudaStream_t s);
void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
void MatvecTCSR(int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,hypre_int *ia, hypre_int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y);
void MatvecCSRAsynch(int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a,hypre_int *ia, hypre_int *ja, HYPRE_Complex *x, HYPRE_Complex beta, HYPRE_Complex *y);
void MatvecCSRAsynchTwoInOne(int num_rows,HYPRE_Complex alpha, HYPRE_Complex *a1,hypre_int *ia1, hypre_int *ja1,  HYPRE_Complex *a2,hypre_int *ia2, hypre_int *ja2, HYPRE_Complex *x1,  HYPRE_Complex *x2,HYPRE_Complex beta, HYPRE_Complex *y);
void MassInnerProd(HYPRE_Int n, HYPRE_Int k, HYPRE_Real **v, HYPRE_Real *u, HYPRE_Real *result);

void MassInnerProdTwoVectorsGPUonly(const double * __restrict__ u1,const double * __restrict__ u2,
    const double * __restrict__ v,
    double * result1,
    const int k,
    const int N);

void MassAxpyGPUonly(int N,    int k,
    const  double  * x_data,
    double *y_data,
    const  double   * alpha);



void AxpyGPUonly(int N,    int k,
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

void GivensRotRight(int N,
    int k1,
    int k2,
    double  * q_data1,
    double  * q_data2,
    const  double   a1, const double a2, const double a3, const double a4);
#endif
