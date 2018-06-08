#ifdef HYPRE_USE_GPU

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif 

int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s);
void VecCopy(double* tgt, const double* src, int size,cudaStream_t s);
void VecSet(double* tgt, int size, double value, cudaStream_t s);
void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void ParRelax(
      int n,
      int relax_points,
      int * cf_marker,
      int * A_diag_i,
      int * A_diag_j,
      double * A_diag_data,
      int * A_offd_i,
      int * A_offd_j,
      double * A_offd_data,
      double * Vext_data,
      double * f_data,
      double * u_data);
void CudaCompileFlagCheck();
#ifdef __cplusplus
}
#endif 

#endif
