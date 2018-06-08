#ifdef HYPRE_USE_GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "_hypre_utilities.h"

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
      HYPRE_Real *__restrict__ u_data,
      HYPRE_Real *__restrict__ u_data_out){

      hypre_int i = blockIdx.x * blockDim.x + threadIdx.x;

      if (i < n &&
          cf_marker[i] == relax_points &&
          A_diag_data[A_diag_i[i]] != 0.0)
      {
         HYPRE_Real res = f_data[i];
         for (int jj = A_diag_i[i]+1; jj < A_diag_i[i+1]; jj++)
         {
            int ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
         }
         for (int jj = A_offd_i[i]; jj < A_offd_i[i+1]; jj++)
         {
            int ii = A_offd_j[jj];
            res -= A_offd_data[jj] * Vext_data[ii];
         }
         u_data_out[i] = res / A_diag_data[A_diag_i[i]];
      }
   }

   void ParRelax(
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
      HYPRE_Real *__restrict__ u_data) {

         hypre_int num_threads=128;
         hypre_int num_blocks=n / num_threads + 1;

         HYPRE_Real * d_u_data_out = NULL;
         cudaMalloc(&d_u_data_out, n * sizeof(HYPRE_Real));

         cudaMemset(d_u_data_out, 0, n * sizeof(HYPRE_Real));

         ParRelaxKernel<<<num_blocks, num_threads>>>(n, relax_points, cf_marker, A_diag_i, A_diag_j, A_diag_data, A_offd_i, A_offd_j, A_offd_data, Vext_data, f_data, u_data, d_u_data_out);

         cudaMemcpy(u_data, d_u_data_out, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

         cudaFree(d_u_data_out);
   }
}
#endif
