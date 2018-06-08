#ifdef HYPRE_USE_GPU
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "_hypre_utilities.h"
#include "gpuErrorCheck.h"

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

         HYPRE_Int * d_cf_marker = NULL;
         checkCudaFunc(cudaMalloc(&d_cf_marker, n * sizeof(HYPRE_Int)));
         checkCudaFunc(cudaMemcpy(d_cf_marker, cf_marker, n * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));

         HYPRE_Int * d_A_diag_i = NULL;
         checkCudaFunc(cudaMalloc(&d_A_diag_i, n * sizeof(HYPRE_Int)));
         checkCudaFunc(cudaMemcpy(d_A_diag_i, A_diag_i, (n + 1) * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));

         HYPRE_Int * d_A_diag_j = NULL;
         checkCudaFunc(cudaMalloc(&d_A_diag_j * sizeof(HYPRE_Int)));
         checkCudaFunc(cudaMemcpy(d_A_diag_j, A_diag_j, (A_diag_i[n]) * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));

         HYPRE_Real * d_A_diag_data = NULL;
         checkCudaFunc(cudaMalloc(&d_A_diag_data * sizeof(HYPRE_Real)));
         checkCudaFunc(cudaMemcpy(d_A_diag_data, A_diag_data, (A_diag_i[n]) * sizeof(HYPRE_Real), cudaMemcpyHostToDevice));

         HYPRE_Int * d_A_offd_i = NULL;
         checkCudaFunc(cudaMalloc(&d_A_offd_i, n * sizeof(HYPRE_Int)));
         checkCudaFunc(cudaMemcpy(d_A_offd_i, A_offd_i, (n + 1) * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));

         HYPRE_Int * d_A_offd_j = NULL;
         checkCudaFunc(cudaMalloc(&d_A_offd_j * sizeof(HYPRE_Int)));
         checkCudaFunc(cudaMemcpy(d_A_offd_j, A_offd_j, (A_offd_i[n]) * sizeof(HYPRE_Int), cudaMemcpyHostToDevice));

         HYPRE_Real * d_A_offd_data = NULL;
         checkCudaFunc(cudaMalloc(&d_A_offd_data * sizeof(HYPRE_Real)));
         checkCudaFunc(cudaMemcpy(d_A_offd_data, A_offd_data, (A_offd_i[n]) * sizeof(HYPRE_Real), cudaMemcpyHostToDevice));

         HYPRE_Real * d_Vext_data = NULL;
         checkCudaFunc(cudaMalloc(&d_Vext_data * sizeof(HYPRE_Real)));
         checkCudaFunc(cudaMemcpy(d_Vext_data, Vext_data, (n) * sizeof(HYPRE_Real), cudaMemcpyHostToDevice));

         HYPRE_Real * d_f_data = NULL;
         checkCudaFunc(cudaMalloc(&d_f_data * sizeof(HYPRE_Real)));
         checkCudaFunc(cudaMemcpy(d_f_data, f_data, (n) * sizeof(HYPRE_Real), cudaMemcpyHostToDevice);)

         HYPRE_Real * d_u_data = NULL;
         checkCudaFunc(cudaMalloc(&d_u_data * sizeof(HYPRE_Real)));
         checkCudaFunc(cudaMemcpy(d_u_data, u_data, (n) * sizeof(HYPRE_Real), cudaMemcpyHostToDevice));

         HYPRE_Real * d_u_data_out = NULL;
         checkCudaFunc(cudaMalloc(&d_u_data_out, n * sizeof(HYPRE_Real)));

         checkCudaFunc(cudaMemset(d_u_data_out, 0, n * sizeof(HYPRE_Real)));

         ParRelaxKernel<<<num_blocks, num_threads>>>(n, relax_points, d_cf_marker, d_A_diag_i, d_A_diag_j, d_A_diag_data, d_A_offd_i, d_A_offd_j, d_A_offd_data, d_Vext_data, d_f_data, d_u_data, d_u_data_out);

         checkCudaLastError();

         checkCudaFunc(cudaMemcpy(u_data, d_u_data_out, n * sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

         checkCudaFunc(cudaFree(d_cf_marker));
         checkCudaFunc(cudaFree(d_A_diag_i));
         checkCudaFunc(cudaFree(d_A_diag_j));
         checkCudaFunc(cudaFree(d_A_diag_data));
         checkCudaFunc(cudaFree(d_A_offd_i);
         checkCudaFunc(cudaFree(d_A_offd_j));
         checkCudaFunc(cudaFree(d_A_offd_data));
         checkCudaFunc(cudaFree(d_Vext_data));
         checkCudaFunc(cudaFree(d_f_data));
         checkCudaFunc(cudaFree(d_u_data));

         checkCudaFunc(cudaFree(d_u_data_out));
   }
}
#endif
