/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/
#include "seq_mv.h"
#include <assert.h>

#ifdef HYPRE_USING_GPU
#include <cublas_v2.h>
#include <cusparse.h>
#include "gpukernels.h"
#endif

#define NUM_TEAMS 128
#define NUM_THREADS 1024
/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/

  hypre_Vector *
hypre_SeqVectorCreate( HYPRE_Int size )
{
  hypre_Vector  *vector;

  vector =  hypre_CTAlloc(hypre_Vector,  1, HYPRE_MEMORY_HOST);

#ifdef HYPRE_USING_GPU
  vector->on_device=0;
#endif
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  vector->mapped=0;
  vector->drc=0;
  vector->hrc=0;
#endif
  hypre_VectorData(vector) = NULL;
  hypre_VectorSize(vector) = size;

  hypre_VectorNumVectors(vector) = 1;
  hypre_VectorMultiVecStorageMethod(vector) = 0;

  /* set defaults */
  hypre_VectorOwnsData(vector) = 1;

  return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

  hypre_Vector *
hypre_SeqMultiVectorCreate( HYPRE_Int size, HYPRE_Int num_vectors )
{
  hypre_Vector *vector = hypre_SeqVectorCreate(size);
  hypre_VectorNumVectors(vector) = num_vectors;
  return vector;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorDestroy( hypre_Vector *vector )
{
  HYPRE_Int  ierr=0;

  if (vector)
  {
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
    if (vector->mapped) {
      //printf("Unmap in hypre_SeqVectorDestroy\n");
      hypre_SeqVectorUnMapFromDevice(vector);
    }
#endif
    if ( hypre_VectorOwnsData(vector) )
    {
      hypre_TFree(hypre_VectorData(vector), HYPRE_MEMORY_SHARED);
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
//printf("destroying VECTOR DEVICE DATA \n");
      hypre_TFree(hypre_VectorDeviceData(vector), HYPRE_MEMORY_DEVICE);
#endif
    }
    hypre_TFree(vector, HYPRE_MEMORY_HOST);
  }

  return ierr;
}

/**** ==============================
 * hypre_SeqVectorCopyDataCPUtoGPU
 * =================================*/
HYPRE_Int

hypre_SeqVectorCopyDataCPUtoGPU( hypre_Vector *vector )
{
  HYPRE_Int  ierr = 0;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Int  size = hypre_VectorSize(vector);
if (size != 0){ 
  HYPRE_Complex *data, *d_data;

  hypre_SeqVectorPrefetchToDevice(vector);
  data =                hypre_VectorData(vector);
  d_data = hypre_VectorDeviceData(vector);
//  if (!d_data) printf("vector: no gpu data\n");
//  if (!data) printf("vector: no cpu data\n");
 cudaDeviceSynchronize();
  cudaMemcpy ( d_data,data,
      size*sizeof(HYPRE_Complex),
      cudaMemcpyDeviceToDevice );

  cudaDeviceSynchronize();
}
#endif  
return ierr;
}

/**** ==============================
 * hypre_SeqVectorCopyDataGPUtoCPU
 * =================================*/
HYPRE_Int

hypre_SeqVectorCopyDataGPUtoCPU( hypre_Vector *vector )
{
  HYPRE_Int  ierr = 0;
  
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
HYPRE_Int  size = hypre_VectorSize(vector);
  HYPRE_Complex *data, *d_data;

  data =                hypre_VectorData(vector);
  d_data = hypre_VectorDeviceData(vector);

  cudaDeviceSynchronize();
  cudaMemcpy (data,d_data,
      size*sizeof(HYPRE_Complex),
      cudaMemcpyDeviceToDevice );
  cudaDeviceSynchronize();
#endif
  return ierr;
}


/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorInitialize( hypre_Vector *vector )
{
  HYPRE_Int  size = hypre_VectorSize(vector);
  HYPRE_Int  ierr = 0;
  HYPRE_Int  num_vectors = hypre_VectorNumVectors(vector);
  HYPRE_Int  multivec_storage_method = hypre_VectorMultiVecStorageMethod(vector);

  if ( ! hypre_VectorData(vector) ){
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  printf("seq vector, init dev data, size %d num vectors %d \n", size, num_vectors);
if (size!=0)    
hypre_VectorDeviceData(vector) = hypre_CTAlloc(HYPRE_Complex,  num_vectors*size, HYPRE_MEMORY_DEVICE);
else 
hypre_VectorDeviceData(vector) = NULL;
#endif   
    hypre_VectorData(vector) = hypre_CTAlloc(HYPRE_Complex,  num_vectors*size, HYPRE_MEMORY_SHARED);
   // printf("seq vector, init cpu data\n");

  }
  if ( multivec_storage_method == 0 )
  {
    hypre_VectorVectorStride(vector) = size;
    hypre_VectorIndexStride(vector) = 1;
  }
  else if ( multivec_storage_method == 1 )
  {
    hypre_VectorVectorStride(vector) = 1;
    hypre_VectorIndexStride(vector) = num_vectors;
  }
  else
    ++ierr;
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateHRC;
#endif
  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorSetDataOwner( hypre_Vector *vector,
    HYPRE_Int     owns_data   )
{
  HYPRE_Int    ierr=0;

  hypre_VectorOwnsData(vector) = owns_data;

  return ierr;
}

/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

  hypre_Vector *
hypre_SeqVectorRead( char *file_name )
{
  hypre_Vector  *vector;

  FILE    *fp;

  HYPRE_Complex *data;
  HYPRE_Int      size;

  HYPRE_Int      j;

  /*----------------------------------------------------------
   * Read in the data
   *----------------------------------------------------------*/

  fp = fopen(file_name, "r");

  hypre_fscanf(fp, "%d", &size);

  vector = hypre_SeqVectorCreate(size);
  hypre_SeqVectorInitialize(vector);

  data = hypre_VectorData(vector);
  for (j = 0; j < size; j++)
  {
    hypre_fscanf(fp, "%le", &data[j]);
  }

  fclose(fp);

  /* multivector code not written yet >>> */
  hypre_assert( hypre_VectorNumVectors(vector) == 1 );

  return vector;
}




/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorPrint( hypre_Vector *vector,
    char         *file_name )
{
  FILE    *fp;

  HYPRE_Complex *data;
  HYPRE_Int      size, num_vectors, vecstride, idxstride;

  HYPRE_Int      i, j;
  HYPRE_Complex  value;

  HYPRE_Int      ierr = 0;

  num_vectors = hypre_VectorNumVectors(vector);
  vecstride = hypre_VectorVectorStride(vector);
  idxstride = hypre_VectorIndexStride(vector);

  /*----------------------------------------------------------
   * Print in the data
   *----------------------------------------------------------*/

  data = hypre_VectorData(vector);
  size = hypre_VectorSize(vector);

  fp = fopen(file_name, "w");

  if ( hypre_VectorNumVectors(vector) == 1 )
  {
    hypre_fprintf(fp, "%d\n", size);
  }
  else
  {
    hypre_fprintf(fp, "%d vectors of size %d\n", num_vectors, size );
  }

  if ( num_vectors>1 )
  {
    for ( j=0; j<num_vectors; ++j )
    {
      hypre_fprintf(fp, "vector %d\n", j );
      for (i = 0; i < size; i++)
      {
	value = data[ j*vecstride + i*idxstride ];
#ifdef HYPRE_COMPLEX
	hypre_fprintf(fp, "%.14e , %.14e\n",
	    hypre_creal(value), hypre_cimag(value));
#else
	hypre_fprintf(fp, "%.14e\n", value);
#endif
      }
    }
  }
  else
  {
    for (i = 0; i < size; i++)
    {
#ifdef HYPRE_COMPLEX
      hypre_fprintf(fp, "%.14e , %.14e\n",
	  hypre_creal(data[i]), hypre_cimag(data[i]));
#else
      hypre_fprintf(fp, "%.14e\n", data[i]);
#endif
    }
  }

  fclose(fp);

  return ierr;
}

/*SeqVectorInnerProdOneOfMult
 * written by KS. Inner Prod for mult-vectors (stored columnwise) 
 * but we multiply ONLY ONE VECTOR BY ONE VECTOR
 *--------------------------------------------------------------------------*/

HYPRE_Real   hypre_SeqVectorInnerProdOneOfMult( hypre_Vector *x, HYPRE_Int k1,
    hypre_Vector *y, HYPRE_Int k2 ){


  HYPRE_Int         x_size = hypre_VectorSize(x);
  HYPRE_Int         y_size = hypre_VectorSize(y);

  //HYPRE_Int size;
  //printf("INSIDE IP DEPTH 2 x_size %d y_size %d \n", x_size, y_size);
  HYPRE_Int size   = hypre_VectorSize(x);
  //HYPRE_Int num_vectors = hypre_VectorNumVectors(x);
 // HYPRE_Int vecstride = hypre_VectorVectorStride(x);


  HYPRE_Real     result = 0.0;
  //printf("about to multiply HYPRE_USING_GPU %di, k1 = %d k2=%d size = %d \n", HYPRE_USING_GPU, k1, k2, size);

#if defined(HYPRE_USING_GPU)
  //printf("will be using cublas \n");
  static cublasHandle_t handle;
  static HYPRE_Int firstcall=1;
  HYPRE_Complex *x_data = hypre_VectorDeviceData(x);
  HYPRE_Complex *y_data = hypre_VectorDeviceData(y);
  //HYPRE_Complex *x_dataCPU = hypre_VectorData(x);
  //printf("IHA SIZE = %d \n", size);
  //cublasStatus_t stat;
  if (firstcall){
    handle = getCublasHandle();
    firstcall=0;
  }
  cublasDdot(handle, (HYPRE_Int)size,  &x_data[x_size*k1], 1,
      &y_data[y_size*k2], 1,
      &result);
  //printf("gpu result = %f \n", result);
  return result;
#else
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  int i;
  for (i = 0; i < size; i++)
    result += hypre_conj(y_data[k2*size+i]) * x_data[k1*size+i];
  //printf("cpu result = %f \n", result);
  return result;
#endif


}






/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpyOneOfMult
 * written by KS. Axpy for multivectors
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorAxpyOneOfMult( HYPRE_Complex alpha,
    hypre_Vector *x,HYPRE_Int k1,
    hypre_Vector *y, HYPRE_Int k2     )
{


  HYPRE_Int size   = hypre_VectorSize(x);
  //HYPRE_Int num_vectors = hypre_VectorNumVectors(x);
  //HYPRE_Int vecstride = hypre_VectorVectorStride(x);

  HYPRE_Int      ierr = 0;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)

  static cublasHandle_t handle;
  static HYPRE_Int firstcall=1;
  HYPRE_Complex *x_data = hypre_VectorDeviceData(x);
  HYPRE_Complex *y_data = hypre_VectorDeviceData(y);
  //cublasStatus_t stat;
  if (firstcall){
    handle = getCublasHandle();
    firstcall=0;
  }

  cublasDaxpy(handle,(HYPRE_Int)size,&alpha,x_data+size*k1,1,y_data+size*k2,1);

#else
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);

  HYPRE_Int      i;
  ierr = 0;

  size *=hypre_VectorNumVectors(x);

#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif

#if  defined(HYPRE_USING_GPU)
  hypre_SeqVectorPrefetchToDevice(x);
  hypre_SeqVectorPrefetchToDevice(y);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,x_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
  //printf("AXPY OMP \n");
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
    y_data[k2*size+i] += alpha * x_data[k1*size+i];

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateDRC(y);
#endif

#endif
  return ierr;
}



/*
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorSetConstantValues( hypre_Vector *v,
    HYPRE_Complex value )
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) /* CUDA */
  HYPRE_Int      ierr  = 0;
    VecSet(hypre_VectorData(v),hypre_VectorSize(v),value,HYPRE_STREAM(4));
  //printf("GPU+unified \n");

#else /*GPU only,  CPU or OMP 4.5 */
#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU)
 // printf("\nGPU+non unified\n");
  //HYPRE_Complex * hData = hypre_VectorData(v);

  //HYPRE_Int      size        = hypre_VectorSize(v);
if (hypre_VectorSize(v)!=0){
  cudaDeviceSynchronize();
/* for (int i = 0; i < size; i++)
  {
    hData[i] = value;
  }*/
  VecSet(hypre_VectorData(v),hypre_VectorSize(v),value,HYPRE_STREAM(4));
  cudaDeviceSynchronize();

  hypre_SeqVectorPrefetchToDevice(v);
  cudaDeviceSynchronize();
  hypre_SeqVectorCopyDataCPUtoGPU(v);

}
  return 0;
#endif
  return 0;
#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
}

/*** written by KS. Scale only a part of multivectors
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorScaleOneOfMult( HYPRE_Complex alpha,
    hypre_Vector *y, HYPRE_Int k1     )
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

  HYPRE_Int      ierr = 0;

  //HYPRE_Int vecstride = hypre_VectorVectorStride(y);
  HYPRE_Int      size   = hypre_VectorSize(y);

#ifdef HYPRE_USING_GPU
  //        return VecScaleScalarGPUonly(y->d_data,alpha, hypre_VectorSize(y),HYPRE_STREAM(4));

  HYPRE_Complex *y_data = hypre_VectorDeviceData(y);


  static cublasHandle_t handle;
  static HYPRE_Int firstcall=1;
  if (firstcall){
    handle=getCublasHandle();
    firstcall=0;
  }

  //  printf("scaling by %16.16f, vec lenght %d, vec start %d, k1 = %d \n", alpha, size, k1*size, k1 );
  cublasDscal(handle, size,
      &alpha,
      y_data + k1*size, 1);
  cudaDeviceSynchronize();
#else
  HYPRE_Complex *y_data = hypre_VectorData(y);
  size   = hypre_VectorSize(y);

  HYPRE_Int      i;
  size *=hypre_VectorNumVectors(y);
#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToDevice(y);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
    y_data[k1*size + i] *= alpha;
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateDRC(y);
#endif
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
#endif
  return ierr;
}

/* hypre_SeqVectorSetRandomValues
 *
 *     returns vector of values randomly distributed between -1.0 and +1.0
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorSetRandomValues( hypre_Vector *v,
    HYPRE_Int           seed )
{
  HYPRE_Complex *vector_data = hypre_VectorData(v);
  HYPRE_Int      size        = hypre_VectorSize(v);

  HYPRE_Int      i;

  HYPRE_Int      ierr  = 0;
  hypre_SeedRand(seed);

  size *=hypre_VectorNumVectors(v);

  /* RDF: threading this loop may cause problems because of hypre_Rand() */
  for (i = 0; i < size; i++)
    vector_data[i] = 2.0 * hypre_Rand() - 1.0;

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * if size of x is larger than y only the first size_y elements of x are
 * copied to y
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorCopy( hypre_Vector *x,
    hypre_Vector *y )
{

HYPRE_Complex *x_data, *y_data;
HYPRE_Int size, size_y; 
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Int ret =  hypre_SeqVectorCopyDevice(x,y);
  hypre_SeqVectorCopyDataGPUtoCPU(y);
  return ret;
#endif
#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Int ierr = hypre_SeqVectorCopyDevice(x,y);
#else
  x_data = hypre_VectorData(x);
  y_data = hypre_VectorData(y);
  size   = hypre_VectorSize(x);
  size_y = hypre_VectorSize(y);
  HYPRE_Int      i;
  HYPRE_Int      ierr = 0;
#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToDevice(y);
#endif
  if (size > size_y)
  {
    size = size_y;
  }
  size *=hypre_VectorNumVectors(x);
#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,x_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
  {
    y_data[i] = x_data[i];
  }
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateDRC(y);
#endif
#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopyOneOfMult
 * copies data from x(k1*stride + ...)  to y(k2*stride + ...)
 * if size of x is larger than y only the first size_y elements of x are 
 * copied to y
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorCopyOneOfMult( hypre_Vector *x, HYPRE_Int k1,
    hypre_Vector *y, HYPRE_Int k2 )
{


  HYPRE_Int      size   = hypre_VectorSize(x);
  HYPRE_Int      size_y   = hypre_VectorSize(y);
  //printf("sizes %d and ^%d \n", size, size_y);
 HYPRE_Complex *x_dataDevice,*y_dataDevice, *y_dataCPU;
#ifdef HYPRE_USING_GPU
  //      return hypre_SeqVectorCopyDevice(x,y);
  //COPY THE GPU DATA FIRSR

  x_dataDevice = hypre_VectorDeviceData(x);
  y_dataDevice = hypre_VectorDeviceData(y);
  y_dataCPU = hypre_VectorData(y);
//  printf("copying gpu v data to gpu v data, size_y = %d, size_x = %d\n", size_y, size);
  cudaDeviceSynchronize();
  cudaMemcpy (&y_dataDevice[k2*size_y],&x_dataDevice[k1*size],
      size_y*sizeof(HYPRE_Complex),
      cudaMemcpyDeviceToDevice );
  //printf("copying gpu v data to gpu v data -- done\n");
  cudaDeviceSynchronize();
  cudaMemcpy (&y_dataCPU[k2*size_y],&y_dataDevice[k2*size_y],
      size_y*sizeof(HYPRE_Complex),
      cudaMemcpyDeviceToDevice);

  cudaDeviceSynchronize();
return 0;
#else
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      i;

  HYPRE_Int      ierr = 0;
#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);

  else SyncVectorToDevice(y);
#endif


  //printf("copying cpu v data to cpu v data\n");
#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,x_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) 
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  int xstart = k1*size, ystart=k2*size_y;

  for (i = 0; i < size; i++)
    y_data[i+ystart] = x_data[i+xstart];

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

  //printf("copying cpu v data to cpu v data DONE\n");
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD   
  UpdateDRC(y);
#endif
  return ierr;
#endif

}



/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

  hypre_Vector *
hypre_SeqVectorCloneDeep( hypre_Vector *x )
{
  HYPRE_Int      size   = hypre_VectorSize(x);
  HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);
  hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

  hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
  hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
  hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

  hypre_SeqVectorInitialize(y);
  hypre_SeqVectorCopy( x, y );
  // UpdateHRC(y); Done in previous statement
  return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

  hypre_Vector *
hypre_SeqVectorCloneShallow( hypre_Vector *x )
{
  HYPRE_Int      size   = hypre_VectorSize(x);
  HYPRE_Int      num_vectors   = hypre_VectorNumVectors(x);
  hypre_Vector * y = hypre_SeqMultiVectorCreate( size, num_vectors );

  hypre_VectorMultiVecStorageMethod(y) = hypre_VectorMultiVecStorageMethod(x);
  hypre_VectorVectorStride(y) = hypre_VectorVectorStride(x);
  hypre_VectorIndexStride(y) = hypre_VectorIndexStride(x);

  hypre_VectorData(y) = hypre_VectorData(x);
  hypre_SeqVectorSetDataOwner( y, 0 );
  hypre_SeqVectorInitialize(y);

  return y;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorScale( HYPRE_Complex alpha,
    hypre_Vector *y     )
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Int ierr = VecScaleScalar(y->data,alpha, hypre_VectorSize(y), HYPRE_STREAM(4));
#else
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      size   = hypre_VectorSize(y);
  HYPRE_Int      i;
  HYPRE_Int      ierr = 0;

  size *= hypre_VectorNumVectors(y);

#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToDevice(y);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
  {
    y_data[i] *= alpha;
  }
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateDRC(y);
#endif
#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorAxpy( HYPRE_Complex alpha,
    hypre_Vector *x,
    hypre_Vector *y     )
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU) 
  HYPRE_Int ierr = hypre_SeqVectorAxpyDevice(alpha,x,y);
#else
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      size   = hypre_VectorSize(x);
  HYPRE_Int      i;
  HYPRE_Int      ierr = 0;

  size *= hypre_VectorNumVectors(x);

#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif

#ifdef HYPRE_USING_UNIFIED_MEMORY
  hypre_SeqVectorPrefetchToDevice(x);
  hypre_SeqVectorPrefetchToDevice(y);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,x_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
  //printf("AXPY OMP \n");
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
  {
    y_data[i] += alpha * x_data[i];
  }
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateDRC(y);
#endif
#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy8
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorMassAxpy8( HYPRE_Complex *alpha,
    hypre_Vector **x,
    hypre_Vector  *y, HYPRE_Int k)
{
  HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
  HYPRE_Complex  *y_data = hypre_VectorData(y);
  HYPRE_Int       size   = hypre_VectorSize(x[0]);

  HYPRE_Int      i, j, jstart, restk;


  restk = (k-(k/8*8));

  if (k > 7)
  {
    for (j = 0; j < k-7; j += 8)
    {
      jstart = j*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	y_data[i] += alpha[j]*x_data[jstart+i] + alpha[j+1]*x_data[jstart+i+size]
	  + alpha[j+2]*x_data[(j+2)*size+i] + alpha[j+3]*x_data[(j+3)*size+i]
	  + alpha[j+4]*x_data[(j+4)*size+i] + alpha[j+5]*x_data[(j+5)*size+i]
	  + alpha[j+6]*x_data[(j+6)*size+i] + alpha[j+7]*x_data[(j+7)*size+i];
      }
    }
  }
  if (restk == 1)
  {
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-1] * x_data[jstart+i];
    }
  }
  else if (restk == 2)
  {
    jstart = (k-2)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-2] * x_data[jstart+i] + alpha[k-1] * x_data[jstart+size+i];
    }
  }
  else if (restk == 3)
  {
    jstart = (k-3)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-3] * x_data[jstart+i] + alpha[k-2] * x_data[jstart+size+i] + alpha[k-1] * x_data[(k-1)*size+i];
    }
  }
  else if (restk == 4)
  {
    jstart = (k-4)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-4]*x_data[(k-4)*size+i] + alpha[k-3]*x_data[(k-3)*size+i]
	+ alpha[k-2]*x_data[(k-2)*size+i] + alpha[k-1]*x_data[(k-1)*size+i];
    }
  }
  else if (restk == 5)
  {
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += + alpha[k-5]*x_data[(k-5)*size+i] + alpha[k-4]*x_data[(k-4)*size+i]
	+ alpha[k-3]*x_data[(k-3)*size+i] + alpha[k-2]*x_data[(k-2)*size+i]
	+ alpha[k-1]*x_data[(k-1)*size+i];
    }
  }
  else if (restk == 6)
  {
    jstart = (k-6)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-6]*x_data[jstart+i] + alpha[k-5]*x_data[jstart+i+size]
	+ alpha[k-4]*x_data[(k-4)*size+i] + alpha[k-3]*x_data[(k-3)*size+i]
	+ alpha[k-2]*x_data[(k-2)*size+i] + alpha[k-1]*x_data[(k-1)*size+i];
    }
  }
  else if (restk == 7)
  {
    jstart = (k-7)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-7]*x_data[jstart+i] + alpha[k-6]*x_data[jstart+i+size]
	+ alpha[k-5]*x_data[(k-5)*size+i] + alpha[k-4]*x_data[(k-4)*size+i]
	+ alpha[k-3]*x_data[(k-3)*size+i] + alpha[k-2]*x_data[(k-2)*size+i]
	+ alpha[k-1]*x_data[(k-1)*size+i];
    }
  }

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy4
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorMassAxpy4( HYPRE_Complex *alpha,
    hypre_Vector **x,
    hypre_Vector  *y, HYPRE_Int k)
{
  HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
  HYPRE_Complex  *y_data = hypre_VectorData(y);
  HYPRE_Int       size   = hypre_VectorSize(x[0]);

  HYPRE_Int      i, j, jstart, restk;


  restk = (k-(k/4*4));

  if (k > 3)
  {
    for (j = 0; j < k-3; j += 4)
    {
      jstart = j*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	y_data[i] += alpha[j]*x_data[jstart+i] + alpha[j+1]*x_data[jstart+i+size]
	  + alpha[j+2]*x_data[(j+2)*size+i] + alpha[j+3]*x_data[(j+3)*size+i];
      }
    }
  }
  if (restk == 1)
  {
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-1] * x_data[jstart+i];
    }
  }
  else if (restk == 2)
  {
    jstart = (k-2)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-2] * x_data[jstart+i] + alpha[k-1] * x_data[jstart+size+i];
    }
  }
  else if (restk == 3)
  {
    jstart = (k-3)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      y_data[i] += alpha[k-3] * x_data[jstart+i] + alpha[k-2] * x_data[jstart+size+i] + alpha[k-1] * x_data[(k-1)*size+i];
    }
  }

  return hypre_error_flag;
}

/*-----
 * Custom blas operation that roates a block of 2 vectors (REMEBER stored as multivector)
*/

void hypre_SeqVectorGivensRotRight(
     HYPRE_Int k1,
    HYPRE_Int k2,
    hypre_Vector  * q1,
    hypre_Vector  * q2,
    HYPRE_Real  a1, HYPRE_Real a2, HYPRE_Real a3,  HYPRE_Real a4){

#ifdef  HYPRE_USING_GPU

  HYPRE_Complex *q1_dataDevice = hypre_VectorDeviceData(q1);
  HYPRE_Complex *q2_dataDevice = hypre_VectorDeviceData(q2);

  HYPRE_Int      size   = hypre_VectorSize(q1);

  GivensRotRight(size,  k1,k2,
      q1_dataDevice + k1*size,
      q2_dataDevice + k2*size,
      a1,a2,a3,a4);
#else

  HYPRE_Real *q1_data = hypre_VectorData(q1);
  HYPRE_Real *q2_data = hypre_VectorData(q2);


  HYPRE_Int      size   = hypre_VectorSize(y);

  HYPRE_Int      ierr = 0;


  HYPRE_Int i;
HYPRE_Real q1i;
  for (i = 0; i<size; i++){
q1i = q1_data[i];

      q1_data[i] = q1i*a1+q2_data[i]*a2;
      q1_data[i] = q1i*a3+q2_data[i]*a4;
  }

#endif

}


/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpyMult
 * written by KS. Mass Axpy; first argument is a mutivector and the
 * other one is a vector
 *--------------------------------------------------------------------------*/



  void
hypre_SeqVectorMassAxpyMult( HYPRE_Real * alpha,
    hypre_Vector *x, HYPRE_Int k,
    hypre_Vector *y, HYPRE_Int k2)
{
#ifdef  HYPRE_USING_GPU

  HYPRE_Complex *x_dataDevice = hypre_VectorDeviceData(x);
  HYPRE_Complex *y_dataDevice = hypre_VectorDeviceData(y);

  HYPRE_Int      size   = hypre_VectorSize(y);

  MassAxpyGPUonly(size,  k,
      x_dataDevice,
      y_dataDevice+k2*size,
      alpha);
  //      return hypre_SeqVectorMassAxpyDevice(alpha,x,y, k);
#else

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

  HYPRE_Real *y_data = hypre_VectorData(y);
  HYPRE_Real *x_data = hypre_VectorData(x);


  HYPRE_Int      size   = hypre_VectorSize(y);

  HYPRE_Int      ierr = 0;

  //size *=hypre_VectorNumVectors(x);

#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif

  /*#ifdef HYPRE_USE_MANAGED
    hypre_SeqVectorPrefetchToDevice(x);
    hypre_SeqVectorPrefetchToDevice(y);
#endif*/
#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) is_device_ptr(y_data,x_data)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS)
#elif defined(HYPRE_USING_OPENMP)
  //printf("AXPY OMP \n");
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
  int i,j;
  for (j = 0; j<k; j++){

    for (i = 0; i < size; i++){
      y_data[i] += alpha[j] * x_data[j*size + i];
    }
  }

#endif
}


/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProdMult
 * written by KS. Mass Inner Product; first argument is a mutivector and the
 * other one is a vector
 *--------------------------------------------------------------------------*/





void  hypre_SeqVectorMassInnerProdMult( hypre_Vector *x,HYPRE_Int k,
    hypre_Vector *y, HYPRE_Int  k2, HYPRE_Real *result )
{
#ifdef HYPRE_USING_GPU
  //      return hypre_SeqVectorMassInnerProdDevice(x,y, k, result );
  // DONT even try
  //

  HYPRE_Real *x_data = hypre_VectorDeviceData(x);
  HYPRE_Real *y_data = hypre_VectorDeviceData(y);

  HYPRE_Int      size   = hypre_VectorSize(x);

  //HYPRE_Int vecstride = hypre_VectorVectorStride(x);
  //printf("about to GPU ip! k2 = %d \n", k2);
  MassInnerProdGPUonly(y_data+k2*size,
      x_data,
      result,
      k,
      size);
#else
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif
  HYPRE_Real *x_data = hypre_VectorData(x);
  HYPRE_Real *y_data = hypre_VectorData(y);

  HYPRE_Int      i;

  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int  num_vectors = hypre_VectorNumVectors(x);

  HYPRE_Int vecstride = hypre_VectorVectorStride(y);




  for (i=0; i<k; ++i){result[i] = 0.0f;}
  PUSH_RANGE("INNER_PROD",0);

  HYPRE_Int j;

  for (j=0; j<k; ++j){
    for (i = 0; i < size; i++){
      result[j] += hypre_conj(x_data[j*size + i]) * y_data[k2*size +i];
    }
  }
  POP_RANGE;
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
  free(y_data);

#endif
}
//2-vec
//

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProdTwoVectorsMult
 * written by KS. Mass Inner Product; first argument is a mutivector and the
 * other two are vectors
 *--------------------------------------------------------------------------*/





void  hypre_SeqVectorMassInnerProdTwoVectorsMult( hypre_Vector *x,HYPRE_Int k,
    hypre_Vector *y1, HYPRE_Int  k2,  hypre_Vector *y2, HYPRE_Int  k3,HYPRE_Real *result )
{
#ifdef HYPRE_USING_GPU
  //      return hypre_SeqVectorMassInnerProdDevice(x,y, k, result );
  // DONT even try
  //

  HYPRE_Real *x_data = hypre_VectorDeviceData(x);
  HYPRE_Real *y1_data = hypre_VectorDeviceData(y1);
  HYPRE_Real *y2_data = hypre_VectorDeviceData(y2);

  HYPRE_Int      size   = hypre_VectorSize(x);

  //HYPRE_Int vecstride = hypre_VectorVectorStride(x);
  //printf("about to GPU ip! k2 = %d \n", k2);
  MassInnerProdTwoVectorsGPUonly(y1_data+k2*size,y2_data+k3*size,
      x_data,
      result,
      k,
      size);
#else
printf("does not work withoug GPU, finish if youy wish\n");
#endif
}



//version with build-in scaling
//

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProdWithScalingMult
 * written by KS. Mass Inner Product; first argument is a mutivector and the
 * other one is a vector
 *--------------------------------------------------------------------------*/



void  hypre_SeqVectorMassInnerProdWithScalingMult( hypre_Vector *x,HYPRE_Int k,
    hypre_Vector *y, HYPRE_Int  k2,HYPRE_Real * scaleFactors, HYPRE_Real *result )
{
#ifdef HYPRE_USING_GPU
  //      return hypre_SeqVectorMassInnerProdDevice(x,y, k, result );
  // DONT even try
  //

  HYPRE_Real *x_data = hypre_VectorDeviceData(x);
  HYPRE_Real *y_data = hypre_VectorDeviceData(y);
  HYPRE_Int      size   = hypre_VectorSize(x);
  //HYPRE_Int vecstride = hypre_VectorVectorStride(x);
  MassInnerProdWithScalingGPUonly(y_data+k2*size,
      x_data,
      scaleFactors,
      result,
      k,
      size);
#else
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif
#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif


  HYPRE_Real *x_data = hypre_VectorData(x);
  HYPRE_Real *y_data = hypre_VectorData(y);

  HYPRE_Int      i;

  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int  num_vectors = hypre_VectorNumVectors(x);

  HYPRE_Int vecstride = hypre_VectorVectorStride(y);
  for (i=0; i<k; ++i){result[i] = 0.0f;}
  PUSH_RANGE("INNER_PROD",0);

  HYPRE_Int j;

  for (j=0; j<k; ++j){
    for (i = 0; i < size; i++){
      result[j] += hypre_conj(x_data[j*size + i]) * y_data[k2*size +i];
      result[j] *= scaleFactors[j];
    }
  }
  POP_RANGE;
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif
  free(y_data);

#endif
}




/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassAxpy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_SeqVectorMassAxpy( HYPRE_Complex *alpha,
    hypre_Vector **x,
    hypre_Vector  *y, HYPRE_Int k, HYPRE_Int unroll)
{
  HYPRE_Complex  *x_data = hypre_VectorData(x[0]);
  HYPRE_Complex  *y_data = hypre_VectorData(y);
  HYPRE_Int       size   = hypre_VectorSize(x[0]);

  HYPRE_Int      i, j, jstart;

  if (unroll == 8)
  {
    hypre_SeqVectorMassAxpy8(alpha, x, y, k);
    return hypre_error_flag;
  }
  else if (unroll == 4)
  {
    hypre_SeqVectorMassAxpy4(alpha, x, y, k);
    return hypre_error_flag;
  }
  else
  {
    for (j = 0; j < k; j++)
    {
      jstart = j*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	y_data[i] += alpha[j]*x_data[jstart+i];
      }
    }
  }

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

  HYPRE_Real
hypre_SeqVectorInnerProd( hypre_Vector *x,
    hypre_Vector *y )
{
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] -= hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU)
  HYPRE_Real result = hypre_SeqVectorInnerProdDevice(x,y);
#else
#if defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
  if (!x->mapped) hypre_SeqVectorMapToDevice(x);
  else SyncVectorToDevice(x);
  if (!y->mapped) hypre_SeqVectorMapToDevice(y);
  else SyncVectorToHost(y);
#endif

  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i;

  HYPRE_Real     result = 0.0;
  //ASSERT_MANAGED(x_data);
  //ASSERT_MANAGED(y_data);
  PUSH_RANGE("INNER_PROD",0);
  size *=hypre_VectorNumVectors(x);
#if defined(HYPRE_USING_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) reduction(+:result) is_device_ptr(y_data,x_data) map(result)
#elif defined(HYPRE_USING_MAPPED_OPENMP_OFFLOAD)
#pragma omp target teams  distribute  parallel for private(i) num_teams(NUM_TEAMS) thread_limit(NUM_THREADS) reduction(+:result)  map(result)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:result) HYPRE_SMP_SCHEDULE
#endif
  for (i = 0; i < size; i++)
  {
    result += hypre_conj(y_data[i]) * x_data[i];
  }
  POP_RANGE;
#endif /* defined(HYPRE_USING_GPU) && defined(HYPRE_USING_UNIFIED_MEMORY) */

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_BLAS1] += hypre_MPI_Wtime();
#endif

  return result;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProd8
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassInnerProd8( hypre_Vector *x,
    hypre_Vector **y, HYPRE_Int k, HYPRE_Real *result)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y[0]);
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, restk;
  HYPRE_Real res1;
  HYPRE_Real res2;
  HYPRE_Real res3;
  HYPRE_Real res4;
  HYPRE_Real res5;
  HYPRE_Real res6;
  HYPRE_Real res7;
  HYPRE_Real res8;
  HYPRE_Int jstart;
  HYPRE_Int jstart1;
  HYPRE_Int jstart2;
  HYPRE_Int jstart3;
  HYPRE_Int jstart4;
  HYPRE_Int jstart5;
  HYPRE_Int jstart6;
  HYPRE_Int jstart7;

  restk = (k-(k/8*8));

  if (k > 7)
  {
    for (j = 0; j < k-7; j += 8)
    {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      res5 = 0;
      res6 = 0;
      res7 = 0;
      res8 = 0;
      jstart = j*size;
      jstart1 = jstart+size;
      jstart2 = jstart1+size;
      jstart3 = jstart2+size;
      jstart4 = jstart3+size;
      jstart5 = jstart4+size;
      jstart6 = jstart5+size;
      jstart7 = jstart6+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6,res7,res8) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
	res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
	res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
	res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
	res5 += hypre_conj(y_data[jstart4+i]) * x_data[i];
	res6 += hypre_conj(y_data[jstart5+i]) * x_data[i];
	res7 += hypre_conj(y_data[jstart6+i]) * x_data[i];
	res8 += hypre_conj(y_data[jstart7+i]) * x_data[i];
      }
      result[j] = res1;
      result[j+1] = res2;
      result[j+2] = res3;
      result[j+3] = res4;
      result[j+4] = res5;
      result[j+5] = res6;
      result[j+6] = res7;
      result[j+7] = res8;
    }
  }
  if (restk == 1)
  {
    res1 = 0;
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
    }
    result[k-1] = res1;
  }
  else if (restk == 2)
  {
    res1 = 0;
    res2 = 0;
    jstart = (k-2)*size;
    jstart1 = jstart+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
    }
    result[k-2] = res1;
    result[k-1] = res2;
  }
  else if (restk == 3)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    jstart = (k-3)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
    }
    result[k-3] = res1;
    result[k-2] = res2;
    result[k-1] = res3;
  }
  else if (restk == 4)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    res4 = 0;
    jstart = (k-4)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
      res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
    }
    result[k-4] = res1;
    result[k-3] = res2;
    result[k-2] = res3;
    result[k-1] = res4;
  }
  else if (restk == 5)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    res4 = 0;
    res5 = 0;
    jstart = (k-5)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
      res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
      res5 += hypre_conj(y_data[jstart4+i]) * x_data[i];
    }
    result[k-5] = res1;
    result[k-4] = res2;
    result[k-3] = res3;
    result[k-2] = res4;
    result[k-1] = res5;
  }
  else if (restk == 6)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    res4 = 0;
    res5 = 0;
    res6 = 0;
    jstart = (k-6)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
    jstart5 = jstart4+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
      res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
      res5 += hypre_conj(y_data[jstart4+i]) * x_data[i];
      res6 += hypre_conj(y_data[jstart5+i]) * x_data[i];
    }
    result[k-6] = res1;
    result[k-5] = res2;
    result[k-4] = res3;
    result[k-3] = res4;
    result[k-2] = res5;
    result[k-1] = res6;
  }
  else if (restk == 7)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    res4 = 0;
    res5 = 0;
    res6 = 0;
    res7 = 0;
    jstart = (k-7)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
    jstart5 = jstart4+size;
    jstart6 = jstart5+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4,res5,res6,res7) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
      res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
      res5 += hypre_conj(y_data[jstart4+i]) * x_data[i];
      res6 += hypre_conj(y_data[jstart5+i]) * x_data[i];
      res7 += hypre_conj(y_data[jstart6+i]) * x_data[i];
    }
    result[k-7] = res1;
    result[k-6] = res2;
    result[k-5] = res3;
    result[k-4] = res4;
    result[k-3] = res5;
    result[k-2] = res6;
    result[k-1] = res7;
  }


  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassInnerProd4
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassInnerProd4( hypre_Vector *x,
    hypre_Vector **y, HYPRE_Int k, HYPRE_Real *result)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y[0]);
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, restk;
  HYPRE_Real res1;
  HYPRE_Real res2;
  HYPRE_Real res3;
  HYPRE_Real res4;
  HYPRE_Int jstart;
  HYPRE_Int jstart1;
  HYPRE_Int jstart2;
  HYPRE_Int jstart3;

  restk = (k-(k/4*4));

  if (k > 3)
  {
    for (j = 0; j < k-3; j += 4)
    {
      res1 = 0;
      res2 = 0;
      res3 = 0;
      res4 = 0;
      jstart = j*size;
      jstart1 = jstart+size;
      jstart2 = jstart1+size;
      jstart3 = jstart2+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3,res4) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
	res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
	res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
	res4 += hypre_conj(y_data[jstart3+i]) * x_data[i];
      }
      result[j] = res1;
      result[j+1] = res2;
      result[j+2] = res3;
      result[j+3] = res4;
    }
  }
  if (restk == 1)
  {
    res1 = 0;
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
    }
    result[k-1] = res1;
  }
  else if (restk == 2)
  {
    res1 = 0;
    res2 = 0;
    jstart = (k-2)*size;
    jstart1 = jstart+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
    }
    result[k-2] = res1;
    result[k-1] = res2;
  }
  else if (restk == 3)
  {
    res1 = 0;
    res2 = 0;
    res3 = 0;
    jstart = (k-3)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res1,res2,res3) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res1 += hypre_conj(y_data[jstart+i]) * x_data[i];
      res2 += hypre_conj(y_data[jstart1+i]) * x_data[i];
      res3 += hypre_conj(y_data[jstart2+i]) * x_data[i];
    }
    result[k-3] = res1;
    result[k-2] = res2;
    result[k-1] = res3;
  }


  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo8
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassDotpTwo8( hypre_Vector *x, hypre_Vector *y,
    hypre_Vector **z, HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Complex *z_data = hypre_VectorData(z[0]);
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, restk;
  HYPRE_Real res_x1;
  HYPRE_Real res_x2;
  HYPRE_Real res_x3;
  HYPRE_Real res_x4;
  HYPRE_Real res_x5;
  HYPRE_Real res_x6;
  HYPRE_Real res_x7;
  HYPRE_Real res_x8;
  HYPRE_Real res_y1;
  HYPRE_Real res_y2;
  HYPRE_Real res_y3;
  HYPRE_Real res_y4;
  HYPRE_Real res_y5;
  HYPRE_Real res_y6;
  HYPRE_Real res_y7;
  HYPRE_Real res_y8;
  HYPRE_Int jstart;
  HYPRE_Int jstart1;
  HYPRE_Int jstart2;
  HYPRE_Int jstart3;
  HYPRE_Int jstart4;
  HYPRE_Int jstart5;
  HYPRE_Int jstart6;
  HYPRE_Int jstart7;

  restk = (k-(k/8*8));

  if (k > 7)
  {
    for (j = 0; j < k-7; j += 8)
    {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_x5 = 0;
      res_x6 = 0;
      res_x7 = 0;
      res_x8 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      res_y5 = 0;
      res_y6 = 0;
      res_y7 = 0;
      res_y8 = 0;
      jstart = j*size;
      jstart1 = jstart+size;
      jstart2 = jstart1+size;
      jstart3 = jstart2+size;
      jstart4 = jstart3+size;
      jstart5 = jstart4+size;
      jstart6 = jstart5+size;
      jstart7 = jstart6+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_x7,res_x8,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6,res_y7,res_y8) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
	res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
	res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
	res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
	res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
	res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
	res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
	res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
	res_x5 += hypre_conj(z_data[jstart4+i]) * x_data[i];
	res_y5 += hypre_conj(z_data[jstart4+i]) * y_data[i];
	res_x6 += hypre_conj(z_data[jstart5+i]) * x_data[i];
	res_y6 += hypre_conj(z_data[jstart5+i]) * y_data[i];
	res_x7 += hypre_conj(z_data[jstart6+i]) * x_data[i];
	res_y7 += hypre_conj(z_data[jstart6+i]) * y_data[i];
	res_x8 += hypre_conj(z_data[jstart7+i]) * x_data[i];
	res_y8 += hypre_conj(z_data[jstart7+i]) * y_data[i];
      }
      result_x[j] = res_x1;
      result_x[j+1] = res_x2;
      result_x[j+2] = res_x3;
      result_x[j+3] = res_x4;
      result_x[j+4] = res_x5;
      result_x[j+5] = res_x6;
      result_x[j+6] = res_x7;
      result_x[j+7] = res_x8;
      result_y[j] = res_y1;
      result_y[j+1] = res_y2;
      result_y[j+2] = res_y3;
      result_y[j+3] = res_y4;
      result_y[j+4] = res_y5;
      result_y[j+5] = res_y6;
      result_y[j+6] = res_y7;
      result_y[j+7] = res_y8;
    }
  }
  if (restk == 1)
  {
    res_x1 = 0;
    res_y1 = 0;
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_y1) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
    }
    result_x[k-1] = res_x1;
    result_y[k-1] = res_y1;
  }
  else if (restk == 2)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_y1 = 0;
    res_y2 = 0;
    jstart = (k-2)*size;
    jstart1 = jstart+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_y1,res_y2) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
    }
    result_x[k-2] = res_x1;
    result_x[k-1] = res_x2;
    result_y[k-2] = res_y1;
    result_y[k-1] = res_y2;
  }
  else if (restk == 3)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    jstart = (k-3)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_y1,res_y2,res_y3) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
    }
    result_x[k-3] = res_x1;
    result_x[k-2] = res_x2;
    result_x[k-1] = res_x3;
    result_y[k-3] = res_y1;
    result_y[k-2] = res_y2;
    result_y[k-1] = res_y3;
  }
  else if (restk == 4)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_x4 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    res_y4 = 0;
    jstart = (k-4)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_y1,res_y2,res_y3,res_y4) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
      res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
      res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
    }
    result_x[k-4] = res_x1;
    result_x[k-3] = res_x2;
    result_x[k-2] = res_x3;
    result_x[k-1] = res_x4;
    result_y[k-4] = res_y1;
    result_y[k-3] = res_y2;
    result_y[k-2] = res_y3;
    result_y[k-1] = res_y4;
  }
  else if (restk == 5)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_x4 = 0;
    res_x5 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    res_y4 = 0;
    res_y5 = 0;
    jstart = (k-5)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_y1,res_y2,res_y3,res_y4,res_y5) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
      res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
      res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
      res_x5 += hypre_conj(z_data[jstart4+i]) * x_data[i];
      res_y5 += hypre_conj(z_data[jstart4+i]) * y_data[i];
    }
    result_x[k-5] = res_x1;
    result_x[k-4] = res_x2;
    result_x[k-3] = res_x3;
    result_x[k-2] = res_x4;
    result_x[k-1] = res_x5;
    result_y[k-5] = res_y1;
    result_y[k-4] = res_y2;
    result_y[k-3] = res_y3;
    result_y[k-2] = res_y4;
    result_y[k-1] = res_y5;
  }
  else if (restk == 6)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_x4 = 0;
    res_x5 = 0;
    res_x6 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    res_y4 = 0;
    res_y5 = 0;
    res_y6 = 0;
    jstart = (k-6)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
    jstart5 = jstart4+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
      res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
      res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
      res_x5 += hypre_conj(z_data[jstart4+i]) * x_data[i];
      res_y5 += hypre_conj(z_data[jstart4+i]) * y_data[i];
      res_x6 += hypre_conj(z_data[jstart5+i]) * x_data[i];
      res_y6 += hypre_conj(z_data[jstart5+i]) * y_data[i];
    }
    result_x[k-6] = res_x1;
    result_x[k-5] = res_x2;
    result_x[k-4] = res_x3;
    result_x[k-3] = res_x4;
    result_x[k-2] = res_x5;
    result_x[k-1] = res_x6;
    result_y[k-6] = res_y1;
    result_y[k-5] = res_y2;
    result_y[k-4] = res_y3;
    result_y[k-3] = res_y4;
    result_y[k-2] = res_y5;
    result_y[k-1] = res_y6;
  }
  else if (restk == 7)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_x4 = 0;
    res_x5 = 0;
    res_x6 = 0;
    res_x7 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    res_y4 = 0;
    res_y5 = 0;
    res_y6 = 0;
    res_y7 = 0;
    jstart = (k-7)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
    jstart3 = jstart2+size;
    jstart4 = jstart3+size;
    jstart5 = jstart4+size;
    jstart6 = jstart5+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_x5,res_x6,res_x7,res_y1,res_y2,res_y3,res_y4,res_y5,res_y6,res_y7) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
      res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
      res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
      res_x5 += hypre_conj(z_data[jstart4+i]) * x_data[i];
      res_y5 += hypre_conj(z_data[jstart4+i]) * y_data[i];
      res_x6 += hypre_conj(z_data[jstart5+i]) * x_data[i];
      res_y6 += hypre_conj(z_data[jstart5+i]) * y_data[i];
      res_x7 += hypre_conj(z_data[jstart6+i]) * x_data[i];
      res_y7 += hypre_conj(z_data[jstart6+i]) * y_data[i];
    }
    result_x[k-7] = res_x1;
    result_x[k-6] = res_x2;
    result_x[k-5] = res_x3;
    result_x[k-4] = res_x4;
    result_x[k-3] = res_x5;
    result_x[k-2] = res_x6;
    result_x[k-1] = res_x7;
    result_y[k-7] = res_y1;
    result_y[k-6] = res_y2;
    result_y[k-5] = res_y3;
    result_y[k-4] = res_y4;
    result_y[k-3] = res_y5;
    result_y[k-2] = res_y6;
    result_y[k-1] = res_y7;
  }


  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo4
 *--------------------------------------------------------------------------*/
HYPRE_Int hypre_SeqVectorMassDotpTwo4( hypre_Vector *x, hypre_Vector *y,
    hypre_Vector **z, HYPRE_Int k, HYPRE_Real *result_x, HYPRE_Real *result_y)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Complex *z_data = hypre_VectorData(z[0]);
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, restk;
  HYPRE_Real res_x1;
  HYPRE_Real res_x2;
  HYPRE_Real res_x3;
  HYPRE_Real res_x4;
  HYPRE_Real res_y1;
  HYPRE_Real res_y2;
  HYPRE_Real res_y3;
  HYPRE_Real res_y4;
  HYPRE_Int jstart;
  HYPRE_Int jstart1;
  HYPRE_Int jstart2;
  HYPRE_Int jstart3;

  restk = (k-(k/4*4));

  if (k > 3)
  {
    for (j = 0; j < k-3; j += 4)
    {
      res_x1 = 0;
      res_x2 = 0;
      res_x3 = 0;
      res_x4 = 0;
      res_y1 = 0;
      res_y2 = 0;
      res_y3 = 0;
      res_y4 = 0;
      jstart = j*size;
      jstart1 = jstart+size;
      jstart2 = jstart1+size;
      jstart3 = jstart2+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_x4,res_y1,res_y2,res_y3,res_y4) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
	res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
	res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
	res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
	res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
	res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
	res_x4 += hypre_conj(z_data[jstart3+i]) * x_data[i];
	res_y4 += hypre_conj(z_data[jstart3+i]) * y_data[i];
      }
      result_x[j] = res_x1;
      result_x[j+1] = res_x2;
      result_x[j+2] = res_x3;
      result_x[j+3] = res_x4;
      result_y[j] = res_y1;
      result_y[j+1] = res_y2;
      result_y[j+2] = res_y3;
      result_y[j+3] = res_y4;
    }
  }
  if (restk == 1)
  {
    res_x1 = 0;
    res_y1 = 0;
    jstart = (k-1)*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_y1) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
    }
    result_x[k-1] = res_x1;
    result_y[k-1] = res_y1;
  }
  else if (restk == 2)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_y1 = 0;
    res_y2 = 0;
    jstart = (k-2)*size;
    jstart1 = jstart+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_y1,res_y2) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
    }
    result_x[k-2] = res_x1;
    result_x[k-1] = res_x2;
    result_y[k-2] = res_y1;
    result_y[k-1] = res_y2;
  }
  else if (restk == 3)
  {
    res_x1 = 0;
    res_x2 = 0;
    res_x3 = 0;
    res_y1 = 0;
    res_y2 = 0;
    res_y3 = 0;
    jstart = (k-3)*size;
    jstart1 = jstart+size;
    jstart2 = jstart1+size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x1,res_x2,res_x3,res_y1,res_y2,res_y3) HYPRE_SMP_SCHEDULE
#endif
    for (i = 0; i < size; i++)
    {
      res_x1 += hypre_conj(z_data[jstart+i]) * x_data[i];
      res_y1 += hypre_conj(z_data[jstart+i]) * y_data[i];
      res_x2 += hypre_conj(z_data[jstart1+i]) * x_data[i];
      res_y2 += hypre_conj(z_data[jstart1+i]) * y_data[i];
      res_x3 += hypre_conj(z_data[jstart2+i]) * x_data[i];
      res_y3 += hypre_conj(z_data[jstart2+i]) * y_data[i];
    }
    result_x[k-3] = res_x1;
    result_x[k-2] = res_x2;
    result_x[k-1] = res_x3;
    result_y[k-3] = res_y1;
    result_y[k-2] = res_y2;
    result_y[k-1] = res_y3;
  }


  return hypre_error_flag;
}

HYPRE_Int hypre_SeqVectorMassInnerProd( hypre_Vector *x,
    hypre_Vector **y, HYPRE_Int k, HYPRE_Int unroll, HYPRE_Real *result)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y[0]);
  HYPRE_Real res;
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, jstart;

  if (unroll == 8)
  {
    hypre_SeqVectorMassInnerProd8(x,y,k,result);
    return hypre_error_flag;
  }
  else if (unroll == 4)
  {
    hypre_SeqVectorMassInnerProd4(x,y,k,result);
    return hypre_error_flag;
  }
  else
  {
    for (j = 0; j < k; j++)
    {
      res = 0;
      jstart = j*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res += hypre_conj(y_data[jstart+i]) * x_data[i];
      }
      result[j] = res;
    }
  }

  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SeqVectorMassDotpTwo
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_SeqVectorMassDotpTwo( hypre_Vector *x, hypre_Vector *y,
    hypre_Vector **z, HYPRE_Int k,  HYPRE_Int unroll,
    HYPRE_Real *result_x, HYPRE_Real *result_y)
{
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Complex *z_data = hypre_VectorData(z[0]);
  HYPRE_Real res_x, res_y;
  HYPRE_Int      size   = hypre_VectorSize(x);

  HYPRE_Int      i, j, jstart;

  if (unroll == 8)
  {
    hypre_SeqVectorMassDotpTwo8(x,y,z,k,result_x,result_y);
    return hypre_error_flag;
  }
  else if (unroll == 4)
  {
    hypre_SeqVectorMassDotpTwo4(x,y,z,k,result_x,result_y);
    return hypre_error_flag;
  }
  else
  {
    for (j = 0; j < k; j++)
    {
      res_x = result_x[j];
      res_y = result_y[j];
      jstart = j*size;
#if defined(HYPRE_USING_OPENMP)
#pragma omp parallel for private(i) reduction(+:res_x,res_y) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < size; i++)
      {
	res_x += hypre_conj(z_data[jstart+i]) * x_data[i];
	res_y += hypre_conj(z_data[jstart+i]) * y_data[i];
      }
      result_x[j] = res_x;
      result_y[j] = res_y;
    }
  }
  return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_VectorSumElts:
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

HYPRE_Complex hypre_VectorSumElts( hypre_Vector *vector )
{
  HYPRE_Complex  sum = 0;
  HYPRE_Complex *data = hypre_VectorData( vector );
  HYPRE_Int      size = hypre_VectorSize( vector );
  HYPRE_Int      i;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
  for ( i=0; i<size; ++i ) sum += data[i];

  return sum;
}

#ifdef HYPRE_USING_GPU
/* Sums of the absolute value of the elements for comparison to cublas device side routine */
HYPRE_Complex hypre_VectorSumAbsElts( hypre_Vector *vector )
{
  HYPRE_Complex  sum = 0;
  HYPRE_Complex *data = hypre_VectorData( vector );
  HYPRE_Int      size = hypre_VectorSize( vector );
  HYPRE_Int      i;

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) reduction(+:sum) HYPRE_SMP_SCHEDULE
#endif
  for ( i=0; i<size; ++i ) sum += fabs(data[i]);

  return sum;
}

  HYPRE_Int
hypre_SeqVectorCopyDevice( hypre_Vector *x,
    hypre_Vector *y )
{

  HYPRE_Int      size   = hypre_VectorSize(x);
  HYPRE_Int      size_y   = hypre_VectorSize(y);
  /* HYPRE_Int      i; */
  HYPRE_Int      ierr = 0;

  if (size > size_y) size = size_y;
  size *=hypre_VectorNumVectors(x);

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)

  HYPRE_Complex *x_data = hypre_VectorDeviceData(x);
  HYPRE_Complex *y_data = hypre_VectorDeviceData(y);

  cudaDeviceSynchronize();
  cudaMemcpy ( y_data,x_data,
      size*sizeof(HYPRE_Complex),
      cudaMemcpyDeviceToDevice );
  cudaDeviceSynchronize();
#else
  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  PUSH_RANGE_PAYLOAD("VECCOPYDEVICE",2,size);
  hypre_SeqVectorPrefetchToDevice(x);
  hypre_SeqVectorPrefetchToDevice(y);
#ifdef HYPRE_USING_GPU
  VecCopy(y_data,x_data,size,HYPRE_STREAM(4));
#endif
  //cudaStreamSynchronize(HYPRE_STREAM(4));
  POP_RANGE;
#endif
  return ierr;
}

  HYPRE_Int
hypre_SeqVectorAxpyDevice( HYPRE_Complex alpha,
    hypre_Vector *x,
    hypre_Vector *y      )
{

  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      size   = hypre_VectorSize(x);
  /* HYPRE_Int      i; */
  HYPRE_Int      ierr = 0;
  /* cublasStatus_t stat; */
  size *=hypre_VectorNumVectors(x);

  PUSH_RANGE_PAYLOAD("DEVAXPY",0,hypre_VectorSize(x));
  hypre_SeqVectorPrefetchToDevice(x);
  hypre_SeqVectorPrefetchToDevice(y);
  static cublasHandle_t handle;
  static HYPRE_Int firstcall=1;
  if (firstcall){
    handle=getCublasHandle();
    firstcall=0;
  }
  cublasErrchk(cublasDaxpy(handle,(HYPRE_Int)size,&alpha,x_data,1,y_data,1));
//  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
  POP_RANGE;
  return ierr;
}

HYPRE_Real   hypre_SeqVectorInnerProdDevice( hypre_Vector *x,
    hypre_Vector *y )
{
  PUSH_RANGE_PAYLOAD("DEVDOT",4,hypre_VectorSize(x));
  static cublasHandle_t handle;
  static HYPRE_Int firstcall=1;

  HYPRE_Complex *x_data = hypre_VectorData(x);
  HYPRE_Complex *y_data = hypre_VectorData(y);
  HYPRE_Int      size   = hypre_VectorSize(x);
  /* HYPRE_Int      i; */
  HYPRE_Real     result = 0.0;
  /* cublasStatus_t stat; */
  if (firstcall){
    handle = getCublasHandle();
    firstcall=0;
  }
  PUSH_RANGE_PAYLOAD("DEVDOT-PRFETCH",5,hypre_VectorSize(x));
  //hypre_SeqVectorPrefetchToDevice(x);
  //hypre_SeqVectorPrefetchToDevice(y);
  POP_RANGE;
  PUSH_RANGE_PAYLOAD("DEVDOT-ACTUAL",0,hypre_VectorSize(x));
  /*stat=*/cublasDdot(handle, (HYPRE_Int)size,
      x_data, 1,
      y_data, 1,
      &result);
//  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
  POP_RANGE;
  POP_RANGE;
  return result;

}
void hypre_SeqVectorPrefetchToDevice(hypre_Vector *x){
  if (hypre_VectorSize(x)==0) return;
#if defined(TRACK_MEMORY_ALLOCATIONS)
  ASSERT_MANAGED(hypre_VectorData(x));
#endif
  //PrintPointerAttributes(hypre_VectorData(x));
  PUSH_RANGE("hypre_SeqVectorPrefetchToDevice",0);
  hypre_CheckErrorDevice(cudaMemPrefetchAsync(hypre_VectorData(x),hypre_VectorSize(x)*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4)));
  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
  POP_RANGE;
}
void hypre_SeqVectorPrefetchToHost(hypre_Vector *x){
  if (hypre_VectorSize(x)==0) return;
  PUSH_RANGE("hypre_SeqVectorPrefetchToHost",0);

cudaError_t err = cudaPeekAtLastError();
printf("before prefetching, error is %s \n", cudaGetErrorName(err));
  hypre_CheckErrorDevice(cudaMemPrefetchAsync(hypre_VectorData(x),hypre_VectorSize(x)*sizeof(HYPRE_Complex),cudaCpuDeviceId,HYPRE_STREAM(4)));
 err = cudaPeekAtLastError();
printf("after prefetching, error is %s \n", cudaGetErrorName(err));
  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
 err = cudaPeekAtLastError();
printf("after synch, error is %s \n", cudaGetErrorName(err));
  POP_RANGE;
}
void hypre_SeqVectorPrefetchToDeviceInStream(hypre_Vector *x, HYPRE_Int index){
  if (hypre_VectorSize(x)==0) return;
#if defined(TRACK_MEMORY_ALLOCATIONS)
  ASSERT_MANAGED(hypre_VectorData(x));
#endif
  PUSH_RANGE("hypre_SeqVectorPrefetchToDevice",0);
  hypre_CheckErrorDevice(cudaMemPrefetchAsync(hypre_VectorData(x),hypre_VectorSize(x)*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(index)));
  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(index)));
  POP_RANGE;
}
hypre_int hypre_SeqVectorIsManaged(hypre_Vector *x){
  return pointerIsManaged((void*)hypre_VectorData(x));
}
#endif




#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD

void hypre_SeqVectorMapToDevice(hypre_Vector *x){
  if (x==NULL) return;
  if (x->size>0){
    //#pragma omp target enter data map(to:x[0:0])
#pragma omp target enter data map(to:x->data[0:x->size])
    x->mapped=1;
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
    SetDRC(x);
#endif
  }
}
void hypre_SeqVectorMapToDevicePrint(hypre_Vector *x){
  printf("SVmap %p [%p,%p] %d Size = %d ",x,x->data,x->data+x->size,x->mapped,x->size);
  if (x->size>0){
    //#pragma omp target enter data map(to:x[0:0])
#pragma omp target enter data map(to:x->data[0:x->size])
    x->mapped=1;
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
    SetDRC(x);
#endif
  }
  printf("...Done\n");
}

void hypre_SeqVectorUnMapFromDevice(hypre_Vector *x){
  //printf("map %p [%p,%p] %d Size = %d\n",x,x->data,x->data+x->size,x->mapped,x->size);
  //#pragma omp target exit data map(from:x[0:0])
#pragma omp target exit data map(from:x->data[0:x->size])
  x->mapped=0;
}
void hypre_SeqVectorUpdateDevice(hypre_Vector *x){
  if (x==NULL) return;
#pragma omp target update to(x->data[0:x->size])
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  SetDRC(x);
#endif
}

void hypre_SeqVectorUpdateHost(hypre_Vector *x){
  if (x==NULL) return;
#pragma omp target update from(x->data[0:x->size])
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  SetHRC(x);
#endif
}
void printRC(hypre_Vector *x,char *id){
  printf("%p At %s HRC = %d , DRC = %d \n",x,id,x->hrc,x->drc);
}
#endif



