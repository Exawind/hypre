/*BHEADER**********************************************************************
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
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_parcsr_mv.h"
#include <assert.h>
//#ifdef HYPRE_USING_GPU
//extern "C"
//{
//void PackOnDevice(HYPRE_Complex *send_data,HYPRE_Complex *x_local_data, HYPRE_Int *send_map, HYPRE_Int begin,HYPRE_Int end,cudaStream_t s);
//}
//#endif

  HYPRE_Int
hypre_ParCSRMatrixMatvecMult( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Int k1,
    HYPRE_Complex       beta,
    hypre_ParVector    *y,
    HYPRE_Int k2 )
{

  return hypre_ParCSRMatrixMatvecMultOutOfPlace(alpha, A, x,k1, beta, y,k2, y,k2);
}



/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec
 *--------------------------------------------------------------------------*/
// y = alpha*A*x + beta*b
  HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlace( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *b,
    hypre_ParVector    *y )
{


  hypre_ParCSRCommHandle **comm_handle;
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
  hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);
  hypre_Vector      *x_local  = hypre_ParVectorLocalVector(x);
  hypre_Vector      *b_local  = hypre_ParVectorLocalVector(b);
  hypre_Vector      *y_local  = hypre_ParVectorLocalVector(y);
  HYPRE_Int          num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
  HYPRE_Int          num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

  hypre_Vector      *x_tmp;
  HYPRE_Int          x_size = hypre_ParVectorGlobalSize(x);
  HYPRE_Int          b_size = hypre_ParVectorGlobalSize(b);
  HYPRE_Int          y_size = hypre_ParVectorGlobalSize(y);
  HYPRE_Int          num_vectors = hypre_VectorNumVectors(x_local);
  HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_Int          ierr = 0;
  HYPRE_Int          num_sends, i, j, jv, index, start;

  HYPRE_Int          vecstride = hypre_VectorVectorStride( x_local );
  HYPRE_Int          idxstride = hypre_VectorIndexStride( x_local );

  HYPRE_Complex     *x_tmp_data, **x_buf_data;
  HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);

  /*---------------------------------------------------------------------
   *  Check for size compatibility.  ParMatvec returns ierr = 11 if
   *  length of X doesn't equal the number of columns of A,
   *  ierr = 12 if the length of Y doesn't equal the number of rows
   *  of A, and ierr = 13 if both are true.
   *
   *  Because temporary vectors are often used in ParMatvec, none of
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/
  PUSH_RANGE_PAYLOAD("PAR_CSR_MATVEC",5,x_size);
  hypre_assert( idxstride>0 );

  if (num_cols != x_size)
    ierr = 11;

  if (num_rows != y_size || num_rows != b_size)
    ierr = 12;

  if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
    ierr = 13;

  hypre_assert( hypre_VectorNumVectors(b_local)==num_vectors );
  hypre_assert( hypre_VectorNumVectors(y_local)==num_vectors );

  if ( num_vectors==1 )
    x_tmp = hypre_SeqVectorCreate( num_cols_offd );
  else
  {
    hypre_assert( num_vectors>1 );
    x_tmp = hypre_SeqMultiVectorCreate( num_cols_offd, num_vectors );
  }

  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings
   *--------------------------------------------------------------------*/
  if (!comm_pkg)
  {
    hypre_MatvecCommPkgCreate(A);
    comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_PACK",3);
  HYPRE_Int use_persistent_comm = 0;
#ifdef HYPRE_USING_PERSISTENT_COMM
  use_persistent_comm = num_vectors == 1;
  // JSP TODO: we can use persistent communication for multi-vectors,
  // but then we need different communication handles for different
  // num_vectors.
  hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

  if ( use_persistent_comm )
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    PUSH_RANGE("PERCOMM1",0);
    persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);

    HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    hypre_assert(num_cols_offd == hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs));

    hypre_VectorData(x_tmp) = (HYPRE_Complex *)persistent_comm_handle->recv_data;
    hypre_SeqVectorSetDataOwner(x_tmp, 0);
    POP_RANGE;
#endif
  }
  else
  {
    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*, num_vectors, HYPRE_MEMORY_HOST);
  }
  hypre_SeqVectorInitialize(x_tmp);
  x_tmp_data = hypre_VectorData(x_tmp);
  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  if (!use_persistent_comm)
  {
    x_buf_data = hypre_CTAlloc( HYPRE_Complex*,  num_vectors , HYPRE_MEMORY_HOST);
    for ( jv=0; jv<num_vectors; ++jv )
      x_buf_data[jv] = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	  (comm_pkg,  num_sends), HYPRE_MEMORY_SHARED);
  }



  if ( num_vectors==1 )
  {
    HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
    HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
#if defined(HYPRE_USING_GPU) 
    PUSH_RANGE("PERCOMM2DEVICE",4);
#ifdef HYPRE_USING_PERSISTENT_COMM
    PackOnDevice((HYPRE_Complex*)persistent_comm_handle->send_data,x_local_data,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),begin,end,HYPRE_STREAM(4));
    //PrintPointerAttributes(persistent_comm_handle->send_data);
#else
#if defined(DEBUG_PACK_ON_DEVICE)
    printf("debug pack on devive 1\n");  
    //  hypre_CheckErrorDevice(cudaPeekAtLastError());
    //  hypre_CheckErrorDevice(cudaDeviceSynchronize());
    ASSERT_MANAGED(x_buf_data[0]);
    ASSERT_MANAGED(x_local_data);
    ASSERT_MANAGED(hypre_ParCSRCommPkgSendMapElmts(comm_pkg));
#endif
    PackOnDevice((HYPRE_Complex*)x_buf_data[0],x_local_data,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),begin,end,HYPRE_STREAM(4));
#if defined(DEBUG_PACK_ON_DEVICE)
    printf("debug pack on devive 2\n");  
    hypre_CheckErrorDevice(cudaPeekAtLastError());
    hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif
#endif
    POP_RANGE;
    SetAsyncMode(1);
    hypre_CheckErrorDevice(cudaDeviceSynchronize());
    hypre_CSRMatrixMatvecOutOfPlace( alpha, diag, x_local, beta, b_local, y_local, 0);
    //hypre_SeqVectorUpdateHost(y_local);
    //hypre_SeqVectorUpdateHost(x_local);
    //hypre_SeqVectorUpdateHost(b_local);
    SetAsyncMode(0);
#else
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
    PUSH_RANGE("MPI_PACK_OMP",4);
    SyncVectorToHost(x_local);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD_NOT_USED)
    HYPRE_Int num_threads=64;
    HYPRE_Int num_teams = (end-begin+(end-begin)%num_threads)/num_threads;
    HYPRE_Int *local_send_map_elmts = comm_pkg->send_map_elmts;
    //printf("USING OFFLOADED PACKING OF BUFER\n");
#pragma omp target teams  distribute  parallel for private(i) num_teams(num_teams) thread_limit(num_threads) is_device_ptr(x_local_data,x_buf_data,comm_pkg,local_send_map_elmts)
#elif defined(HYPRE_USING_OPENMP)
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
    for (i = begin; i < end; i++)
    {
#ifdef HYPRE_USING_PERSISTENT_COMM
      ((HYPRE_Complex *)persistent_comm_handle->send_data)[i - begin]
#else
	x_buf_data[0][i - begin]
#endif
	= x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
    }
    POP_RANGE; // "MPI_PACK_OMP"
#endif
  }
  else
    for ( jv=0; jv<num_vectors; ++jv )
    {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	  x_buf_data[jv][index++]
	    = x_local_data[
	    jv*vecstride +
	    idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ];
      }
    }

  hypre_assert( idxstride==1 );
  /* ... The assert is because the following loop only works for 'column'
     storage of a multivector. This needs to be fixed to work more generally,
     at least for 'row' storage. This in turn, means either change CommPkg so
     num_sends is no.zones*no.vectors (not no.zones) or, less dangerously, put
     a stride in the logic of CommHandleCreate (stride either from a new arg or
     a new variable inside CommPkg).  Or put the num_vector iteration inside
     CommHandleCreate (perhaps a new multivector variant of it).
     */
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
  POP_RANGE;
  PUSH_RANGE("MPI_HALO_EXC_SEND",4);
  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle);
#endif
  }
  else
  {
    for ( jv=0; jv<num_vectors; ++jv )
    {
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
	( 1, comm_pkg, x_buf_data[jv], &(x_tmp_data[jv*num_cols_offd]) );
    }
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
  POP_RANGE;
#if !defined(HYPRE_USING_GPU) 
  hypre_CSRMatrixMatvecOutOfPlace( alpha, diag, x_local, beta, b_local, y_local, 0);
#endif
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_HALO_EXC_RECV",6);
  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle);
#endif
  }
  else
  {
    for ( jv=0; jv<num_vectors; ++jv )
    {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
    }
    hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);
  }
  POP_RANGE;
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

  //hypre_SeqVectorUpdateDevice(x_tmp);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateHRC(x_tmp);
#endif
  if (num_cols_offd) hypre_CSRMatrixMatvec( alpha, offd, x_tmp, 1.0, y_local);
  //if (num_cols_offd) hypre_SeqVectorUpdateHost(y_local);
  //hypre_SeqVectorUpdateHost(x_tmp);
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_UNPACK",5);
  hypre_SeqVectorDestroy(x_tmp);
  x_tmp = NULL;
  if (!use_persistent_comm)
  {
    for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(x_buf_data[jv], HYPRE_MEMORY_SHARED);
    hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif
  POP_RANGE;
#if defined(HYPRE_USING_GPU) 
  //printf("stream sybch \n");  
  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
#endif
  POP_RANGE; // PAR_CSR
  return ierr;
}

//version for multivectors KS
//





  HYPRE_Int
hypre_ParCSRMatrixMatvec( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *y )
{
  return hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y);
}

#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  HYPRE_Int
hypre_ParCSRMatrixMatvec3( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *y )
{
  HYPRE_Int rval=hypre_ParCSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y);
  hypre_SeqVectorUpdateHost(y->local_vector);
}
  HYPRE_Int
hypre_ParCSRMatrixMatvecOutOfPlace3( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *b,
    hypre_ParVector    *y )
{
  hypre_ParCSRMatrixMatvecOutOfPlace(alpha,A,x,beta,b,y);
  hypre_SeqVectorUpdateHost(y->local_vector);
}
#endif
/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParCSRMatrixMatvecT( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *y )
{
  hypre_ParCSRCommHandle **comm_handle;
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_CSRMatrix     *diag = hypre_ParCSRMatrixDiag(A);
  hypre_CSRMatrix     *offd = hypre_ParCSRMatrixOffd(A);
  hypre_Vector        *x_local = hypre_ParVectorLocalVector(x);
  hypre_Vector        *y_local = hypre_ParVectorLocalVector(y);
  hypre_Vector        *y_tmp;
  HYPRE_Int            vecstride = hypre_VectorVectorStride( y_local );
  HYPRE_Int            idxstride = hypre_VectorIndexStride( y_local );
  HYPRE_Complex       *y_tmp_data, **y_buf_data;
  HYPRE_Complex       *y_local_data = hypre_VectorData(y_local);

  HYPRE_Int         num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
  HYPRE_Int         num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
  HYPRE_Int         num_cols_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_Int         x_size = hypre_ParVectorGlobalSize(x);
  HYPRE_Int         y_size = hypre_ParVectorGlobalSize(y);
  HYPRE_Int         num_vectors = hypre_VectorNumVectors(y_local);

  HYPRE_Int         i, j, jv, index, start, num_sends;

  HYPRE_Int         ierr  = 0;

  if (y==NULL) {
    printf("NULLY %p\b", (void*) y);
    return 1;
  }
  /*---------------------------------------------------------------------
   *  Check for size compatibility.  MatvecT returns ierr = 1 if
   *  length of X doesn't equal the number of rows of A,
   *  ierr = 2 if the length of Y doesn't equal the number of
   *  columns of A, and ierr = 3 if both are true.
   *
   *  Because temporary vectors are often used in MatvecT, none of
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/

  if (num_rows != x_size)
    ierr = 1;

  if (num_cols != y_size)
    ierr = 2;

  if (num_rows != x_size && num_cols != y_size)
    ierr = 3;
  /*-----------------------------------------------------------------------
   *-----------------------------------------------------------------------*/

  if ( num_vectors==1 )
  {
    y_tmp = hypre_SeqVectorCreate(num_cols_offd);
  }
  else
  {
    y_tmp = hypre_SeqMultiVectorCreate(num_cols_offd,num_vectors);
  }

  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings
   *--------------------------------------------------------------------*/
  if (!comm_pkg)
  {
    hypre_MatvecCommPkgCreate(A);
    comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

  HYPRE_Int use_persistent_comm = 0;
#ifdef HYPRE_USING_PERSISTENT_COMM
  use_persistent_comm = num_vectors == 1;
  // JSP TODO: we can use persistent communication for multi-vectors,
  // but then we need different communication handles for different
  // num_vectors.
  hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    // JSP TODO: we should be also able to use persistent communication for multiple vectors
    persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(2, comm_pkg);

    HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    hypre_assert(num_cols_offd == hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs));

    hypre_VectorData(y_tmp) = (HYPRE_Complex *)persistent_comm_handle->send_data;
    hypre_SeqVectorSetDataOwner(y_tmp, 0);
#endif
  }
  else
  {
    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle*, num_vectors, HYPRE_MEMORY_HOST);
  }
  hypre_SeqVectorInitialize(y_tmp);

  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  if (!use_persistent_comm)
  {
    y_buf_data = hypre_CTAlloc( HYPRE_Complex*,  num_vectors , HYPRE_MEMORY_HOST);
    for ( jv=0; jv<num_vectors; ++jv )
      y_buf_data[jv] = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	  (comm_pkg,  num_sends), HYPRE_MEMORY_HOST);
  }
  y_tmp_data = hypre_VectorData(y_tmp);
  y_local_data = hypre_VectorData(y_local);

  hypre_assert( idxstride==1 ); /* only 'column' storage of multivectors
				 * implemented so far */
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

printf("par_cycle.c 5.1: the vector norm is %16.16f  \n",  hypre_SeqVectorInnerProd(x_local, x_local));
  if (num_cols_offd)
  {
    if (A->offdT)
    {
      // offdT is optional. Used only if it's present.
      hypre_CSRMatrixMatvec(alpha, A->offdT, x_local, 0.0, y_tmp);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
      SyncVectorToHost(y_tmp);
#endif
    }
    else
    {
      hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);
    }
  }

printf("par_cycle.c 5.2: the vector norm is %16.16f  \n",  hypre_SeqVectorInnerProd(y_tmp, y_tmp));
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle);
#endif
  }
  else
  {
    for ( jv=0; jv<num_vectors; ++jv )
    {
      /* this is where we assume multivectors are 'column' storage */
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
	( 2, comm_pkg, &(y_tmp_data[jv*num_cols_offd]), y_buf_data[jv] );
    }
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif

printf("par_cycle.c 5.3: the vector norm is %16.16f  \n",  hypre_SeqVectorInnerProd(x_local, x_local));
  if (A->diagT)
  {
    // diagT is optional. Used only if it's present.
    hypre_CSRMatrixMatvec(alpha, A->diagT, x_local, beta, y_local);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
    SyncVectorToHost(y_local);
#endif
  }
  else
  {
    hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);
  }

printf("par_cycle.c 5.4: the vector norm is %16.16f  \n",  hypre_SeqVectorInnerProd(y_local, y_local));
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif

  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle);
#endif
  }
  else
  {
    for ( jv=0; jv<num_vectors; ++jv )
    {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
    }
    hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif

  if ( num_vectors==1 )
  {
    index = 0;
    for (i = 0; i < num_sends; i++)
    {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
#ifdef HYPRE_USING_PERSISTENT_COMM
	  += ((HYPRE_Complex *)persistent_comm_handle->recv_data)[index++];
#else
      += y_buf_data[0][index++];
#endif
    }
  }
  else
    for ( jv=0; jv<num_vectors; ++jv )
    {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
	start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
	for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	  y_local_data[ jv*vecstride +
	    idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ]
	    += y_buf_data[jv][index++];
      }
    }
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD
  UpdateHRC(y_local);
#endif
  hypre_SeqVectorDestroy(y_tmp);
  y_tmp = NULL;
  if (!use_persistent_comm)
  {
    for ( jv=0; jv<num_vectors; ++jv ) hypre_TFree(y_buf_data[jv], HYPRE_MEMORY_HOST);
    hypre_TFree(y_buf_data, HYPRE_MEMORY_HOST);
  }

#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif

  return ierr;
}




#if 0
/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecMult
 *--------------------------------------------------------------------------*/



// y(:, k2) = alpha*A*x(:, k1) + beta*b(:, k3)
  HYPRE_Int
hypre_ParCSRMatrixMatvecMultOutOfPlace( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Int k1,
    HYPRE_Complex       beta,
    hypre_ParVector    *b,
    HYPRE_Int k3,
    hypre_ParVector    *y,
    HYPRE_Int k2)
{
  //printf("starting matvec\n");
#if 1
  hypre_ParCSRCommHandle *comm_handle;
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  //printf("test 1 passed \n");
  hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
  hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);
  hypre_Vector      *b_local  = hypre_ParVectorLocalVector(b);
  hypre_Vector      *y_local  = hypre_ParVectorLocalVector(y);
  hypre_Vector      *x_local  = hypre_ParVectorLocalVector(x);
  HYPRE_Int          num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
  HYPRE_Int          num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

  hypre_Vector      *x_tmp;
  HYPRE_Int          x_size = hypre_ParVectorGlobalSize(x);
  HYPRE_Int          b_size = hypre_ParVectorGlobalSize(b);
  HYPRE_Int          y_size = hypre_ParVectorGlobalSize(y);
  HYPRE_Int          num_vectors = 1; //while multivec storage, we use only one vector
  //printf("num vectors = %d, x_size %d y_size %d \n", num_vectors,hypre_VectorSize(x_local) , hypre_VectorSize(y_local));
  HYPRE_Int x_local_size = hypre_VectorSize(x_local);

  HYPRE_Int y_local_size = hypre_VectorSize(y_local);
  //hypre_VectorNumVectors(x_local);
  HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_Int          ierr = 0;
  HYPRE_Int          num_sends, i, j, jv, index, start;

  HYPRE_Int          vecstride = hypre_VectorVectorStride( x_local );
  HYPRE_Int          idxstride = hypre_VectorIndexStride( x_local );
  //KS there is ONE vector in multivector!
  HYPRE_Complex     *x_tmp_data, *x_buf_data;
  //get the gpu data
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Complex     *x_local_data = hypre_VectorDeviceData(x_local);

  //printf("test 2 passed \n");
#else

  HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);

  //printf("test 2 alt passed \n");
#endif
  //works
  /*---------------------------------------------------------------------
   *  Check for size compatibility.  ParMatvec returns ierr = 11 if
   *  length of X doesn't equal the number of columns of A,
   *  ierr = 12 if the length of Y doesn't equal the number of rows
   *  of A, and ierr = 13 if both are true.
   *
   *  Because temporary vectors are often used in ParMatvec, none of 
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/
  PUSH_RANGE_PAYLOAD("PAR_CSR_MATVEC",5,x_size);
  hypre_assert( idxstride>0 );

  if (num_cols != x_size)
    ierr = 11;

  if (num_rows != y_size || num_rows != b_size)
    ierr = 12;

  if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
    ierr = 13;
  x_tmp = hypre_SeqVectorCreate( num_cols_offd );
  //printf("test 3 passed \n");
  /*---------------------------------------------------------------------
   * If there exists no CommPkg for A, a CommPkg is generated using
   * equally load balanced partitionings
   *--------------------------------------------------------------------*/
  if (!comm_pkg)
  {
    hypre_MatvecCommPkgCreate(A);
    comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  }

  //printf("test 4 passed \n");
  //printf("1.7\n");
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_PACK",3);
  HYPRE_Int use_persistent_comm = 0;
#ifdef HYPRE_USING_PERSISTENT_COMM
  use_persistent_comm = 1;
  // JSP TODO: we can use persistent co/x_bmmunication for multi-vectors,
  // but then we need different communication handles for different
  // num_vectors.
  hypre_ParCSRPersistentCommHandle *persistent_comm_handle;
#endif

  if ( use_persistent_comm )
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    PUSH_RANGE("PERCOMM1",0);
    persistent_comm_handle = hypre_ParCSRCommPkgGetPersistentCommHandle(1, comm_pkg);

    HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    hypre_assert(num_cols_offd == hypre_ParCSRCommPkgRecvVecStart(comm_pkg, num_recvs));

    hypre_VectorData(x_tmp) = (HYPRE_Complex *)persistent_comm_handle->recv_data;
    hypre_SeqVectorSetDataOwner(x_tmp, 0);
    POP_RANGE;
#endif
  }
  else
  {
    comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1, HYPRE_MEMORY_HOST);
  }

  //printf("1.8\n");

  hypre_SeqVectorInitialize(x_tmp);

  //printf("after initializing x_tmp\n");
  //printf("test xx passed \n");
#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU) 
  //printf("x_tmp_data dev\n"); 
  x_tmp_data = hypre_VectorDeviceData(x_tmp);
#else
  //printf("x_tmp_data CPU\n"); 
  x_tmp_data = hypre_VectorData(x_tmp);
#endif

  //printf("1.9\n");
  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  //works if if 0 is here
  if (!use_persistent_comm)
  {
    //x_buf_data = hypre_CTAlloc( HYPRE_Complex*,  num_vectors , HYPRE_MEMORY_HOST);
    //for ( jv=0; jv<num_vectors; ++jv )
    //printf("x buf data alloc\n");
#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU) 
    //printf("allocating x_buf_data, size %d \n",hypre_ParCSRCommPkgSendMapStart(comm_pkg,  num_sends) );
    x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	(comm_pkg,  num_sends), HYPRE_MEMORY_DEVICE);
#else
    x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	(comm_pkg,  num_sends), HYPRE_MEMORY_SHARED);
#endif
  }

  //printf("buf data created\n");
  HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);


  //printf("1.10\n");
  //printf("first, begin = %d , end = %d \n", begin, end);
#ifdef HYPRE_USING_GPU
  PUSH_RANGE("PERCOMM2DEVICE",4);
#ifdef HYPRE_USING_PERSISTENT_COMM

  //printf("packing on device using persistent comm\n");    
  PackOnDevice((HYPRE_Complex*)persistent_comm_handle->send_data,&x_local_data[k1*x_size],hypre_ParCSRCommPkgSendMapElmts(comm_pkg),begin,end,HYPRE_STREAM(4));
  //PrintPointerAttributes(persistent_comm_handle->send_data);
#else//persistent comm
#if defined(DEBUG_PACK_ON_DEVICE)
  //printf("packing on device using MANAGED\n");    
  hypre_CheckErrorDevice(cudaDeviceSynchronize());
  ASSERT_MANAGED(x_buf_data);
  ASSERT_MANAGED(&x_local_data[k1*x_size]);
  ASSERT_MANAGED(hypre_ParCSRCommPkgSendMapElmts(comm_pkg));
#endif //debug pack on device

#if defined (HYPRE_USING_GPU) 
  printf("packing on device 0\n");    
  PackOnDevice((HYPRE_Complex*)x_buf_data,&x_local_data[k1*x_size],hypre_ParCSRCommPkgSendMapElmts(comm_pkg),begin,end,HYPRE_STREAM(4));
#endif
#if defined(DEBUG_PACK_ON_DEVICE)
  printf("packing on device -- devug\n");    
  hypre_CheckErrorDevice(cudaDeviceSynchronize());
#endif//debug pack on device
#endif //persistent commm
  POP_RANGE;
  SetAsyncMode(1);
  //printf("1.12.1\n");
  //     hypre_CheckErrorDevice(cudaDeviceSynchronize());
  //printf("1.11\n");
  hypre_CSRMatrixMatvecMultOutOfPlace( alpha, diag, x_local,k1, beta, b_local,k3, y_local, k2, 0);

  //printf("1.12\n");
  //hypre_SeqVectorUpdateHost(y_local);
  //hypre_SeqVectorUpdateHost(x_local);
  //hypre_SeqVectorUpdateHost(b_local);
  SetAsyncMode(0);
#ifndef HYPRE_USING_UNIFIED_MEMORY
  HYPRE_Int *comm_d;
  HYPRE_Int *comm_h = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
  //printf("abour to malloc commm_d\n");
  cudaMalloc(&comm_d, (end-begin) * sizeof(HYPRE_Int));
  //for (int ii=begin; ii<end; ++ii){
  //printf("comm[%d] =  %d \n", ii, comm_h[ii]);
  //}
  //printf("commmm copying %d elements to the GPU \n ", (end-begin));
  cudaMemcpy(comm_d,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),  (end-begin) * sizeof(HYPRE_Int),cudaMemcpyHostToDevice  );
  //printf("1.13 pack on dev GPU only\n");
  PackOnDeviceGPUonly((HYPRE_Complex*)x_buf_data,
      &x_local_data[k1*x_local_size],
      comm_d,
      begin,
      end);
  //printf("1.14\n");
  //  cudaMemcpy(&x_buf_data[0],&x_local_data[begin],hypre_ParCSRCommPkgSendMapElmts(comm_pkg),begin,end);
#endif

#else //if def using GPU
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD 
  PUSH_RANGE("MPI_PACK_OMP",4);
  SyncVectorToHost(x_local);
#endif

#if defined(HYPRE_USING_OPENMP_OFFLOAD_NOT_USED)
  HYPRE_Int num_threads=64;
  HYPRE_Int num_teams = (end-begin+(end-begin)%num_threads)/num_threads;
  HYPRE_Int *local_send_map_elmts = comm_pkg->send_map_elmts;
  //printf("USING OFFLOADED PACKING OF BUFER\n");
#pragma omp target teams  distribute  parallel for private(i) num_teams(num_teams) thread_limit(num_threads) is_device_ptr(x_local_data,x_buf_data,comm_pkg,local_send_map_elmts)
#elif defined(HYPRE_USING_OPENMP) //open mp offload
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
  //printf("second loop, begin = %d , end = %d \n", begin, end);
  for (i = begin; i < end; i++)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    //printf("not using GPU, persostent commmmmm!\n");
    ((HYPRE_Complex *)persistent_comm_handle->send_data)[i - begin]  = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,i)];
#else

    //printf("not using persistent commmmmm nor GPU!\n");
#endif//persistent comm

  }//for
  POP_RANGE; // "MPI_PACK_OMP"
#endif//hypre_use_gpu
  //WORKS
  //printf("test 1.15\n");
  hypre_assert( idxstride==1 );
  /* ... The assert is because the following loop only works for 'column'
     storage of a multivector. This needs to be fixed to work more generally,
     at least for 'row' storage. This in turn, means either change CommPkg so
     num_sends is no.zones*no.vectors (not no.zones) or, less dangerously, put
     a stride in the logic of CommHandleCreate (stride either from a new arg or
     a new variable inside CommPkg).  Or put the num_vector iteration inside
     CommHandleCreate (perhaps a new multivector variant of it).
     */
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
  POP_RANGE;
  PUSH_RANGE("MPI_HALO_EXC_SEND",4);
  // WORKS

  if (use_persistent_comm)
  {
    //printf("PERSISTENT COMM!\n");
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleStart(persistent_comm_handle);
#endif
  }
  else
  {
    jv = 0;
    //printf("creating comm handle \n");
    //printf("jv = %d num_cols_offd = %d \n", 0, num_cols_offd);
    //printf("asking for x_tmp_data[%d] i.e. %d*%d \n", jv*num_cols_offd, jv+k1*x_local_size, num_cols_offd);     
    comm_handle = hypre_ParCSRCommHandleCreate
      ( 1, comm_pkg, x_buf_data,&x_tmp_data[k1*x_local_size] );

    //printf("1.15\n");
  }
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
  POP_RANGE;
#ifndef HYPRE_USING_GPU
  /// hypre_CSRMatrixMatvecOutOfPlace( alpha, diag, x_local+k1*x_local_size, beta, b_local, y_local+k2*y_local_size, 0);
#endif

  //nprintf("test 1.16\n");
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_HALO_EXC_RECV",6);
  if (use_persistent_comm)
  {
#ifdef HYPRE_USING_PERSISTENT_COMM
    hypre_ParCSRPersistentCommHandleWait(persistent_comm_handle);
#endif
  }
  else
  {
    comm_handle = NULL;
    hypre_TFree(comm_handle, HYPRE_MEMORY_HOST);
  }
  POP_RANGE;
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_HALO_EXCHANGE] += hypre_MPI_Wtime();
#endif
  //11/20/2018 works
  //hypre_SeqVectorUpdateDevice(x_tmp);
#ifdef HYPRE_USING_MAPPED_OPENMP_OFFLOAD   
  UpdateHRC(x_tmp);
#endif
  //printf("num cols of diag %d \n", num_cols_offd);
  if (num_cols_offd) {hypre_CSRMatrixMatvecMult( alpha, offd, x_tmp, k1, 1.0, y_local, k2); }
  //crashes (jan 2019) 
  //11/20/2018 works
  //CRASHED 
  //if (num_cols_offd) hypre_SeqVectorUpdateHost(y_local);  
  //hypre_SeqVectorUpdateHost(x_tmp); 
#ifdef HYPRE_PROFILE
  // hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] -= hypre_MPI_Wtime();
#endif
  PUSH_RANGE("MPI_UNPACK",5);
  // hypre_SeqVectorDestroy(x_tmp);
  // x_tmp = NULL;
  //this causes crash!!!
  //
  if (!use_persistent_comm)
  {

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
    //printf("freeeeing x_buf_dta \n");     
    //causes crash 
    //hypre_TFree(x_buf_data, HYPRE_MEMORY_DEVICE);

#else
    hypre_TFree(x_buf_data, HYPRE_MEMORY_SHARED);
#endif
  }
  //20/11/2018 crashes!!!  
#ifdef HYPRE_PROFILE
  hypre_profile_times[HYPRE_TIMER_ID_PACK_UNPACK] += hypre_MPI_Wtime();
#endif
  POP_RANGE;
#ifdef HYPRE_USING_GPU
  //  hypre_CheckErrorDevice(cudaStreamSynchronize(HYPRE_STREAM(4)));
#endif
  POP_RANGE; // PAR_CSR
  return ierr;
#endif //if 0
}
#endif
  HYPRE_Int
hypre_ParCSRMatrixMatvecMultOutOfPlace( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Int k1,
    HYPRE_Complex       beta,
    hypre_ParVector    *b,
    HYPRE_Int k3,
    hypre_ParVector    *y,
    HYPRE_Int k2)
{
  //works if commented out completely
  hypre_ParCSRCommHandle *comm_handle;
  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  //printf("test 1 passed \n");
  //works if commented here
  hypre_CSRMatrix   *diag   = hypre_ParCSRMatrixDiag(A);
  hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(A);
  hypre_Vector      *b_local  = hypre_ParVectorLocalVector(b);
  hypre_Vector      *y_local  = hypre_ParVectorLocalVector(y);
  hypre_Vector      *x_local  = hypre_ParVectorLocalVector(x);
  //HYPRE_Int          num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
  //HYPRE_Int          num_cols = hypre_ParCSRMatrixGlobalNumCols(A);
  //works if commenrted here
  hypre_Vector      *x_tmp = A->x_tmp;
  //   hypre_Vector      *x_tmp;

  //HYPRE_Int          x_size = hypre_ParVectorGlobalSize(x);
  //HYPRE_Int          b_size = hypre_ParVectorGlobalSize(b);
  //HYPRE_Int          y_size = hypre_ParVectorGlobalSize(y);
  //HYPRE_Int          num_vectors = 1; //while multivec storage, we use only one vector
  HYPRE_Int x_local_size = hypre_VectorSize(x_local);
  //works if commented below
  //HYPRE_Int y_local_size = hypre_VectorSize(y_local);
  HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_Int          ierr = 0;
  HYPRE_Int          num_sends;

  //HYPRE_Int          vecstride = hypre_VectorVectorStride( x_local );
  //HYPRE_Int          idxstride = hypre_VectorIndexStride( x_local );
  HYPRE_Complex     *x_tmp_data, *x_buf_data;
  //works if commented here
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  HYPRE_Complex     *x_local_data = hypre_VectorDeviceData(x_local);
#else
  HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);
#endif
#if 1 

  if ( x_local_data==NULL){
    //printf("x_local_data == NULL, x_local_size= %d \n", x_local_size);
    return 0;
  }
#endif

  //workss
  if (!comm_pkg)
  {
    hypre_MatvecCommPkgCreate(A);
    comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  }
  comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1, HYPRE_MEMORY_HOST);
  //works
  //x_tmp = hypre_SeqVectorCreate( num_cols_offd );
  // hypre_SeqVectorInitialize(x_tmp);
  if (x_tmp == NULL){
    //printf("x_tmp is NULL! and num_cols_offd = %d \n", num_cols_offd);
    x_tmp = hypre_SeqVectorCreate( num_cols_offd );

    hypre_SeqVectorInitialize(x_tmp);
  }


#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU) 
  x_tmp_data = hypre_VectorDeviceData(x_tmp);
#else
  x_tmp_data = hypre_VectorData(x_tmp);
#endif
  //works if commented below

  //fails if commented here
  num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  if (A->x_buf == NULL){

    A->x_buf = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	(comm_pkg,  num_sends), HYPRE_MEMORY_DEVICE);
  }

#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU) 
  x_buf_data = A->x_buf;

  //  x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
  //    (comm_pkg,  num_sends), HYPRE_MEMORY_DEVICE);
#else
  x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
      (comm_pkg,  num_sends), HYPRE_MEMORY_SHARED);
#endif
  //fails
  hypre_CSRMatrixMatvecMultOutOfPlace( alpha, diag, x_local,k1, beta, b_local,k3, y_local, k2, 0);

  //  HYPRE_Int *comm_d = A->comm_d;
  HYPRE_Int *comm_d;
  HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
  HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
  //printf("begin = %d end = %d \n", begin, end);
  if (A->comm_d == NULL)
  {
    if ((end-begin) != 0)
    {

      A->comm_d =  hypre_CTAlloc(HYPRE_Int,  (end-begin), HYPRE_MEMORY_DEVICE);;

      cudaMemcpy(A->comm_d,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),  (end-begin) * sizeof(HYPRE_Int),cudaMemcpyHostToDevice  );
    }

  }
  comm_d = A->comm_d;
  //HYPRE_Int *comm_h = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
#if 1
  //cudaMalloc(&comm_d, (end-begin) * sizeof(HYPRE_Int));
  //cudaMemcpy(comm_d,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),  (end-begin)* sizeof(HYPRE_Int),cudaMemcpyHostToDevice  );
  comm_d = A->comm_d;
  //firstCall = 0;
#endif 
  //printf("matvec: need to copy %d intes \n", end-begin);
  if ((end-begin) !=0){
    //printf("begin = %d end = %d \n", begin, end); 
    PackOnDeviceGPUonly((HYPRE_Complex*)x_buf_data,
	&x_local_data[k1*x_local_size],
	comm_d,
	begin,
	end);
  }

  comm_handle = hypre_CTAlloc(hypre_ParCSRCommHandle, 1, HYPRE_MEMORY_HOST);
#if 0 
  comm_handle = hypre_ParCSRCommHandleCreate
    ( 111, comm_pkg, x_buf_data,&x_tmp_data[k1*x_local_size] );
#endif
  comm_handle = hypre_ParCSRCommHandleCreate
    ( 111, comm_pkg, x_buf_data,x_tmp_data );
  //WORKS!
  if (num_cols_offd) {hypre_CSRMatrixMatvecMult( alpha, offd, x_tmp, k1, 1.0, y_local, k2); }

#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)

  //  hypre_TFree(x_buf_data, HYPRE_MEMORY_DEVICE);
#else
  hypre_TFree(x_buf_data, HYPRE_MEMORY_SHARED);
#endif
  return ierr;
}




/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/
  HYPRE_Int
hypre_ParCSRMatrixMatvec_FF( HYPRE_Complex       alpha,
    hypre_ParCSRMatrix *A,
    hypre_ParVector    *x,
    HYPRE_Complex       beta,
    hypre_ParVector    *y,
    HYPRE_Int          *CF_marker,
    HYPRE_Int           fpt )
{
  MPI_Comm                comm = hypre_ParCSRMatrixComm(A);
  hypre_ParCSRCommHandle *comm_handle;
  hypre_ParCSRCommPkg    *comm_pkg = hypre_ParCSRMatrixCommPkg(A);
  hypre_CSRMatrix        *diag   = hypre_ParCSRMatrixDiag(A);
  hypre_CSRMatrix        *offd   = hypre_ParCSRMatrixOffd(A);
  hypre_Vector           *x_local  = hypre_ParVectorLocalVector(x);
  hypre_Vector           *y_local  = hypre_ParVectorLocalVector(y);
  HYPRE_Int               num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
  HYPRE_Int               num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

  hypre_Vector      *x_tmp;
  HYPRE_Int          x_size = hypre_ParVectorGlobalSize(x);
  HYPRE_Int          y_size = hypre_ParVectorGlobalSize(y);
  HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);
  HYPRE_Int          ierr = 0;
  HYPRE_Int          num_sends, i, j, index, start, num_procs;
  HYPRE_Int         *int_buf_data = NULL;
  HYPRE_Int         *CF_marker_offd = NULL;

  HYPRE_Complex     *x_tmp_data = NULL;
  HYPRE_Complex     *x_buf_data = NULL;
  HYPRE_Complex     *x_local_data = hypre_VectorData(x_local);
  /*---------------------------------------------------------------------
   *  Check for size compatibility.  ParMatvec returns ierr = 11 if
   *  length of X doesn't equal the number of columns of A,
   *  ierr = 12 if the length of Y doesn't equal the number of rows
   *  of A, and ierr = 13 if both are true.
   *
   *  Because temporary vectors are often used in ParMatvec, none of
   *  these conditions terminates processing, and the ierr flag
   *  is informational only.
   *--------------------------------------------------------------------*/

  hypre_MPI_Comm_size(comm,&num_procs);

  if (num_cols != x_size)
    ierr = 11;

  if (num_rows != y_size)
    ierr = 12;

  if (num_cols != x_size && num_rows != y_size)
    ierr = 13;

  if (num_procs > 1)
  {
    if (num_cols_offd)
    {
      x_tmp = hypre_SeqVectorCreate( num_cols_offd );
      hypre_SeqVectorInitialize(x_tmp);
      x_tmp_data = hypre_VectorData(x_tmp);
    }

    /*---------------------------------------------------------------------
     * If there exists no CommPkg for A, a CommPkg is generated using
     * equally load balanced partitionings
     *--------------------------------------------------------------------*/
    if (!comm_pkg)
    {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
    }

    num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    if (num_sends)
      x_buf_data = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
	  (comm_pkg,  num_sends), HYPRE_MEMORY_HOST);

    index = 0;
    for (i = 0; i < num_sends; i++)
    {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	x_buf_data[index++]
	  = x_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
    }
    comm_handle =
      hypre_ParCSRCommHandleCreate ( 1, comm_pkg, x_buf_data, x_tmp_data );
  }
  hypre_CSRMatrixMatvec_FF( alpha, diag, x_local, beta, y_local, CF_marker,
      CF_marker, fpt);

  if (num_procs > 1)
  {
    hypre_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    if (num_sends)
      int_buf_data = hypre_CTAlloc(HYPRE_Int,  hypre_ParCSRCommPkgSendMapStart
	  (comm_pkg,  num_sends), HYPRE_MEMORY_HOST);
    if (num_cols_offd) CF_marker_offd = hypre_CTAlloc(HYPRE_Int,  num_cols_offd, HYPRE_MEMORY_HOST);
    index = 0;
    for (i = 0; i < num_sends; i++)
    {
      start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
      for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
	int_buf_data[index++]
	  = CF_marker[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
    }
    comm_handle =
      hypre_ParCSRCommHandleCreate(11,comm_pkg,int_buf_data,CF_marker_offd );

    hypre_ParCSRCommHandleDestroy(comm_handle);
    comm_handle = NULL;

    if (num_cols_offd) hypre_CSRMatrixMatvec_FF( alpha, offd, x_tmp, 1.0, y_local,
	CF_marker, CF_marker_offd, fpt);

    hypre_SeqVectorDestroy(x_tmp);
    x_tmp = NULL;
    hypre_TFree(x_buf_data, HYPRE_MEMORY_HOST);
    hypre_TFree(int_buf_data, HYPRE_MEMORY_HOST);
    hypre_TFree(CF_marker_offd, HYPRE_MEMORY_HOST);
  }

  return ierr;
}
