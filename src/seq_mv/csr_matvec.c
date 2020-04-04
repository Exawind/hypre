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
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include <assert.h>

#include "gpukernels.h"


// for multivectors
//

/* y[offset:end, k2] = alpha*A[offset:end,:]*x(:, k1) + beta*b[offset:end, k3] */
	HYPRE_Int
hypre_CSRMatrixMatvecMultOutOfPlace( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int        offset     )
{
	HYPRE_Int ret, ierr = 101;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)  
	ret=hypre_CSRMatrixMatvecMultDevice( alpha,A,x,k1, beta,b,k3, y,k2, offset);
	return ret;
#else
	printf("it does not work with MANAGED nor without GPU! change your parameters or write a proper function\n");
#endif
	return ierr;
}

	HYPRE_Int
hypre_CSRMatrixMatvecTMultOutOfPlace( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int        offset     )
{
	HYPRE_Int ret, ierr = 101;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)  
	ret=hypre_CSRMatrixMatvecTMultDevice( alpha,A,x,k1, beta,b,k3, y,k2, offset);
	return ret;
#else
	printf("it does not work with MANAGED nor without GPU! change your parameters or write a proper function\n");
#endif
	return ierr;
}
#if 1
HYPRE_Int hypre_CSRMatrixMatvecMultAsynch( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int        offset     )
{
	HYPRE_Int ret, ierr = 101;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)  
	ret=hypre_CSRMatrixMatvecMultAsynchDevice( alpha,A,x,k1, beta,b,k3, y,k2, offset);
	return ret;
#else
	printf("it does not work with MANAGED nor without GPU! change your parameters or write a proper function\n");
#endif
	return ierr;
}
//two matvecs in one, experimental kernel (two inputs)

HYPRE_Int hypre_CSRMatrixMatvecMultAsynchTwoInOne( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A1,
		hypre_CSRMatrix *A2,
		hypre_Vector    *x1,
		HYPRE_Int k11,
		hypre_Vector    *x2,
		HYPRE_Int k12,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int        offset     )
{
	HYPRE_Int ret, ierr = 101;
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)  
	ret=hypre_CSRMatrixMatvecMultAsynchTwoInOneDevice( alpha,A1,A2, x1,k11,x2, k12, beta,b,k3, y,k2, offset);
	return ret;
#else
	printf("it does not work with MANAGED nor without GPU! change your parameters or write a proper function\n");
#endif
	return ierr;
}

#endif
	HYPRE_Int
hypre_CSRMatrixMatvecMult( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *y,
		HYPRE_Int k2     )
{
	return hypre_CSRMatrixMatvecMultOutOfPlace(alpha, A, x,k1, beta, y,k2, y,k2, 0);
}



/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
	HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlace( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		hypre_Vector    *y,
		HYPRE_Int        offset )
{
#ifdef HYPRE_PROFILE
	HYPRE_Real time_begin = hypre_MPI_Wtime();
#endif

#if defined(HYPRE_USING_GPU)  /* CUDA */
	PUSH_RANGE_PAYLOAD("MATVEC",0, hypre_CSRMatrixNumRows(A));
	HYPRE_Int ierr = hypre_CSRMatrixMatvecDevice( alpha,A,x,beta,b,y,offset );
	POP_RANGE;
#elif defined(HYPRE_USING_OPENMP_OFFLOAD) /* OMP 4.5 */
	PUSH_RANGE_PAYLOAD("MATVEC-OMP",0, hypre_CSRMatrixNumRows(A));
	HYPRE_Int ierr = hypre_CSRMatrixMatvecOutOfPlaceOOMP( alpha,A,x,beta,b,y,offset );
	POP_RANGE;
#else /* CPU */
	HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
	HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
	HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
	HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A) - offset;
	HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);
	/*HYPRE_Int         num_nnz  = hypre_CSRMatrixNumNonzeros(A);*/

	HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
	HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

	HYPRE_Complex    *x_data = hypre_VectorData(x);
	HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
	HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         b_size = hypre_VectorSize(b) - offset;
	HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
	HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
	HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
	HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
	/*HYPRE_Int         idxstride_b = hypre_VectorIndexStride(b);
		HYPRE_Int         vecstride_b = hypre_VectorVectorStride(b);*/
	HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
	HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);
	HYPRE_Complex     temp, tempx;
	HYPRE_Int         i, j, jj, m, ierr=0;
	HYPRE_Real        xpar=0.7;
	hypre_Vector     *x_tmp = NULL;

	/*---------------------------------------------------------------------
	 *  Check for size compatibility.  Matvec returns ierr = 1 if
	 *  length of X doesn't equal the number of columns of A,
	 *  ierr = 2 if the length of Y doesn't equal the number of rows
	 *  of A, and ierr = 3 if both are true.
	 *
	 *  Because temporary vectors are often used in Matvec, none of
	 *  these conditions terminates processing, and the ierr flag
	 *  is informational only.
	 *--------------------------------------------------------------------*/

	hypre_assert( num_vectors == hypre_VectorNumVectors(y) );
	hypre_assert( num_vectors == hypre_VectorNumVectors(b) );

	if (num_cols != x_size)
		ierr = 1;

	if (num_rows != y_size || num_rows != b_size)
		ierr = 2;

	if (num_cols != x_size && (num_rows != y_size || num_rows != b_size))
		ierr = 3;

	/*-----------------------------------------------------------------------
	 * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
	 *-----------------------------------------------------------------------*/

	if (alpha == 0.0)
	{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i = 0; i < num_rows*num_vectors; i++)
			y_data[i] = beta*b_data[i];

#ifdef HYPRE_PROFILE
		hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

		return ierr;
	}

	if (x == y)
	{
		x_tmp = hypre_SeqVectorCloneDeep(x);
		x_data = hypre_VectorData(x_tmp);
	}

	/*-----------------------------------------------------------------------
	 * y = (beta/alpha)*y
	 *-----------------------------------------------------------------------*/

	temp = beta / alpha;

	/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

	if (num_rownnz < xpar*(num_rows) || num_vectors > 1)
	{
		/*-----------------------------------------------------------------------
		 * y = (beta/alpha)*y
		 *-----------------------------------------------------------------------*/

		if (temp != 1.0)
		{
			if (temp == 0.0)
			{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
				for (i = 0; i < num_rows*num_vectors; i++)
					y_data[i] = 0.0;
			}
			else
			{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
				for (i = 0; i < num_rows*num_vectors; i++)
					y_data[i] = b_data[i]*temp;
			}
		}
		else
		{
			for (i = 0; i < num_rows*num_vectors; i++)
				y_data[i] = b_data[i];
		}


		/*-----------------------------------------------------------------
		 * y += A*x
		 *-----------------------------------------------------------------*/

		if (num_rownnz < xpar*(num_rows))
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,m,tempx) HYPRE_SMP_SCHEDULE
#endif

			for (i = 0; i < num_rownnz; i++)
			{
				m = A_rownnz[i];

				/*
				 * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
				 * {
				 *         j = A_j[jj];
				 *  y_data[m] += A_data[jj] * x_data[j];
				 * } */
				if ( num_vectors==1 )
				{
					tempx = 0;
					for (jj = A_i[m]; jj < A_i[m+1]; jj++)
						tempx +=  A_data[jj] * x_data[A_j[jj]];
					y_data[m] += tempx;
				}
				else
					for ( j=0; j<num_vectors; ++j )
					{
						tempx = 0;
						for (jj = A_i[m]; jj < A_i[m+1]; jj++)
							tempx +=  A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
						y_data[ j*vecstride_y + m*idxstride_y] += tempx;
					}
			}
		}
		else // num_vectors > 1
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,jj,tempx) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_rows; i++)
			{
				for (j = 0; j < num_vectors; ++j)
				{
					tempx = 0;
					for (jj = A_i[i]; jj < A_i[i+1]; jj++)
					{
						tempx += A_data[jj] * x_data[ j*vecstride_x + A_j[jj]*idxstride_x ];
					}
					y_data[ j*vecstride_y + i*idxstride_y ] += tempx;
				}
			}
		}

		/*-----------------------------------------------------------------
		 * y = alpha*y
		 *-----------------------------------------------------------------*/

		if (alpha != 1.0)
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_rows*num_vectors; i++)
				y_data[i] *= alpha;
		}
	}
	else
	{ // JSP: this is currently the only path optimized
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,tempx)
#endif
		{
			HYPRE_Int iBegin = hypre_CSRMatrixGetLoadBalancedPartitionBegin(A);
			HYPRE_Int iEnd = hypre_CSRMatrixGetLoadBalancedPartitionEnd(A);
			hypre_assert(iBegin <= iEnd);
			hypre_assert(iBegin >= 0 && iBegin <= num_rows);
			hypre_assert(iEnd >= 0 && iEnd <= num_rows);

			if (0 == temp)
			{
				if (1 == alpha) // JSP: a common path
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = 0.0;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = A*x
				else if (-1 == alpha)
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = 0.0;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx -= A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = -A*x
				else
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = 0.0;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = alpha*tempx;
					}
				} // y = alpha*A*x
			} // temp == 0
			else if (-1 == temp) // beta == -alpha
			{
				if (1 == alpha) // JSP: a common path
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = -b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = A*x - y
				else if (-1 == alpha) // JSP: a common path
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx -= A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = -A*x + y
				else
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = -b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = alpha*tempx;
					}
				} // y = alpha*(A*x - y)
			} // temp == -1
			else if (1 == temp)
			{
				if (1 == alpha) // JSP: a common path
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = A*x + y
				else if (-1 == alpha)
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = -b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx -= A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = -A*x - y
				else
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = b_data[i];
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = alpha*tempx;
					}
				} // y = alpha*(A*x + y)
			}
			else
			{
				if (1 == alpha) // JSP: a common path
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = b_data[i]*temp;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = A*x + temp*y
				else if (-1 == alpha)
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = -b_data[i]*temp;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx -= A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = tempx;
					}
				} // y = -A*x - temp*y
				else
				{
					for (i = iBegin; i < iEnd; i++)
					{
						tempx = b_data[i]*temp;
						for (jj = A_i[i]; jj < A_i[i+1]; jj++)
						{
							tempx += A_data[jj] * x_data[A_j[jj]];
						}
						y_data[i] = alpha*tempx;
					}
				} // y = alpha*(A*x + temp*y)
			} // temp != 0 && temp != -1 && temp != 1
		} // omp parallel
	}

	if (x == y) hypre_SeqVectorDestroy(x_tmp);

#endif /* CPU */

#ifdef HYPRE_PROFILE
	hypre_profile_times[HYPRE_TIMER_ID_MATVEC] += hypre_MPI_Wtime() - time_begin;
#endif

	return ierr;
}

	HYPRE_Int
hypre_CSRMatrixMatvec( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *y     )
{
	return hypre_CSRMatrixMatvecOutOfPlace(alpha, A, x, beta, y, y, 0);
}

#if defined (HYPRE_USING_UNIFIED_MEMORY)
	HYPRE_Int
hypre_CSRMatrixMatvec3( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *y     )
{
	return hypre_CSRMatrixMatvecOutOfPlaceOOMP3(alpha, A, x, beta, y, y, 0);
}
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *  This version is using a different (more efficient) threading scheme

 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_CSRMatrixMatvecT( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *y     )
{
	HYPRE_Complex    *A_data    = hypre_CSRMatrixData(A);
	HYPRE_Int        *A_i       = hypre_CSRMatrixI(A);
	HYPRE_Int        *A_j       = hypre_CSRMatrixJ(A);
	HYPRE_Int         num_rows  = hypre_CSRMatrixNumRows(A);
	HYPRE_Int         num_cols  = hypre_CSRMatrixNumCols(A);

	HYPRE_Complex    *x_data = hypre_VectorData(x);
	HYPRE_Complex    *y_data = hypre_VectorData(y);
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         y_size = hypre_VectorSize(y);
	HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
	HYPRE_Int         idxstride_y = hypre_VectorIndexStride(y);
	HYPRE_Int         vecstride_y = hypre_VectorVectorStride(y);
	HYPRE_Int         idxstride_x = hypre_VectorIndexStride(x);
	HYPRE_Int         vecstride_x = hypre_VectorVectorStride(x);

	HYPRE_Complex     temp;

	HYPRE_Complex    *y_data_expand;
	HYPRE_Int         my_thread_num = 0, offset = 0;

	HYPRE_Int         i, j, jv, jj;
	HYPRE_Int         num_threads;

	HYPRE_Int         ierr  = 0;

	hypre_Vector     *x_tmp = NULL;

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

	hypre_assert( num_vectors == hypre_VectorNumVectors(y) );

	if (num_rows != x_size)
		ierr = 1;

	if (num_cols != y_size)
		ierr = 2;

	if (num_rows != x_size && num_cols != y_size)
		ierr = 3;
	/*-----------------------------------------------------------------------
	 * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
	 *-----------------------------------------------------------------------*/

	if (alpha == 0.0)
	{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i = 0; i < num_cols*num_vectors; i++)
			y_data[i] *= beta;

		return ierr;
	}

	if (x == y)
	{
		x_tmp = hypre_SeqVectorCloneDeep(x);
		x_data = hypre_VectorData(x_tmp);
	}

	/*-----------------------------------------------------------------------
	 * y = (beta/alpha)*y
	 *-----------------------------------------------------------------------*/

	temp = beta / alpha;

	if (temp != 1.0)
	{
		if (temp == 0.0)
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_cols*num_vectors; i++)
				y_data[i] = 0.0;
		}
		else
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_cols*num_vectors; i++)
				y_data[i] *= temp;
		}
	}

	/*-----------------------------------------------------------------
	 * y += A^T*x
	 *-----------------------------------------------------------------*/
	num_threads = hypre_NumThreads();
	if (num_threads > 1)
	{
		y_data_expand = hypre_CTAlloc(HYPRE_Complex,  num_threads*y_size, HYPRE_MEMORY_HOST);

		if ( num_vectors==1 )
		{

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel private(i,jj,j,my_thread_num,offset)
#endif
			{
				my_thread_num = hypre_GetThreadNum();
				offset =  y_size*my_thread_num;
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
				for (i = 0; i < num_rows; i++)
				{
					for (jj = A_i[i]; jj < A_i[i+1]; jj++)
					{
						j = A_j[jj];
						y_data_expand[offset + j] += A_data[jj] * x_data[i];
					}
				}

				/* implied barrier (for threads)*/
#ifdef HYPRE_USING_OPENMP
#pragma omp for HYPRE_SMP_SCHEDULE
#endif
				for (i = 0; i < y_size; i++)
				{
					for (j = 0; j < num_threads; j++)
					{
						y_data[i] += y_data_expand[j*y_size + i];

					}
				}

			} /* end parallel threaded region */
		}
		else
		{
			/* multiple vector case is not threaded */
			for (i = 0; i < num_rows; i++)
			{
				for ( jv=0; jv<num_vectors; ++jv )
				{
					for (jj = A_i[i]; jj < A_i[i+1]; jj++)
					{
						j = A_j[jj];
						y_data[ j*idxstride_y + jv*vecstride_y ] +=
							A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x];
					}
				}
			}
		}

		hypre_TFree(y_data_expand, HYPRE_MEMORY_HOST);

	}
	else
	{
		for (i = 0; i < num_rows; i++)
		{
			if ( num_vectors==1 )
			{
				for (jj = A_i[i]; jj < A_i[i+1]; jj++)
				{
					j = A_j[jj];
					y_data[j] += A_data[jj] * x_data[i];
				}
			}
			else
			{
				for ( jv=0; jv<num_vectors; ++jv )
				{
					for (jj = A_i[i]; jj < A_i[i+1]; jj++)
					{
						j = A_j[jj];
						y_data[ j*idxstride_y + jv*vecstride_y ] +=
							A_data[jj] * x_data[ i*idxstride_x + jv*vecstride_x ];
					}
				}
			}
		}
	}
	/*-----------------------------------------------------------------
	 * y = alpha*y
	 *-----------------------------------------------------------------*/

	if (alpha != 1.0)
	{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i = 0; i < num_cols*num_vectors; i++)
			y_data[i] *= alpha;
	}

	if (x == y) hypre_SeqVectorDestroy(x_tmp);

	return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_CSRMatrixMatvec_FF( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *y,
		HYPRE_Int       *CF_marker_x,
		HYPRE_Int       *CF_marker_y,
		HYPRE_Int        fpt )
{
	HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
	HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
	HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
	HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
	HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

	HYPRE_Complex    *x_data = hypre_VectorData(x);
	HYPRE_Complex    *y_data = hypre_VectorData(y);
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         y_size = hypre_VectorSize(y);

	HYPRE_Complex      temp;

	HYPRE_Int         i, jj;

	HYPRE_Int         ierr = 0;

	/*---------------------------------------------------------------------
	 *  Check for size compatibility.  Matvec returns ierr = 1 if
	 *  length of X doesn't equal the number of columns of A,
	 *  ierr = 2 if the length of Y doesn't equal the number of rows
	 *  of A, and ierr = 3 if both are true.
	 *
	 *  Because temporary vectors are often used in Matvec, none of
	 *  these conditions terminates processing, and the ierr flag
	 *  is informational only.
	 *--------------------------------------------------------------------*/

	if (num_cols != x_size)
		ierr = 1;

	if (num_rows != y_size)
		ierr = 2;

	if (num_cols != x_size && num_rows != y_size)
		ierr = 3;

	/*-----------------------------------------------------------------------
	 * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
	 *-----------------------------------------------------------------------*/

	if (alpha == 0.0)
	{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i = 0; i < num_rows; i++)
			if (CF_marker_x[i] == fpt) y_data[i] *= beta;

		return ierr;
	}

	/*-----------------------------------------------------------------------
	 * y = (beta/alpha)*y
	 *-----------------------------------------------------------------------*/

	temp = beta / alpha;

	if (temp != 1.0)
	{
		if (temp == 0.0)
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_rows; i++)
				if (CF_marker_x[i] == fpt) y_data[i] = 0.0;
		}
		else
		{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
			for (i = 0; i < num_rows; i++)
				if (CF_marker_x[i] == fpt) y_data[i] *= temp;
		}
	}

	/*-----------------------------------------------------------------
	 * y += A*x
	 *-----------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj) HYPRE_SMP_SCHEDULE
#endif

	for (i = 0; i < num_rows; i++)
	{
		if (CF_marker_x[i] == fpt)
		{
			temp = y_data[i];
			for (jj = A_i[i]; jj < A_i[i+1]; jj++)
				if (CF_marker_y[A_j[jj]] == fpt) temp += A_data[jj] * x_data[A_j[jj]];
			y_data[i] = temp;
		}
	}

	/*-----------------------------------------------------------------
	 * y = alpha*y
	 *-----------------------------------------------------------------*/

	if (alpha != 1.0)
	{
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
		for (i = 0; i < num_rows; i++)
			if (CF_marker_x[i] == fpt) y_data[i] *= alpha;
	}

	return ierr;
}
#if defined(HYPRE_USING_GPU)
	HYPRE_Int
hypre_CSRMatrixMatvecDevice( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		hypre_Vector    *y,
		HYPRE_Int        offset )
{

	static cusparseHandle_t handle;
	static cusparseMatDescr_t descr;
	static HYPRE_Int FirstCall=1;
	cusparseStatus_t status;
	static cudaStream_t s[10];
	static HYPRE_Int myid;

	if (b!=y){

		PUSH_RANGE_PAYLOAD("MEMCPY",1,y->size-offset);
		VecCopy(y->data,b->data,(y->size-offset),HYPRE_STREAM(4));
		POP_RANGE
	}

	if (x==y) hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");

	if (FirstCall){
		PUSH_RANGE("FIRST_CALL",4);

		handle=getCusparseHandle();

		status= cusparseCreateMatDescr(&descr);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR:: Matrix descriptor initialization failed\n");
			return hypre_error_flag;
		}
		cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

		FirstCall=0;
		hypre_int jj;
		for(jj=0;jj<5;jj++)
			s[jj]=HYPRE_STREAM(jj);
		nvtxNameCudaStreamA(s[4], "HYPRE_COMPUTE_STREAM");
		hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
		myid++;
		POP_RANGE;
	}

	PUSH_RANGE("PREFETCH+SPMV",2);

	hypre_CSRMatrixPrefetchToDevice(A);
	hypre_SeqVectorPrefetchToDevice(x);
	hypre_SeqVectorPrefetchToDevice(y);

	//if (offset!=0) hypre_printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: \n");
	cusparseDcsrmv(handle ,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			A->num_rows-offset, A->num_cols, A->num_nonzeros,
			&alpha, descr,
			A->data ,A->i+offset,A->j,
			x->data, &beta, y->data+offset);
	if (!GetAsyncMode()){
		//  hypre_CheckErrorDevice(cudaStreamSynchronize(s[4]));
	}
	POP_RANGE;

	return 0;

}


	HYPRE_Int
hypre_CSRMatrixMatvecMultAsynchDevice( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int offset )
{ 
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         y_size = hypre_VectorSize(y);
	HYPRE_Complex * xddata =  x->d_data;
	HYPRE_Complex * yddata =  y->d_data;



#if 1
	if (A->num_nonzeros){
		// printf("k1 = %d k2 = %d this is matvec. num rows in A %d , num cols in A %d  nnz in A %d alpha = %f beta = %f\n",k1, k2,A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta );
		if ((A->d_data == NULL)) {
			//    printf("A: d_data is NULL; updating!\n");
			//hypre_SeqVectorPrefetchToDevice(A->data);
			//cudaMemPrefetchAsync(ptr,size,device,stream);
			// cudaError_t err; 
			cudaMemPrefetchAsync(A->data,A->num_nonzeros*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4));
			//printf("error code in prefetch %d string %s \n", err, cudaGetErrorString(err));

			hypre_CSRMatrixDeviceData(A)    = hypre_CTAlloc(HYPRE_Complex,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_data,A->data,
					A->num_nonzeros*sizeof(HYPRE_Complex),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_i == NULL)) {
			//  printf("A d_i is NULL, updating!\n");

			hypre_CSRMatrixDeviceI(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_rows + 1, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->i,(A->num_rows+1)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_i,A->i,
					(A->num_rows+1)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_j == NULL)) {
			// printf("A d_j is NULL, updating\n");

			hypre_CSRMatrixDeviceJ(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->j,(A->num_nonzeros)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_j,A->j,
					(A->num_nonzeros)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();

		}
		if ((xddata == NULL)) printf("x (input) data is NULL\n");
		if ((yddata == NULL)) printf("y (output) data is NULL\n");
		MatvecCSRAsynch(A->num_rows-offset,
				alpha,
				A->d_data ,
				A->d_i,
				A->d_j,
				&xddata[k1*x_size], beta, &yddata[k2*y_size]);
	}
#endif


	return 0;
}


// Two in One, experimental kernel
	HYPRE_Int
hypre_CSRMatrixMatvecMultAsynchTwoInOneDevice( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A1,
		hypre_CSRMatrix *A2,
		hypre_Vector    *x1,
		HYPRE_Int k11,
		hypre_Vector    *x2,
		HYPRE_Int k12,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int offset )
{ 
	HYPRE_Int         x_size = hypre_VectorSize(x1);
	HYPRE_Int         y_size = hypre_VectorSize(y);
	HYPRE_Complex * x1ddata =  x1->d_data;
	HYPRE_Complex * x2ddata =  x2->d_data;
	HYPRE_Complex * yddata =  y->d_data;



#if 1
	if (A1->num_nonzeros){
		if ((A1->d_data == NULL)) {
			//printf("no offdiag data\n ");
			cudaMemPrefetchAsync(A1->data,A1->num_nonzeros*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4));

			hypre_CSRMatrixDeviceData(A1)    = hypre_CTAlloc(HYPRE_Complex,  A1->num_nonzeros, HYPRE_MEMORY_DEVICE);
			cudaDeviceSynchronize();

			cudaMemcpy ( A1->d_data,A1->data,
					A1->num_nonzeros*sizeof(HYPRE_Complex),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A1->d_i == NULL)) {

			//printf("no offdiag i\n ");
			hypre_CSRMatrixDeviceI(A1)    = hypre_CTAlloc(HYPRE_Int,  A1->num_rows + 1, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A1->i,(A1->num_rows+1)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A1->d_i,A1->i,
					(A1->num_rows+1)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A1->d_j == NULL)) {
			//printf("no offdiag j\n ");
			// printf("A d_j is NULL, updating\n");

			hypre_CSRMatrixDeviceJ(A1)    = hypre_CTAlloc(HYPRE_Int,  A1->num_nonzeros, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A1->j,(A1->num_nonzeros)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A1->d_j,A1->j,
					(A1->num_nonzeros)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();

		}
	}
	else{
		//printf("OFFDIAG EMPTY\n ");
	}
	if (A2->num_nonzeros){
		if ((A2->d_data == NULL)) {
			//printf("no diag data\n ");
			cudaMemPrefetchAsync(A2->data,A2->num_nonzeros*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4));

			hypre_CSRMatrixDeviceData(A2)    = hypre_CTAlloc(HYPRE_Complex,  A2->num_nonzeros, HYPRE_MEMORY_DEVICE);
			cudaDeviceSynchronize();

			cudaMemcpy ( A2->d_data,A2->data,
					A2->num_nonzeros*sizeof(HYPRE_Complex),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A2->d_i == NULL)) {

			//printf("no diag i\n ");
			hypre_CSRMatrixDeviceI(A2)    = hypre_CTAlloc(HYPRE_Int,  A2->num_rows + 1, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A2->i,(A2->num_rows+1)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A2->d_i,A2->i,
					(A2->num_rows+1)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A2->d_j == NULL)) {
			//printf("no diag j\n ");
			// printf("A d_j is NULL, updating\n");

			hypre_CSRMatrixDeviceJ(A2)    = hypre_CTAlloc(HYPRE_Int,  A2->num_nonzeros, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A2->j,(A2->num_nonzeros)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A2->d_j,A2->j,
					(A2->num_nonzeros)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();

		}
	}
	else{ 
		//printf("DIAG EMPTY\n ");
	}
	if ((A1->num_nonzeros) || (A2->num_nonzeros)){
		MatvecCSRAsynchTwoInOne(A1->num_rows-offset,
				alpha,
				A1->d_data ,
				A1->d_i,
				A1->d_j,
				A2->d_data ,
				A2->d_i,
				A2->d_j,
				&x1ddata[k11*x_size], 	&x2ddata[k12*x_size],beta, &yddata[k2*y_size]);
	}
	else{
		//printf("both diag and non diag EmPTY\n");
	}

#endif


	return 0;
}
	HYPRE_Int
hypre_CSRMatrixMatvecMultDevice( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int offset )
{
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         y_size = hypre_VectorSize(y);
	HYPRE_Complex * xddata =  x->d_data;
	HYPRE_Complex * yddata =  y->d_data;

	//printf("TEST TEST TEST k1 = %d x_size = %d \n", k1, x_size);
	static cusparseHandle_t handle;
	static cusparseMatDescr_t descr;
	static HYPRE_Int FirstCall=1;
	cusparseStatus_t status;
	static cudaStream_t s[10];
	static HYPRE_Int myid;

	if (b!=y){
		PUSH_RANGE_PAYLOAD("MEMCPY",1,y->size-offset);
		//    VecCopy(y->data,b->data,(y->size-offset),HYPRE_STREAM(4));
   hypre_SeqVectorCopyOneOfMult(b,k3, y, k2);
	
	//HYPRE_Complex * bddata =  b->d_data;
//	cudaMemcpy( yddata, bddata, y->size, cudaMemcpyDeviceToDevice);
	POP_RANGE
	}
	// if (x==y) fprintf(stderr,"MULTIVEC ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");

	if (FirstCall){
		PUSH_RANGE("FIRST_CALL",4);

		handle=getCusparseHandle();

		status= cusparseCreateMatDescr(&descr);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			exit(2);
		}

		cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
		FirstCall=0;
		hypre_int jj;
		for(jj=0;jj<5;jj++)
			s[jj]=HYPRE_STREAM(jj);
		nvtxNameCudaStreamA(s[4], "HYPRE_COMPUTE_STREAM");
		hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
		myid++;
		POP_RANGE;
	}

	//cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice);
	PUSH_RANGE("PREFETCH+SPMV",2);

	if (offset!=0) printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: %d \n",offset);
	// printf("k1 = %d k2 = %d this is matvec. num rows in A %d , num cols in A %d  nnz in A %d alpha = %f beta = %f\n",k1, k2,A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta );
#if 1
	if (A->num_nonzeros){
		// printf("k1 = %d k2 = %d this is matvec. num rows in A %d , num cols in A %d  nnz in A %d alpha = %f beta = %f\n",k1, k2,A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta );
		if ((A->d_data == NULL)) {
			//    printf("A: d_data is NULL; updating!\n");
			//hypre_SeqVectorPrefetchToDevice(A->data);
			//cudaMemPrefetchAsync(ptr,size,device,stream);
			// cudaError_t err; 
			cudaMemPrefetchAsync(A->data,A->num_nonzeros*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4));
			//printf("error code in prefetch %d string %s \n", err, cudaGetErrorString(err));

			hypre_CSRMatrixDeviceData(A)    = hypre_CTAlloc(HYPRE_Complex,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_data,A->data,
					A->num_nonzeros*sizeof(HYPRE_Complex),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_i == NULL)) {
			//  printf("A d_i is NULL, updating!\n");

			hypre_CSRMatrixDeviceI(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_rows + 1, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->i,(A->num_rows+1)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_i,A->i,
					(A->num_rows+1)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_j == NULL)) {
			// printf("A d_j is NULL, updating\n");

			hypre_CSRMatrixDeviceJ(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->j,(A->num_nonzeros)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_j,A->j,
					(A->num_nonzeros)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();

		}
		if ((xddata == NULL)) printf("x (input) data is NULL\n");
		if ((yddata == NULL)) printf("y (output) data is NULL\n");

		status = cusparseDcsrmv(handle ,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				A->num_rows-offset, A->num_cols, A->num_nonzeros,
				&alpha, descr,
				A->d_data ,A->d_i,A->d_j,
				&xddata[k1*x_size], &beta, &yddata[k2*y_size]);
	}
#endif
	POP_RANGE;

	return 0;
}

	HYPRE_Int
hypre_CSRMatrixMatvecTMultDevice( HYPRE_Complex    alpha,
		hypre_CSRMatrix *A,
		hypre_Vector    *x,
		HYPRE_Int k1,
		HYPRE_Complex    beta,
		hypre_Vector    *b,
		HYPRE_Int k3,
		hypre_Vector    *y,
		HYPRE_Int k2,
		HYPRE_Int offset )
{
	HYPRE_Int         x_size = hypre_VectorSize(x);
	HYPRE_Int         y_size = hypre_VectorSize(y);
	HYPRE_Complex * xddata =  x->d_data;
	HYPRE_Complex * yddata =  y->d_data;

	//printf("TEST TEST TEST k1 = %d x_size = %d \n", k1, x_size);
	static cusparseHandle_t handle;
	static cusparseMatDescr_t descr;
	static HYPRE_Int FirstCall=1;
	cusparseStatus_t status;
	static cudaStream_t s[10];
	static HYPRE_Int myid;

	if (b!=y){
		PUSH_RANGE_PAYLOAD("MEMCPY",1,y->size-offset);
		//    VecCopy(y->data,b->data,(y->size-offset),HYPRE_STREAM(4));
   hypre_SeqVectorCopyOneOfMult(b,k3, y, k2);
	
	//HYPRE_Complex * bddata =  b->d_data;
//	cudaMemcpy( yddata, bddata, y->size, cudaMemcpyDeviceToDevice);
	POP_RANGE
	}
	// if (x==y) fprintf(stderr,"MULTIVEC ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");

	if (FirstCall){
		PUSH_RANGE("FIRST_CALL",4);

		handle=getCusparseHandle();

		status= cusparseCreateMatDescr(&descr);
		if (status != CUSPARSE_STATUS_SUCCESS) {
			exit(2);
		}

		cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
		FirstCall=0;
		hypre_int jj;
		for(jj=0;jj<5;jj++)
			s[jj]=HYPRE_STREAM(jj);
		nvtxNameCudaStreamA(s[4], "HYPRE_COMPUTE_STREAM");
		hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid );
		myid++;
		POP_RANGE;
	}

	//cudaMemcpy( dst, src, size, cudaMemcpyHostToDevice);
	PUSH_RANGE("PREFETCH+SPMV",2);

	if (offset!=0) printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: %d \n",offset);
	// printf("k1 = %d k2 = %d this is matvec. num rows in A %d , num cols in A %d  nnz in A %d alpha = %f beta = %f\n",k1, k2,A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta );
#if 1
	if (A->num_nonzeros){
		// printf("k1 = %d k2 = %d this is matvec. num rows in A %d , num cols in A %d  nnz in A %d alpha = %f beta = %f\n",k1, k2,A->num_rows, A->num_cols, A->num_nonzeros, alpha, beta );
		if ((A->d_data == NULL)) {
			    printf("A: d_data is NULL; updating!\n");
			//hypre_SeqVectorPrefetchToDevice(A->data);
			//cudaMemPrefetchAsync(ptr,size,device,stream);
			// cudaError_t err; 
			cudaMemPrefetchAsync(A->data,A->num_nonzeros*sizeof(HYPRE_Complex),HYPRE_DEVICE,HYPRE_STREAM(4));
			//printf("error code in prefetch %d string %s \n", err, cudaGetErrorString(err));

			hypre_CSRMatrixDeviceData(A)    = hypre_CTAlloc(HYPRE_Complex,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_data,A->data,
					A->num_nonzeros*sizeof(HYPRE_Complex),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_i == NULL)) {
			  printf("A d_i is NULL, updating!\n");

			hypre_CSRMatrixDeviceI(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_rows + 1, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->i,(A->num_rows+1)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_i,A->i,
					(A->num_rows+1)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();
		}
		if ((A->d_j == NULL)) {
		 printf("A d_j is NULL, updating\n");

			hypre_CSRMatrixDeviceJ(A)    = hypre_CTAlloc(HYPRE_Int,  A->num_nonzeros, HYPRE_MEMORY_DEVICE);

			cudaMemPrefetchAsync(A->j,(A->num_nonzeros)*sizeof(HYPRE_Int),HYPRE_DEVICE,HYPRE_STREAM(4));
			cudaDeviceSynchronize();

			cudaMemcpy ( A->d_j,A->j,
					(A->num_nonzeros)*sizeof(HYPRE_Int),
					cudaMemcpyDeviceToDevice );
			cudaDeviceSynchronize();

		}
		if ((xddata == NULL)) printf("x (input) data is NULL\n");
		if ((yddata == NULL)) printf("y (output) data is NULL\n");
printf("inside matvecT, before GPU call, num rows %d num cols %d nnz %d size x %d size y %d \n",  A->num_rows,  A->num_cols,  A->num_nonzeros, x_size, y_size);
#if 0
		status = cusparseDcsrmv(handle ,
				CUSPARSE_OPERATION_TRANSPOSE,
				A->num_rows-offset, A->num_cols, A->num_nonzeros,
				&alpha, descr,
				A->d_data ,A->d_i,A->d_j,
				&xddata[k1*x_size], &beta, &yddata[k2*y_size]);
#endif
	}
#endif
	POP_RANGE;

	return 0;
}
#endif

