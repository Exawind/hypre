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
 * Header info for CSR Matrix data structures
 *
 * Note: this matrix currently uses 0-based indexing.
 *
 *****************************************************************************/

#ifndef hypre_CSR_MATRIX_HEADER
#define hypre_CSR_MATRIX_HEADER

/*--------------------------------------------------------------------------
 * CSR Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int     *i;
   HYPRE_Int     *j;
   HYPRE_Int      num_rows;
   HYPRE_Int      num_cols;
   HYPRE_Int      num_nonzeros;

   /* Does the CSRMatrix create/destroy `data', `i', `j'? */
   HYPRE_Int      owns_data;

   HYPRE_Complex  *data;

   /* for compressing rows in matrix multiplication  */
   HYPRE_Int     *rownnz;
   HYPRE_Int      num_rownnz;

 /* KS: for GPU storage - only used when unified memory not used */
   HYPRE_Int *d_i, *d_j, *d_rownnz;
   HYPRE_Complex *d_data;
 HYPRE_Int copied;
   HYPRE_Int hostOnly=0;
HYPRE_Complex * r_local_data, *v_local_data, *y_local_data;  
} hypre_CSRMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRMatrixData(matrix)         ((matrix) -> data)
#define hypre_CSRMatrixI(matrix)            ((matrix) -> i)
#define hypre_CSRMatrixJ(matrix)            ((matrix) -> j)
#define hypre_CSRMatrixNumRows(matrix)      ((matrix) -> num_rows)
#define hypre_CSRMatrixNumCols(matrix)      ((matrix) -> num_cols)
#define hypre_CSRMatrixNumNonzeros(matrix)  ((matrix) -> num_nonzeros)
#define hypre_CSRMatrixRownnz(matrix)       ((matrix) -> rownnz)
#define hypre_CSRMatrixNumRownnz(matrix)    ((matrix) -> num_rownnz)
#define hypre_CSRMatrixOwnsData(matrix)     ((matrix) -> owns_data)

//by KS
#define hypre_CSRMatrixDeviceData(matrix)         ((matrix) -> d_data)
#define hypre_CSRMatrixDeviceI(matrix)            ((matrix) -> d_i)
#define hypre_CSRMatrixDeviceJ(matrix)            ((matrix) -> d_j)

#define hypre_CSRMatrixHostOnly(matrix)         ((matrix) -> hostOnly)

#define hypre_CSRMatrixHasBeenCopied(matrix)    ((matrix) -> copied)

#define hypre_CSRMatrixRdata(matrix)            ((matrix) -> r_local_data)
#define hypre_CSRMatrixYdata(matrix)            ((matrix) -> y_local_data)
#define hypre_CSRMatrixVdata(matrix)            ((matrix) -> v_local_data)


HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionBegin( hypre_CSRMatrix *A );
HYPRE_Int hypre_CSRMatrixGetLoadBalancedPartitionEnd( hypre_CSRMatrix *A );

/*--------------------------------------------------------------------------
 * CSR Boolean Matrix
 *--------------------------------------------------------------------------*/

typedef struct
{
   HYPRE_Int    *i;
   HYPRE_Int    *j;
   HYPRE_Int     num_rows;
   HYPRE_Int     num_cols;
   HYPRE_Int     num_nonzeros;
   HYPRE_Int     owns_data;

} hypre_CSRBooleanMatrix;

/*--------------------------------------------------------------------------
 * Accessor functions for the CSR Boolean Matrix structure
 *--------------------------------------------------------------------------*/

#define hypre_CSRBooleanMatrix_Get_I(matrix)        ((matrix)->i)
#define hypre_CSRBooleanMatrix_Get_J(matrix)        ((matrix)->j)
#define hypre_CSRBooleanMatrix_Get_NRows(matrix)    ((matrix)->num_rows)
#define hypre_CSRBooleanMatrix_Get_NCols(matrix)    ((matrix)->num_cols)
#define hypre_CSRBooleanMatrix_Get_NNZ(matrix)      ((matrix)->num_nonzeros)
#define hypre_CSRBooleanMatrix_Get_OwnsData(matrix) ((matrix)->owns_data)

#endif

