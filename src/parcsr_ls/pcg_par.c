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

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCAlloc
 *--------------------------------------------------------------------------*/

  void *
hypre_ParKrylovCAlloc( HYPRE_Int count,
    HYPRE_Int elt_size )
{
  return( (void*) hypre_CTAlloc( char, count * elt_size , HYPRE_MEMORY_HOST) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovFree
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovFree( void *ptr )
{
  HYPRE_Int ierr = 0;

  hypre_Free( ptr , HYPRE_MEMORY_HOST);

  return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVector
 *--------------------------------------------------------------------------*/

  void *
hypre_ParKrylovCreateVector( void *vvector )
{
  hypre_ParVector *vector = (hypre_ParVector *) vvector;
  hypre_ParVector *new_vector;

  new_vector = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
      hypre_ParVectorGlobalSize(vector),	
      hypre_ParVectorPartitioning(vector) );
  hypre_ParVectorSetPartitioningOwner(new_vector,0);
  hypre_ParVectorInitialize(new_vector);

  return ( (void *) new_vector );
}


//for multivectors AKA Krylov space (stored as one cont chunk of memory)

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateMultiVector
 *--------------------------------------------------------------------------*/

  void *
hypre_ParKrylovCreateMultiVector( void *vvector, HYPRE_Int num_vectors)
{
  hypre_ParVector *vector = (hypre_ParVector *) vvector;
  hypre_ParVector *new_vector;

  new_vector = hypre_ParMultiVectorCreate( hypre_ParVectorComm(vector),
      hypre_ParVectorGlobalSize(vector),	
      hypre_ParVectorPartitioning(vector), num_vectors );
  hypre_ParVectorSetPartitioningOwner(new_vector,0);
  hypre_ParVectorInitialize(new_vector);

  return ( (void *) new_vector );
}
//update CPU if needed (important if the result lives on the CPU)

void *
hypre_ParKrylovUpdateVectorCPU( void *vvector){

  return((void *) (hypre_ParVectorCopyDataGPUtoCPU((hypre_ParVector *) vvector)));


}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVectorArray
 * Note: one array will be allocated for all vectors, with vector 0 owning
 * the data, vector i will have data[i*size] assigned, not owning data
 *--------------------------------------------------------------------------*/

  void *
hypre_ParKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
  hypre_ParVector *vector = (hypre_ParVector *) vvector;

  hypre_ParVector **new_vector;
  HYPRE_Int i, size;
  HYPRE_Complex *array_data;

  size = hypre_VectorSize(hypre_ParVectorLocalVector(vector));
  array_data = hypre_CTAlloc(HYPRE_Complex, (n*size), HYPRE_MEMORY_SHARED);
  new_vector = hypre_CTAlloc(hypre_ParVector*, n, HYPRE_MEMORY_HOST);
  for (i=0; i < n; i++)
  {
    new_vector[i] = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
	hypre_ParVectorGlobalSize(vector),	
	hypre_ParVectorPartitioning(vector) );
    hypre_ParVectorSetPartitioningOwner(new_vector[i],0);
    hypre_VectorData(hypre_ParVectorLocalVector(new_vector[i])) = &array_data[i*size];
    hypre_ParVectorInitialize(new_vector[i]);
    if (i) hypre_VectorOwnsData(hypre_ParVectorLocalVector(new_vector[i]))=0;
    hypre_ParVectorActualLocalSize(new_vector[i]) = size;
  }

  return ( (void *) new_vector );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovDestroyVector
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovDestroyVector( void *vvector )
{
  hypre_ParVector *vector = (hypre_ParVector *) vvector;

  return( hypre_ParVectorDestroy( vector ) );
}



  void *
hypre_ParKrylovGetAuxVector( void   *A,HYPRE_Int id )
{
  
  hypre_ParCSRMatrix * AA = (hypre_ParCSRMatrix *) A;
if (id ==0)
return AA->w;
else return AA->w_2;
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/


  void *
hypre_ParKrylovMatvecCreate( void   *A,
    void   *x )
{
  void *matvec_data;

  matvec_data = NULL;
//new


#if !defined(HYPRE_USING_UNIFIED_MEMORY) && defined(HYPRE_USING_GPU) 
  hypre_ParCSRMatrix * AA = (hypre_ParCSRMatrix *) A;

  hypre_CSRMatrix   *offd   = hypre_ParCSRMatrixOffd(AA);

  HYPRE_Int          num_cols_offd = hypre_CSRMatrixNumCols(offd);

  hypre_ParCSRCommPkg *comm_pkg = hypre_ParCSRMatrixCommPkg(AA);

  if (!comm_pkg)
  {
    hypre_MatvecCommPkgCreate(AA);
    comm_pkg = hypre_ParCSRMatrixCommPkg(AA);
  }

  HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
  AA->x_tmp = hypre_SeqVectorCreate( num_cols_offd );

  hypre_SeqVectorInitialize(AA->x_tmp);

  AA->x_buf = hypre_CTAlloc(HYPRE_Complex,  hypre_ParCSRCommPkgSendMapStart
      (comm_pkg,  num_sends), HYPRE_MEMORY_DEVICE);
//round 2 of new code
//

 HYPRE_Int begin = hypre_ParCSRCommPkgSendMapStart(comm_pkg, 0);
 HYPRE_Int end   = hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends);
  AA->comm_d =  hypre_CTAlloc(HYPRE_Int,  (end-begin), HYPRE_MEMORY_DEVICE);;
if ((end-begin) != 0)
{
  cudaMemcpy(AA->comm_d,hypre_ParCSRCommPkgSendMapElmts(comm_pkg),  (end-begin) * sizeof(HYPRE_Int),cudaMemcpyHostToDevice  );
}
if (AA->w == NULL) {
AA->w =  hypre_ParVectorCreate(hypre_ParCSRMatrixComm(AA),
      hypre_ParCSRMatrixGlobalNumRows(AA),
      hypre_ParCSRMatrixRowStarts(AA));
hypre_ParVectorInitialize(AA->w);
} 
if (AA->w_2 == NULL) {
AA->w_2 =  hypre_ParVectorCreate(hypre_ParCSRMatrixComm(AA),
      hypre_ParCSRMatrixGlobalNumRows(AA),
      hypre_ParCSRMatrixRowStarts(AA));
hypre_ParVectorInitialize(AA->w_2);
} 
#endif



//endof new




  return ( matvec_data );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvec
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovMatvec( void   *matvec_data,
    HYPRE_Complex  alpha,
    void   *A,
    void   *x,
    HYPRE_Complex  beta,
    void   *y           )
{
  return ( hypre_ParCSRMatrixMatvec ( alpha,
	(hypre_ParCSRMatrix *) A,
	(hypre_ParVector *) x,
	beta,
	(hypre_ParVector *) y ) );
}


/*Matvec for multivectors i.e. y(:, k2) = A*x(:, k1) */
        
      
  HYPRE_Int
hypre_ParKrylovMatvecMult( void   *matvec_data,
    HYPRE_Complex  alpha,
    void   *A,
    void   *x,
    HYPRE_Int k1,
    HYPRE_Complex  beta,
    void   *y, HYPRE_Int k2           )
{

    hypre_CheckErrorDevice(cudaPeekAtLastError());
  return ( hypre_ParCSRMatrixMatvecMult ( alpha,
        (hypre_ParCSRMatrix *) A,
        (hypre_ParVector *) x,k1,
        beta,
        (hypre_ParVector *) y, k2 ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecT
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovMatvecT(void   *matvec_data,
    HYPRE_Complex  alpha,
    void   *A,
    void   *x,
    HYPRE_Complex  beta,
    void   *y           )
{
  return ( hypre_ParCSRMatrixMatvecT( alpha,
	(hypre_ParCSRMatrix *) A,
	(hypre_ParVector *) x,
	beta,
	(hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecDestroy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovMatvecDestroy( void *matvec_data )
{
  return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovInnerProd
 *--------------------------------------------------------------------------*/

  HYPRE_Real
hypre_ParKrylovInnerProd( void *x, 
    void *y )
{
  return ( hypre_ParVectorInnerProd( (hypre_ParVector *) x,
	(hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovInnerProdOneOfMult
 *--------------------------------------------------------------------------*/

  HYPRE_Real 
hypre_ParKrylovInnerProdOneOfMult( void *x, HYPRE_Int k1,
    void *y, HYPRE_Int k2 )
{
  return ( hypre_ParVectorInnerProdOneOfMult( (hypre_ParVector *) x,k1,
	(hypre_ParVector *) y, k2 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovDoubleInnerProdOneOfMult
 *--------------------------------------------------------------------------*/

  HYPRE_Int 
hypre_ParKrylovDoubleInnerProdOneOfMult( void *x, HYPRE_Int k1,
    void *y, HYPRE_Int k2, void *res )
{
  return ( hypre_ParVectorDoubleInnerProdOneOfMult( (hypre_ParVector *) x,k1,
	(hypre_ParVector *) y, k2, (HYPRE_Real*) res ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovAxpyOneOfMult
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovAxpyOneOfMult( HYPRE_Complex alpha,void *x, HYPRE_Int k1,
    void *y, HYPRE_Int k2 )
{
  return ( hypre_ParVectorAxpyOneOfMult( alpha, (hypre_ParVector *) x,k1,
	(hypre_ParVector *) y, k2 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProdMult // written by KS //for multivectors
 * x is the space, y is the single vector 
 *--------------------------------------------------------------------------*/
 HYPRE_Int
hypre_ParKrylovMassInnerProdMult( void *x,HYPRE_Int k,
    void *y, HYPRE_Int k2, void  * result )
{
// void HYPRE_ParVectorMassInnerProdMult ( HYPRE_ParVector x , HYPRE_Int k, HYPRE_ParVector y , HYPRE_Int k2, HYPRE_Real *prod );
 ( hypre_ParVectorMassInnerProdMult( (hypre_ParVector *) x,
 k,
(hypre_ParVector *) y,
k2 ,
(HYPRE_Real*)result ));
return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProdTwoVectorsMult // written by KS //for multivectors
 * x is the space, y1 and y2 are  single vectors 
 *--------------------------------------------------------------------------*/
 HYPRE_Int
hypre_ParKrylovMassInnerProdTwoVectorsMult( void *x,HYPRE_Int k,
    void *y1, HYPRE_Int k2, void * y2, HYPRE_Int k3, void  * result )
{
// void HYPRE_ParVectorMassInnerProdMult ( HYPRE_ParVector x , HYPRE_Int k, HYPRE_ParVector y , HYPRE_Int k2, HYPRE_Real *prod );
 ( hypre_ParVectorMassInnerProdTwoVectorsMult( (hypre_ParVector *) x,
 k,
(hypre_ParVector *) y1,
k2 ,
(hypre_ParVector *) y2,
k3 ,
(HYPRE_Real*)result ));
return 0;
}
/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProd 
 *--------------------------------------------------------------------------*/
  HYPRE_Int
hypre_ParKrylovMassInnerProd( void *x, 
    void **y, HYPRE_Int k, HYPRE_Int unroll, void  * result )
{
  return ( hypre_ParVectorMassInnerProd( (hypre_ParVector *) x,(hypre_ParVector **) y, k, unroll, (HYPRE_Real*)result ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassDotpTwo
 *--------------------------------------------------------------------------*/
  HYPRE_Int
hypre_ParKrylovMassDotpTwo( void *x, void *y, 
    void **z, HYPRE_Int k, HYPRE_Int unroll, void  *result_x, void *result_y )
{
  return ( hypre_ParVectorMassDotpTwo( (hypre_ParVector *) x, (hypre_ParVector *) y, (hypre_ParVector **) z, k, 
	unroll, (HYPRE_Real *)result_x, (HYPRE_Real *)result_y ) );
}



/*--------------------------------------------------------------------------
 * hypre_ParKrylovCopyVector
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovCopyVector( void *x, 
    void *y )
{
  return ( hypre_ParVectorCopy( (hypre_ParVector *) x,
	(hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovCopyVectorOneOfMult
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovCopyVectorOneOfMult( void *x, HYPRE_Int k1,
    void *y, HYPRE_Int k2 )
{
  return ( hypre_ParVectorCopyOneOfMult( (hypre_ParVector *) x,k1,
	(hypre_ParVector *) y, k2 ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovClearVector
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovClearVector( void *x )
{
  return ( hypre_ParVectorSetConstantValues( (hypre_ParVector *) x, 0.0 ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovScaleVector
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovScaleVector( HYPRE_Complex  alpha,
    void   *x     )
{
  return ( hypre_ParVectorScale( alpha, (hypre_ParVector *) x ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovScaleVectorOneOfMult
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovScaleVectorOneOfMult( HYPRE_Complex  alpha,
    void   *x, HYPRE_Int k1     )
{
  //printf("scale \n");
  return ( hypre_ParVectorScaleOneOfMult( alpha, (hypre_ParVector *) x, k1 ) );
}
/*--------------------------------------------------------------------------
 * hypre_ParKrylovAxpy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovAxpy( HYPRE_Complex alpha,
    void   *x,
    void   *y )
{
  return ( hypre_ParVectorAxpy( alpha, (hypre_ParVector *) x,
	(hypre_ParVector *) y ) );
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassAxpy
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovMassAxpy( HYPRE_Complex * alpha,
    void   **x,
    void   *y ,
    HYPRE_Int k, HYPRE_Int unroll)
{
  return ( hypre_ParVectorMassAxpy( alpha, (hypre_ParVector **) x,
	(hypre_ParVector *) y ,  k, unroll));
}
/*===
 *
 * For space rotatiuons
 * */


HYPRE_Int hypre_ParKrylovGivensRotRight(
     HYPRE_Int k1,
    HYPRE_Int k2,
    void * q1,
    void  * q2,
    HYPRE_Real  a1, HYPRE_Real a2, HYPRE_Real a3, HYPRE_Real a4){
    hypre_ParVectorGivensRotRight(k1, k2, (hypre_ParVector *)q1, (hypre_ParVector *)q2, a1,a2,a3,a4);
return 0;
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassAxpyMult (for multivectors, x is a multivector)
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParKrylovMassAxpyMult( HYPRE_Real * alpha,
    void   *x,
    HYPRE_Int k,
    void   *y ,
    HYPRE_Int k2){
    hypre_ParVectorMassAxpyMult( alpha, (hypre_ParVector *) x, k,
	(hypre_ParVector *) y ,  k2);
return 0;
}
/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProdWithScalingMult // written by KS //for multivectors
 * x is the space, y is the single vector 
 *--------------------------------------------------------------------------*/
  HYPRE_Int
hypre_ParKrylovMassInnerProdWithScalingMult( void *x,HYPRE_Int k,
    void *y, HYPRE_Int k2, void *scaleFactors,  void  * result )
{
 ( hypre_ParVectorMassInnerProdWithScalingMult( (hypre_ParVector *) x,
	k,
	(hypre_ParVector *) y,
	k2 ,
	(HYPRE_Real *) scaleFactors,
	(HYPRE_Real*)result ) );
return 0;
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovCommInfo
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovCommInfo( void   *A, HYPRE_Int *my_id, HYPRE_Int *num_procs)
{
  MPI_Comm comm = hypre_ParCSRMatrixComm ( (hypre_ParCSRMatrix *) A);
  hypre_MPI_Comm_size(comm,num_procs);
  hypre_MPI_Comm_rank(comm,my_id);
  return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentitySetup
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovIdentitySetup( void *vdata,
    void *A,
    void *b,
    void *x     )

{
  return 0;
}

/*--------------------------------------------------------------------------
 * hypre_ParKrylovIdentity
 *--------------------------------------------------------------------------*/

  HYPRE_Int
hypre_ParKrylovIdentity( void *vdata,
    void *A,
    void *b,
    void *x     )

{
  return( hypre_ParKrylovCopyVector( b, x ) );
}

