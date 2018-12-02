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
#include "_hypre_parcsr_mv.h"

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
//printf("global size %d \n", hypre_ParVectorGlobalSize(vector));
	new_vector = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
			hypre_ParVectorGlobalSize(vector),	
			hypre_ParVectorPartitioning(vector) );
	hypre_ParVectorSetPartitioningOwner(new_vector,0);
	hypre_ParVectorInitialize(new_vector);

	return ( (void *) new_vector );
}

HYPRE_Int hypre_ParKrylovVectorSize(void *vvector)
{
	hypre_ParVector *vector = (hypre_ParVector *) vvector;
	return vector->global_size;

}
/*--------------------------------------------------------------------------
 * hypre_ParKrylovCreateVectorArray
 *--------------------------------------------------------------------------*/

	void *
hypre_ParKrylovCreateVectorArray(HYPRE_Int n, void *vvector )
{
	hypre_ParVector *vector = (hypre_ParVector *) vvector;
	hypre_ParVector **new_vector;
	HYPRE_Int i;

	new_vector = hypre_CTAlloc(hypre_ParVector*, n, HYPRE_MEMORY_HOST);
	for (i=0; i < n; i++)
	{
		new_vector[i] = hypre_ParVectorCreate( hypre_ParVectorComm(vector),
				hypre_ParVectorGlobalSize(vector),	
				hypre_ParVectorPartitioning(vector) );
		hypre_ParVectorSetPartitioningOwner(new_vector[i],0);
		hypre_ParVectorInitialize(new_vector[i]);
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

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMatvecCreate
 *--------------------------------------------------------------------------*/

	void *
hypre_ParKrylovMatvecCreate( void   *A,
		void   *x )
{
	void *matvec_data;

	matvec_data = NULL;

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

//printf("ParKrylovMatvec \n");
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

//  printf("ParKrylovMatvec Multivectors\n");
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
	void 
hypre_ParKrylovMassInnerProdMult( void *x,HYPRE_Int k, 
		void *y, HYPRE_Int k2, void  * result )
{
// void HYPRE_ParVectorMassInnerProdMult ( HYPRE_ParVector x , HYPRE_Int k, HYPRE_ParVector y , HYPRE_Int k2, HYPRE_Real *prod );
	return ( hypre_ParVectorMassInnerProdMult( (hypre_ParVector *) x, 
 k, 
(hypre_ParVector *) y,
k2 , 
(HYPRE_Real*)result ) );
}

//version with scaling (custom)
//

/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProdWithScalingMult // written by KS //for multivectors
 * x is the space, y is the single vector 
 *--------------------------------------------------------------------------*/
	void 
hypre_ParKrylovMassInnerProdWithScalingMult( void *x,HYPRE_Int k, 
		void *y, HYPRE_Int k2, void *scaleFactors,  void  * result )
{
	return ( hypre_ParVectorMassInnerProdWithScalingMult( (hypre_ParVector *) x, 
 k, 
(hypre_ParVector *) y,
k2 ,
(HYPRE_Real *) scaleFactors,  
(HYPRE_Real*)result ) );
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassInnerProd // written by KS
 *--------------------------------------------------------------------------*/
	void 
hypre_ParKrylovMassInnerProd( void *x, 
		void **y, int k, void  * result )
{
	return ( hypre_ParVectorMassInnerProd( (hypre_ParVector *) x,(hypre_ParVector **) y, k, (HYPRE_Real*)result ) );
}

//	void hypre_ParVectorMassInnerProdGPU ( HYPRE_Real * x , HYPRE_Real *y , HYPRE_Int k, HYPRE_Int n,HYPRE_Real *prod );
	void 
hypre_ParKrylovMassInnerProdGPU( void *x, 
		void *y, int k, int n, void  * result )
{

	return hypre_ParVectorMassInnerProdGPU( (HYPRE_Real*) x,(HYPRE_Real*) y, k, n, (HYPRE_Real*)result); 
//return chujemuje( (HYPRE_Real*) x,(HYPRE_Real*) y, k, n, (HYPRE_Real*)result); 


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
printf("ParKrylov copy \n");
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
printf("scale \n");
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

void
hypre_ParKrylovMassAxpy( HYPRE_Real * alpha,
		void   **x,
		void   *y ,
		HYPRE_Int k){
	return ( hypre_ParVectorMassAxpy( alpha, (hypre_ParVector **) x,
				(hypre_ParVector *) y ,  k));
}


/*--------------------------------------------------------------------------
 * hypre_ParKrylovMassAxpyMult (for multivectors, x is a multivector)
 *--------------------------------------------------------------------------*/

void
hypre_ParKrylovMassAxpyMult( HYPRE_Real * alpha,
		void   *x,
HYPRE_Int k,
		void   *y ,
		HYPRE_Int k2){
	return ( hypre_ParVectorMassAxpyMult( alpha, (hypre_ParVector *) x, k,
				(hypre_ParVector *) y ,  k2));
}




//for super optimized GPU version that does not use hypre-vectors.
void
hypre_ParKrylovMassAxpyGPU( HYPRE_Real * alpha,
		void   *x,
		void   *y ,
		HYPRE_Int k, HYPRE_Int n){
	//return ( hypre_ParVectorMassAxpyGPU( alpha, (HYPRE_Real *)  x,
	//                              (HYPRE_Real *) y ,  k, n));
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

