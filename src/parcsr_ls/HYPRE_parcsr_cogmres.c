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
 * HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
	hypre_COGMRESFunctions * cogmres_functions;

	if (!solver)
	{
		hypre_error_in_arg(2);
		return hypre_error_flag;
	}

/*
  hypre_COGMRESFunctionsCreate(
      void *       (*CAlloc)        ( size_t count, size_t elt_size, HYPRE_Int location ),
      HYPRE_Int    (*Free)          ( void *ptr ),
      HYPRE_Int    (*CommInfo)      ( void  *A, HYPRE_Int   *my_id,
        HYPRE_Int   *num_procs ),
      void *       (*CreateVector)  ( void *vector ),
      void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
        void *x, HYPRE_Complex beta, void *y ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
      void         (*MassInnerProd) (void *x, void *y, int k, int n,  void *result),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
      HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
      void         (*MassAxpy)      ( HYPRE_Real *alpha, void *x, void *y, HYPRE_Int k, HYPRE_Int n),
      HYPRE_Int    (*VectorSize)    (void * vvector),
      HYPRE_Int    (*PrecondSetup)  ( void *vdata, void *A, void *b, void *x ),
      HYPRE_Int    (*Precond)       ( void *vdata, void *A, void *b, void *x )
)

 * */
	cogmres_functions =
		hypre_COGMRESFunctionsCreate(
				hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
				hypre_ParKrylovCreateVector,
				hypre_ParKrylovCreateVectorArray,
				hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
				hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
				hypre_ParKrylovInnerProd,
				hypre_ParKrylovMassInnerProdGPU,  
				hypre_ParKrylovCopyVector,
				hypre_ParKrylovClearVector,
				hypre_ParKrylovScaleVector, 
				hypre_ParKrylovAxpy,
				hypre_ParKrylovMassAxpyGPU,
				hypre_ParKrylovVectorSize,
				hypre_ParKrylovIdentitySetup, 
				hypre_ParKrylovIdentity );
	*solver = ( (HYPRE_Solver) hypre_COGMRESCreate( cogmres_functions ) );

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

	HYPRE_Int 
HYPRE_ParCSRCOGMRESDestroy( HYPRE_Solver solver )
{
	return( hypre_COGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

	HYPRE_Int 
HYPRE_ParCSRCOGMRESSetup( HYPRE_Solver solver,
		HYPRE_ParCSRMatrix A,
		HYPRE_ParVector b,
		HYPRE_ParVector x      )
{
	return( HYPRE_COGMRESSetup( solver,
				(HYPRE_Matrix) A,
				(HYPRE_Vector) b,
				(HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

	HYPRE_Int 
HYPRE_ParCSRCOGMRESSolve( HYPRE_Solver solver,
		HYPRE_ParCSRMatrix A,
		HYPRE_ParVector b,
		HYPRE_ParVector x      )
{
	return( HYPRE_COGMRESSolve( solver,
				(HYPRE_Matrix) A,
				(HYPRE_Vector) b,
				(HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetKDim( HYPRE_Solver solver,
		HYPRE_Int             k_dim    )
{
	return( HYPRE_COGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetTol( HYPRE_Solver solver,
		HYPRE_Real         tol    )
{
	return( HYPRE_COGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetAbsoluteTol( HYPRE_Solver solver,
		HYPRE_Real         a_tol    )
{
	return( HYPRE_COGMRESSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetMinIter( HYPRE_Solver solver,
		HYPRE_Int          min_iter )
{
	return( HYPRE_COGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetMaxIter( HYPRE_Solver solver,
		HYPRE_Int          max_iter )
{
	return( HYPRE_COGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetStopCrit - OBSOLETE
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetStopCrit( HYPRE_Solver solver,
		HYPRE_Int          stop_crit )
{
	return( HYPRE_GMRESSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetPrecond( HYPRE_Solver          solver,
		HYPRE_PtrToParSolverFcn  precond,
		HYPRE_PtrToParSolverFcn  precond_setup,
		HYPRE_Solver          precond_solver )
{
	return( HYPRE_COGMRESSetPrecond( solver,
				(HYPRE_PtrToSolverFcn) precond,
				(HYPRE_PtrToSolverFcn) precond_setup,
				precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESGetPrecond( HYPRE_Solver  solver,
		HYPRE_Solver *precond_data_ptr )
{
	return( HYPRE_COGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetLogging( HYPRE_Solver solver,
		HYPRE_Int logging)
{
	return( HYPRE_COGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESSetPrintLevel( HYPRE_Solver solver,
		HYPRE_Int print_level)
{
	return( HYPRE_COGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESGetNumIterations( HYPRE_Solver  solver,
		HYPRE_Int    *num_iterations )
{
	return( HYPRE_COGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

	HYPRE_Int
HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
		HYPRE_Real   *norm   )
{
	return( HYPRE_COGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}
