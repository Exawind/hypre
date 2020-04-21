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

//gou should be set here(???)
//KS code
////end of KS code

  hypre_COGMRESFunctions * cogmres_functions;

  if (!solver)
  {
    hypre_error_in_arg(2);
    return hypre_error_flag;
  }
  cogmres_functions =
    hypre_COGMRESFunctionsCreate(
	hypre_CAlloc, hypre_ParKrylovFree, 
	hypre_ParKrylovCommInfo,
	hypre_ParKrylovCreateVector,
	hypre_ParKrylovCreateMultiVector,
	hypre_ParKrylovUpdateVectorCPU,
	hypre_ParKrylovDestroyVector, 
	hypre_ParKrylovMatvecCreate,
	hypre_ParKrylovMatvecMult, 
	hypre_ParKrylovMatvecDestroy,
	hypre_ParKrylovInnerProdOneOfMult, 
	hypre_ParKrylovMassInnerProdMult, 
	hypre_ParKrylovMassInnerProdTwoVectorsMult, 
	hypre_ParKrylovMassInnerProdWithScalingMult,    
	hypre_ParKrylovDoubleInnerProdOneOfMult, 
	hypre_ParKrylovCopyVectorOneOfMult,
	//hypre_ParKrylovCopyVector,
	hypre_ParKrylovClearVector,
	hypre_ParKrylovScaleVectorOneOfMult, 
	hypre_ParKrylovAxpyOneOfMult,
	hypre_ParKrylovMassAxpyMult,
  hypre_ParKrylovGetAuxVector,
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
  HYPRE_Int ret =  HYPRE_COGMRESSolve( solver,
      (HYPRE_Matrix) A,
      (HYPRE_Vector) b,
      (HYPRE_Vector) x ) ;

  //update GPU version if necessary
#if defined(HYPRE_USING_GPU) && !defined(HYPRE_USING_UNIFIED_MEMORY)
  hypre_ParVectorCopyDataGPUtoCPU(x);
#endif
  return ret;
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
 * HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

  HYPRE_Int
HYPRE_ParCSRCOGMRESSetUnroll( HYPRE_Solver solver,
    HYPRE_Int             unroll    )
{
  return( HYPRE_COGMRESSetUnroll( solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

  HYPRE_Int
HYPRE_ParCSRCOGMRESSetCGS( HYPRE_Solver solver,
    HYPRE_Int             cgs    )
{
  return( HYPRE_COGMRESSetCGS( solver, cgs ) );
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

HYPRE_Int HYPRE_ParCSRCOGMRESSetGSoption( HYPRE_Solver solver,
    HYPRE_Int GSoption)
{
  return( HYPRE_COGMRESSetGSoption( solver, GSoption ) );
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

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetResidual
 *--------------------------------------------------------------------------*/

  HYPRE_Int
HYPRE_ParCSRCOGMRESGetResidual( HYPRE_Solver  solver,
    HYPRE_ParVector *residual)
{
  return( HYPRE_COGMRESGetResidual( solver, (void *) residual ) );
}
