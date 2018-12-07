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
 * implemented in NREL for ExaWind project
 * communication optimal GMRES requires less synchronizations 
 *
 *
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * COGMRES cogmres
 *
 *****************************************************************************/

#include "krylov.h"
#include "_hypre_utilities.h"
#include "_hypre_parcsr_ls.h"
#ifdef HYPRE_USE_GPU
#include "../seq_mv/gpukernels.h"
#endif
/*--------------------------------------------------------------------------
 * hypre_COGMRESFunctionsCreate
 *--------------------------------------------------------------------------*/

	hypre_COGMRESFunctions *
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
{
	hypre_COGMRESFunctions * cogmres_functions;
	cogmres_functions = (hypre_COGMRESFunctions *)
		CAlloc( 1, sizeof(hypre_COGMRESFunctions), HYPRE_MEMORY_HOST );

	cogmres_functions->CAlloc            = CAlloc;
	cogmres_functions->Free              = Free;
	cogmres_functions->CommInfo          = CommInfo; /* not in PCGFunctionsCreate */
	cogmres_functions->CreateVector      = CreateVector;
	cogmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
	cogmres_functions->DestroyVector     = DestroyVector;
	cogmres_functions->MatvecCreate      = MatvecCreate;
	cogmres_functions->Matvec            = Matvec;
	cogmres_functions->MatvecDestroy     = MatvecDestroy;
	cogmres_functions->InnerProd         = InnerProd;
	cogmres_functions->MassInnerProd     = MassInnerProd;
	cogmres_functions->CopyVector        = CopyVector;
	cogmres_functions->ClearVector       = ClearVector;
	cogmres_functions->ScaleVector       = ScaleVector;
	cogmres_functions->Axpy              = Axpy;
	cogmres_functions->MassAxpy          = MassAxpy;

	cogmres_functions->VectorSize        = VectorSize;
	/* default preconditioner must be set here but can be changed later... */
	cogmres_functions->precond_setup     = PrecondSetup;
	cogmres_functions->precond           = Precond;

	return cogmres_functions;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESCreate
 *--------------------------------------------------------------------------*/

	void *
hypre_COGMRESCreate( hypre_COGMRESFunctions *cogmres_functions )
{
	hypre_COGMRESData *cogmres_data;

	cogmres_data = hypre_CTAllocF(hypre_COGMRESData, 1, cogmres_functions, HYPRE_MEMORY_HOST);

	cogmres_data->functions = cogmres_functions;

	/* set defaults */
	(cogmres_data -> k_dim)          = 5;
	(cogmres_data -> tol)            = 1.0e-06; /* relative residual tol */
	(cogmres_data -> cf_tol)         = 0.0;
	(cogmres_data -> a_tol)          = 0.0; /* abs. residual tol */
	(cogmres_data -> min_iter)       = 0;
	(cogmres_data -> max_iter)       = 1000;
	(cogmres_data -> rel_change)     = 0;
	(cogmres_data -> skip_real_r_check) = 0;
	(cogmres_data -> stop_crit)      = 0; /* rel. residual norm  - this is obsolete!*/
	(cogmres_data -> converged)      = 0;
	(cogmres_data -> precond_data)   = NULL;
	(cogmres_data -> print_level)    = 0;
	(cogmres_data -> logging)        = 0;
	(cogmres_data -> p)              = NULL;
	(cogmres_data -> r)              = NULL;
	(cogmres_data -> w)              = NULL;
	(cogmres_data -> w_2)            = NULL;
	(cogmres_data -> matvec_data)    = NULL;
	(cogmres_data -> norms)          = NULL;
	(cogmres_data -> log_file_name)  = NULL;

	return (void *) cogmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESDestroy
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESDestroy( void *cogmres_vdata )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
	HYPRE_Int i;

	if (cogmres_data)
	{
		hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
		if ( (cogmres_data->logging>0) || (cogmres_data->print_level) > 0 )
		{
			if ( (cogmres_data -> norms) != NULL )
				hypre_TFreeF( cogmres_data -> norms, cogmres_functions );
		}

		if ( (cogmres_data -> matvec_data) != NULL )
			(*(cogmres_functions->MatvecDestroy))(cogmres_data -> matvec_data);

		if ( (cogmres_data -> r) != NULL )
			(*(cogmres_functions->DestroyVector))(cogmres_data -> r);
		if ( (cogmres_data -> w) != NULL )
			(*(cogmres_functions->DestroyVector))(cogmres_data -> w);
		if ( (cogmres_data -> w_2) != NULL )
			(*(cogmres_functions->DestroyVector))(cogmres_data -> w_2);


		if ( (cogmres_data -> p) != NULL )
		{
			/*for (i = 0; i < (cogmres_data -> k_dim+1); i++)
				{
				if ( (cogmres_data -> p)[i] != NULL )
			//(*(cogmres_functions->DestroyVector))( (cogmres_data -> p) [i]);
			}
			hypre_TFreeF( cogmres_data->p, cogmres_functions );
			*/
			cudaFree(cogmres_data -> p);
		}
		hypre_TFreeF( cogmres_data, cogmres_functions );
		hypre_TFreeF( cogmres_functions, cogmres_functions );
	}

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int hypre_COGMRESGetResidual( void *cogmres_vdata, void **residual )
{
	/* returns a pointer to the residual vector */

	hypre_COGMRESData  *cogmres_data     = (hypre_COGMRESData *)cogmres_vdata;
	*residual = cogmres_data->r;
	return hypre_error_flag;

}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetup
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetup( void *cogmres_vdata,
		void *A,
		void *b,
		void *x         )
{
	hypre_COGMRESData *cogmres_data     = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;

	HYPRE_Int k_dim                                     = (cogmres_data -> k_dim);
	HYPRE_Int max_iter                                  = (cogmres_data -> max_iter);
	HYPRE_Int (*precond_setup)(void*,void*,void*,void*) = (cogmres_functions->precond_setup);
	void       *precond_data                            = (cogmres_data -> precond_data);
	HYPRE_Int rel_change                                = (cogmres_data -> rel_change);



	(cogmres_data -> A) = A;

	/*--------------------------------------------------
	 * The arguments for NewVector are important to
	 * maintain consistency between the setup and
	 * compute phases of matvec and the preconditioner.
	 *--------------------------------------------------*/

	if ((cogmres_data -> r) == NULL)
		(cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);
//	if ((cogmres_data -> w) == NULL)
	//	(cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
	//  if ((cogmres_data -> p) == NULL)
	//  (cogmres_data -> p) = (void**)(*(cogmres_functions->CreateVectorArray))(k_dim+1,x);

	if (rel_change)
	{  
	//	if ((cogmres_data -> w_2) == NULL)
		//	(cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
	}


	if ((cogmres_data -> matvec_data) == NULL)
		(cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);

	precond_setup(precond_data, A, b, x);

	/*-----------------------------------------------------
	 * Allocate space for log info
	 *-----------------------------------------------------*/

	if ( (cogmres_data->logging)>0 || (cogmres_data->print_level) > 0 )
	{
		if ((cogmres_data -> norms) == NULL)
			(cogmres_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1,cogmres_functions, HYPRE_MEMORY_HOST);
	}
	if ( (cogmres_data->print_level) > 0 ) {
		if ((cogmres_data -> log_file_name) == NULL)
			(cogmres_data -> log_file_name) = (char*)"cogmres.out.log";
	}

	return hypre_error_flag;
}
HYPRE_Int idx(HYPRE_Int r, HYPRE_Int c, HYPRE_Int n){
	//n is the # el IN THE COLUMN
	return c*n+r;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSolve
 *-------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSolve(void  *cogmres_vdata,
		void  *A,
		void  *b,
		void  *x)
{
	double time1, time2, time3, time4;

	time1                                     = MPI_Wtime(); 
	hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
	HYPRE_Int               k_dim             = (cogmres_data -> k_dim);
	HYPRE_Int               min_iter          = (cogmres_data -> min_iter);
	HYPRE_Int               max_iter          = (cogmres_data -> max_iter);
	HYPRE_Int               rel_change        = (cogmres_data -> rel_change);
	HYPRE_Int               skip_real_r_check = (cogmres_data -> skip_real_r_check);
	HYPRE_Real              r_tol             = (cogmres_data -> tol);
	HYPRE_Real              cf_tol            = (cogmres_data -> cf_tol);
	HYPRE_Real              a_tol             = (cogmres_data -> a_tol);
	void                   *matvec_data       = (cogmres_data -> matvec_data);

	void                        *r            = (cogmres_data -> r);
	hypre_ParVector             *w            = (hypre_ParVector *) (cogmres_data -> w);
	/* note: w_2 is only allocated if rel_change = 1 */
	void             *w_2          = (cogmres_data -> w_2); 

	HYPRE_Real            *p            = (HYPRE_Real*) (cogmres_data -> p);


	HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
	HYPRE_Int  *precond_data                      = (HYPRE_Int*)(cogmres_data -> precond_data);

	HYPRE_Int print_level = (cogmres_data -> print_level);
	HYPRE_Int logging     = (cogmres_data -> logging);

	HYPRE_Real     *norms          = (cogmres_data -> norms);
	/* not used yet   char           *log_file_name  = (cogmres_data -> log_file_name);*/
	/*   FILE           *fp; */

	HYPRE_Int  break_value = 0;
	HYPRE_Int  i, j, k;
	/*KS: rv is the norm history */
	HYPRE_Real *rs, *hh, *c, *s, *rs_2, *rv;
	HYPRE_Real *x_GPUonly, *b_GPUonly, *r_GPUonly;
	//, *tmp; 
	HYPRE_Int  iter; 
	HYPRE_Int  my_id, num_procs;
	HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
	HYPRE_Real w_norm;

	HYPRE_Real epsmac = 1.e-16; 
	HYPRE_Real ieee_check = 0.;

	HYPRE_Real guard_zero_residual; 
	HYPRE_Real cf_ave_0 = 0.0;
	HYPRE_Real cf_ave_1 = 0.0;
	HYPRE_Real weight;
	HYPRE_Real r_norm_0;
	HYPRE_Real relative_error = 1.0;

	HYPRE_Int        rel_change_passed = 0, num_rel_change_check = 0;

	HYPRE_Real real_r_norm_old, real_r_norm_new;

	(cogmres_data -> converged) = 0;
	/*-----------------------------------------------------------------------
	 * With relative change convergence test on, it is possible to attempt
	 * another iteration with a zero residual. This causes the parameter
	 * alpha to go NaN. The guard_zero_residual parameter is to circumvent
	 * this. Perhaps it should be set to something non-zero (but small).
	 *-----------------------------------------------------------------------*/
	guard_zero_residual = 0.0;

	(*(cogmres_functions->CommInfo))(A,&my_id,&num_procs);
	if ( logging>0 || print_level>0 )
	{
		norms          = (cogmres_data -> norms);
	}

	/* initialize work arrays */
	rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	c  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	s  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	//tmp  = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	if (rel_change) rs_2 = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST); 


	rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_functions, HYPRE_MEMORY_HOST);
	//KS copy matrix to GPU once!!!
	//
	/*
		 cusparseErrchk(cusparseDcsrmv(handle ,
		 CUSPARSE_OPERATION_NON_TRANSPOSE,
		 A->num_rows-offset, A->num_cols, A->num_nonzeros,
		 &alpha, descr,
		 A->data ,A->i+offset,A->j,
		 x->data, &beta, y->data+offset));

	 *
	 * */

//hypre_CSRMatrixPrefetchToDevice((hypre_CSRMatrix *)   A);
	hypre_CSRMatrix * AA = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A);
//hypre_CSRMatrixPrefetchToDevice(A);
	//   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(AA);
	//   HYPRE_Int        *A_i      = hypre_CSRMatrixI(AA);
	//   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(AA);
	HYPRE_Int         num_rows = AA->num_rows;
	HYPRE_Int         num_cols = AA->num_cols;
	HYPRE_Int         num_nonzeros = AA->num_nonzeros;
	HYPRE_Real * A_dataGPUonly;
	HYPRE_Int * A_iGPUonly, *A_jGPUonly;
	//allocate
	cudaMalloc ( &A_dataGPUonly, num_nonzeros*sizeof(HYPRE_Real)); 
	cudaMalloc ( &A_jGPUonly, num_nonzeros*sizeof(HYPRE_Int)); 
	cudaMalloc ( &A_iGPUonly, (num_rows+1)*sizeof(HYPRE_Int)); 
	//KS new
int err;
	err = cudaMemcpy (A_dataGPUonly,AA->data, 
			num_nonzeros*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice );
//printf("one: value %d \n", err);
	cudaMemcpy (A_iGPUonly,AA->i, 
			(num_rows+1)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 
//printf("two: value %d \n", err);
	cudaMemcpy (A_jGPUonly,AA->j, 
			(num_nonzeros)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 

//printf("three: value %d \n", err);

	HYPRE_Int sz;
	sz = (*(cogmres_functions->VectorSize))(r);
	HYPRE_Real * tempH;
	//  if ((cogmres_drc/parcsr_ls/HYPRE_parcsr_cogmres.ca -> p) == NULL)


	cudaMalloc ( &cogmres_data -> p, sz*sizeof(HYPRE_Real)*(k_dim+1)); 

	cudaMalloc ( &tempH, sizeof(HYPRE_Real)*(k_dim+1)); 
	p = (HYPRE_Real*) cogmres_data->p;
	HYPRE_Real * tempV;
	cudaMalloc ( &tempV, sizeof(HYPRE_Real)*(sz)); 


	//hh = hypre_CTAllocF(HYPRE_Real*,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	//for (i=0; i < k_dim+1; i++)
	//{ 
	//hh[i] = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	//}
	//
	hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
	//cudaMallocManaged ( hh, sizeof(HYPRE_Real)*(k_dim+1)*k_dim); 
	//	hypre_ParVector * bb;

	//	bb = (hypre_ParVector*) (*(cogmres_functions->CreateVector))(b);

	HYPRE_Real *bbtemp, *wtemp;

	hypre_ParVector * cc =  (hypre_ParVector*) b;

	hypre_ParVector * xx =  (hypre_ParVector*) x;

	cudaMalloc ( &b_GPUonly, sizeof(HYPRE_Real)*(sz)); 
	cudaMemcpy (b_GPUonly,
			cc->local_vector->data, 
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 


	cudaMalloc ( &x_GPUonly, sizeof(HYPRE_Real)*(sz)); 
	cudaMemcpy (x_GPUonly,
			xx->local_vector->data, 
			(sz)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 

	cudaMalloc ( &r_GPUonly, sizeof(HYPRE_Real)*(sz)); 
	cudaMalloc ( &bbtemp, sizeof(HYPRE_Real)*(sz)); 

	cudaMalloc ( &wtemp, sizeof(HYPRE_Real)*(sz)); 
	cudaMemcpy (p,b_GPUonly, 
			(sz)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 

	cusparseHandle_t myHandle;
	cusparseMatDescr_t myDescr;
	cusparseCreate(&myHandle);

	cusparseCreateMatDescr(&myDescr); 
	cusparseSetMatType(myDescr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(myDescr,CUSPARSE_INDEX_BASE_ZERO);

	/* compute initial residual */
	//bb->local_vector->data = &p[0];
	// source, dest
	//(*(gmres_functions->CopyVector))(b,p[0]);
	//	(*(cogmres_functions->CopyVector))(b,bb);

	//	(*(cogmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, bb);
	double one = 1.0f, minusone = -1.0f, zero = 0.0f;
	cusparseDcsrmv(myHandle ,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			num_rows, num_cols, num_nonzeros,
			&minusone, myDescr,
			A_dataGPUonly,A_iGPUonly,A_jGPUonly,
			x_GPUonly, &one, p);

	HYPRE_Int testError;
	//	testError = cudaMemcpy (&p[0],bb->local_vector->data, 
	//		sz*sizeof(HYPRE_Real),
	//	cudaMemcpyDeviceToDevice ); 

	//	b_norm = sqrt((*(cogmres_functions->InnerProd))(b,b));
	InnerProdGPUonly(p,  
			p, 
			&b_norm, 
			sz);
	b_norm = sqrt(b_norm);
	real_r_norm_old = b_norm;

	/* Since it is does not diminish performance, attempt to return an error flag
		 and notify users when they supply bad input. */
	if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
	if (ieee_check != ieee_check)
	{
		/* ...INFs or NaNs in input can make ieee_check a NaN.  This test
			 for ieee_check self-equality works on all IEEE-compliant compilers/
				InnerProdGPUonly(&p[i*sz],  
				InnerProdGPUonly(&p[i*sz],  
						&p[i*sz], 
						&rv[i], 
						sz);
						&p[i*sz], 
						&rv[i], 
						sz);
			 machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
			 by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
			 found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
		if (logging > 0 || print_level > 0)
		{
			hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
			hypre_printf("ERROR -- hypre_COGMRESSolve: INFs and/or NaNs detected in input.\n");
			hypre_printf("User probably placed non-numerics in supplied b.\n");
			hypre_printf("Returning error flag += 101.  Program not terminated.\n");
			hypre_printf("ERROR detected by Hypre ... END\n\n\n");
		}
		hypre_error(HYPRE_ERROR_GENERIC);
		return hypre_error_flag;
	}

	//b->local_vectpr->  
	//bb->local_vector->data = &p[0];
	//at this point bb and p[0] are tje same
	//	r_norm   = sqrt((*(cogmres_functions->InnerProd))(bb,bb));
	InnerProdGPUonly(p,  
			p, 
			&r_norm, 
			sz);
r_norm = sqrt(r_norm);
printf("INITIAL r norm (after matvec) %f \n", r_norm);
	r_norm_0 = r_norm;
	/* Since it is does not diminish performance, attempt to return an error flag
		 and notify users when they supply bad input. */
	if (r_norm != 0.) ieee_check = r_norm/r_norm; /* INF -> NaN conversion */
	if (ieee_check != ieee_check)
	{
		/* ...INFs or NaNs in input can make ieee_check a NaN.  This test
			 for ieee_check self-equality works on all IEEE-compliant compilers/
			 machines, c.f. page 8 of "Lecture Notes on the Status of IEEE 754"
			 by W. Kahan, May 31, 1996.  Currently (July 2002) this paper may be
			 found at http://HTTP.CS.Berkeley.EDU/~wkahan/ieee754status/IEEE754.PDF */
		if (logging > 0 || print_level > 0)
		{
			hypre_printf("\n\nERROR detected by Hypre ... BEGIN\n");
			hypre_printf("ERROR -- hypre_COGMRESSolve: INFs and/or NaNs detected in input.\n");
			hypre_printf("User probably placed non-numerics in supplied A or x_0.\n");
			hypre_printf("Returning error flag += 101.  Program not terminated.\n");
			hypre_printf("ERROR detected by Hypre ... END\n\n\n");
		}
		hypre_error(HYPRE_ERROR_GENERIC);
		return hypre_error_flag;
	}

	if ( logging>0 || print_level > 0)
	{
		norms[0] = r_norm;
		if ( print_level>1 && my_id == 0 )
		{

			hypre_printf("L2 norm of b: %e\n", b_norm);
			if (b_norm == 0.0)
				hypre_printf("Rel_resid_norm actually contains the residual norm\n");
			hypre_printf("Initial L2 norm of residual: %e\n", r_norm);

		}
	}
	iter = 0;

	if (b_norm > 0.0)
	{
		/* convergence criterion |r_i|/|b| <= accuracy if |b| > 0 */
		den_norm= b_norm;
	}
	else
	{
		/* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
		den_norm= r_norm;
	};


	/* convergence criteria: |r_i| <= max( a_tol, r_tol * den_norm)
		 den_norm = |r_0| or |b|
note: default for a_tol is 0.0, so relative residual criteria is used unless
user specifies a_tol, or sets r_tol = 0.0, which means absolute
tol only is checked  */

	epsilon = hypre_max(a_tol,r_tol*den_norm);

	/* so now our stop criteria is |r_i| <= epsilon */

	if ( print_level>1 && my_id == 0 )
	{
		if (b_norm > 0.0)
		{hypre_printf("=============================================\n\n");
			hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
			hypre_printf("-----    ------------    ---------- ------------\n");

		}

		else
		{hypre_printf("=============================================\n\n");
			hypre_printf("Iters     resid.norm     conv.rate\n");
			hypre_printf("-----    ------------    ----------\n");

		};
	}


	/* once the rel. change check has passed, we do not want to check it again */
	rel_change_passed = 0;

	time2 = MPI_Wtime();
	if (my_id == 0){
		hypre_printf("GMRES INIT TIME: %16.16f", time2-time1); 
	} 
	/* outer iteration cycle */
	double gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
	double massAxpyTime =0.0; 
	double massIPTime = 0.0f, preconTime = 0.0f, mvTime = 0.0f;    
	while (iter < max_iter)
	{
		/* initialize first term of hessenberg system */
		time1 = MPI_Wtime();

		rs[0] = r_norm;
		rv[0] = 1.0;
		if (r_norm == 0.0)
		{
printf("here 1\n");
			hypre_TFreeF(c,cogmres_functions); 
			hypre_TFreeF(s,cogmres_functions); 
			hypre_TFreeF(rs,cogmres_functions);
			if (rel_change)  hypre_TFreeF(rs_2,cogmres_functions);
			//for (i=0; i < k_dim+1; i++) hypre_TFreeF(hh[i],cogmres_functions);
			hypre_TFreeF(hh,cogmres_functions); 
			return hypre_error_flag;

		}

		/* see if we are already converged and 
			 should print the final norm and exit */

		PUSH_RANGE("COGMRES_RES_CALC", 1);
		if (r_norm  <= epsilon && iter >= min_iter) 
		{
printf("here 2\n");
			if (!rel_change) /* shouldn't exit after no iterations if
												* relative change is on*/
			{

printf("here 3\n");
				//	(*(cogmres_functions->CopyVector))(b,r);	
				//  (*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
				//	r_norm = sqrt((*(cogmres_functions->InnerProd))(r,r));
				cudaMemcpy (r_GPUonly, b_GPUonly,
						sz*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice );

				cusparseErrchk(cusparseDcsrmv(myHandle ,
							CUSPARSE_OPERATION_NON_TRANSPOSE,
							num_rows, num_cols, num_nonzeros,
							&minusone, myDescr,
							A_dataGPUonly,A_iGPUonly,A_jGPUonly,
							x_GPUonly, &one, r_GPUonly));

	InnerProdGPUonly(r_GPUonly,  
						r_GPUonly, 
						&r_norm, 
						sz);
     r_norm = sqrt(r_norm);	
			if (r_norm  <= epsilon)
				{

printf("here 4\n");
					if ( print_level>1 && my_id == 0)
					{

printf("here 5\n");
						hypre_printf("\n\n");
						hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
					}
					break;
				}
				else
					if ( print_level>0 && my_id == 0)
						hypre_printf("false convergence 1\n");
			}
		}
		POP_RANGE;

printf("\n initial res %f \n", r_norm );

		t = 1.0 / r_norm;
  //  cudaDeviceSynchronize();
		// (*(cogmres_functions->ScaleVector))(t,bb);
		//	testError = cudaMemcpy (&p[0],bb->local_vector->data, 
		//		sz*sizeof(HYPRE_Real),
		//	cudaMemcpyDeviceToDevice ); 
		printf("scaling by %f \n", t);	

		double wtf;
		InnerProdGPUonly(&p[0],  
			&p[0], 
			&wtf, 
			sz);

		printf("before  scaling , norm %f \n", sqrt(wtf) );
			
		ScaleGPUonly(p, 
				t, 
				sz);
		 InnerProdGPUonly(p,  
			 p, 
			 &wtf, 
			 sz);

			 printf("after scaling , norm %f \n", sqrt( wtf ));
		i = 0;
		time2 = MPI_Wtime();
		remainingTime += (time2-time1);
		/***RESTART CYCLE (right-preconditioning) ***/
		/*

			 void ScaleGPUonly(double * __restrict__ u, 
			 const double alpha, 
			 const int N);
			 void AxpyGPUonly(const double * __restrict__ u,  
			 double * __restrict__ v,
			 const double alpha, 
			 const int N); 
			 void InnerProdGPUonly(const double * __restrict__ u,  
			 const double * __restrict__ v, 
			 double result, 
			 const int N);

		 * */		


		while (i < k_dim && iter < max_iter)
		{

			time1 = MPI_Wtime();
			i++;
			iter++;

			//		(*(cogmres_functions->ClearVector))(r);

			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
			time1 = MPI_Wtime();
			PUSH_RANGE("COGMRES_PRECOND", 2);
			//	testError = cudaMemcpy ( bb->local_vector->data, &p[(i-1)*sz], 
			//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice ); 

			//  precond(precond_data, A, &p[(i-1)*sz], r);
			//	precond(precond_data, A, bb, r);
			//printf("norm aftet precond %f, \n", sqrt( (*(cogmres_functions->InnerProd))(r,r) ));

			POP_RANGE;
			time3 = MPI_Wtime();
			preconTime += (time3 - time1);
			PUSH_RANGE("COGMRES_MATVEC1", 3);
		//	(*cogmres_functions->Matvec)(matvec_data, 1.0, A, r, 0.0, bb);
			//printf("after matvec %f, ", sqrt( (*(cogmres_functions->InnerProd))(bb,bb) ));
     
	cusparseDcsrmv(myHandle ,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			num_rows, num_cols, num_nonzeros,
			&one, myDescr,
			A_dataGPUonly,A_iGPUonly,A_jGPUonly,
			&p[(i-1)*sz], &zero, &p[i*sz]);
	
//			t = sqrt( (*(cogmres_functions->InnerProd))(bb,bb) );

	InnerProdGPUonly(&p[i*sz],  
			&p[i*sz], 
			&t, 
			sz);
t = sqrt(t);
printf("initial norm of p[%d] = %f\n", i, t);
//		testError = cudaMemcpy ( (HYPRE_Real*) (&p[(i)*sz]), bb->local_vector->data,
		//			sz*sizeof(HYPRE_Real),
			//		cudaMemcpyDeviceToDevice ); 
			POP_RANGE;
			time2 = MPI_Wtime();
			mvTime += (time2-time3);
			matvecPreconTime += (time2-time1);   
			/** KS & ST: we replace mGS with custom version **/
			//  H(1:i,i) = (2 - vm(1:i)).*(V(:,1:i)'*w);
			// BRUTE FORCE
			//
			//need multiIP here
			/* for (j=0; j<i; j++){
				 hh[j][i-1] = (*(cogmres_functions->InnerProd))(&p[j*sz],&p[i*sz]);
				 hh[j][i-1] = (2-rv[j])*hh[j][i-1];
			//   hypre_printf("BF h[%d][%d] = %16.16f \n", j, i-1, hh[j][i-1]);
			}*/
			// hypre_printf("about to start multi IP \n");  
			//(*(cogmres_functions->MassInnerProd))((void *) p[i], p,(HYPRE_Int) i, tmp);
			PUSH_RANGE("COGMRES_DOTP", 4);
			time1=MPI_Wtime();


			MassInnerProdGPUonly(&p[i*sz],
					p,
					tempH,				
					i,
					sz);
//cudaDeviceSynchronize();
double wtf2;
			cudaMemcpy ( &hh[idx(0, i-1,k_dim+1)],tempH,
					i*sizeof(HYPRE_Real),
					cudaMemcpyDeviceToHost );

				InnerProdGPUonly(&p[i*sz],  
						&p[i*sz], 
						&wtf2, 
						sz);
printf("after mass innter prod, norm is %f\n", sqrt(wtf2));

			POP_RANGE;
			time3 = MPI_Wtime();
			massIPTime += time3-time1;
	///		(*(cogmres_functions->ClearVector))(w);


			HYPRE_Real t2 = 0.0;
			for (j=0; j<i; j++){
				HYPRE_Int id = idx(j, i-1,k_dim+1);
				hh[id]       = (2.0f-rv[j])*hh[id];
				printf("hh[%d] = %f \n",id, hh[id]);		
				t2          += (hh[id]*hh[id]);        
			}

			cudaMemcpy ( tempH,&hh[idx(0, i-1,k_dim+1)],
					i*sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );

			PUSH_RANGE("COGMRES_AXPY", 5);
			//		(*(cogmres_functions->MassAxpy))(&hh[(i-1)*(k_dim+1)],p,w,i,sz);
			//int, int, double const*, double*, double const*

			time4 = MPI_Wtime();
			MassAxpyGPUonly(sz,  i,
					p,				
					&p[sz*i],
					tempH);	
			//		&hh[(i-1)*(k_dim+1)]);

			cudaMemcpy ( &hh[idx(0, i-1,k_dim+1)],tempH,
					i*sizeof(HYPRE_Real),
					cudaMemcpyDeviceToHost );
			POP_RANGE;
			time3 = MPI_Wtime();
			massAxpyTime += time3-time4;      
//			for (j=0; j<i; j++){
	//			HYPRE_Int id = idx(j, i-1,k_dim+1);
		//		hh[id]       = hh[id];
	//		}


			//testError = cudaMemcpy ( tempH,&hh[idx(0, i-1,k_dim+1)],
			//	i*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice );

			PUSH_RANGE("COGMRES_DOTP1", 6);
			t2 = sqrt(t2)*sqrt(rv[i-1]);
//update t

			printf("t = %f t2 = %f \n", t, t2);
			hh[idx(i, i-1,k_dim+1)] = sqrt(t-t2)*sqrt(t2+t);
			POP_RANGE;



			/** KS & ST: end of our code **/

			//		(*(cogmres_functions->ClearVector))(bb);

			//	testError = cudaMemcpy ( bb->local_vector->data,&p[i*sz],
			//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice );

			PUSH_RANGE("COGMRES_GS", 1);
			if (hh[idx(i,i-1,k_dim+1)] != 0.0)
			{

printf("here 6\n");
				t = 1.0/hh[idx(i,i-1,k_dim+1)];
				// (*(cogmres_functions->ScaleVector))(t,bb);

				InnerProdGPUonly(&p[i*sz],  
						&p[i*sz], 
						&rv[i], 
						sz);


printf("scaling p[%d] by %f, true norm %f, should scale by %f \n",i, t, sqrt(rv[i]), 1.0/sqrt(rv[i]));			
	ScaleGPUonly(&p[sz*i], 
						t, 
						sz);

				//testError = cudaMemcpy (&p[sz*i], bb->local_vector->data,
				//	sz*sizeof(HYPRE_Real),
				//	cudaMemcpyDeviceToDevice ); 


				//  rv[i] = sqrt( (*(cogmres_functions->InnerProd))(bb, bb) );
				InnerProdGPUonly(&p[i*sz],  
						&p[i*sz], 
						&rv[i], 
						sz);
rv[i] = sqrt(rv[i]);			
	printf("AFTER  NORM IS %f \n", rv[i]);

			}

			//	testError = cudaMemcpy (&p[(i)*sz],bb->local_vector->data , 
			//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice ); 


			POP_RANGE;
			time2 = MPI_Wtime();
			gsTime += (time2-time1);
			/* done with modified Gram_schmidt and Arnoldi step.
				 update factorization of hh */
			time1 = MPI_Wtime();
			for (j = 1; j < i; j++)
			{
				t = hh[idx(j-1,i-1,k_dim+1)];
				hh[idx(j-1,i-1,k_dim+1)] = s[j-1]*hh[idx(j,i-1,k_dim+1)] + c[j-1]*t;
				hh[idx(j,i-1, k_dim+1)]  = -s[j-1]*t + c[j-1]*hh[idx(j,i-1,k_dim+1)];

			}
			t     = hh[idx(i, i-1, k_dim+1)]*hh[idx(i,i-1, k_dim+1)];
			t    += hh[idx(i-1,i-1, k_dim+1)]*hh[idx(i-1,i-1, k_dim+1)];
			gamma = sqrt(t);
			if (gamma == 0.0) {gamma = epsmac;

printf("here 7\n");
}

			c[i-1]  = hh[idx(i-1,i-1, k_dim+1)]/gamma;
			s[i-1]  = hh[idx(i,i-1, k_dim+1)]/gamma;
			rs[i]   = -hh[idx(i,i-1, k_dim+1)]*rs[i-1];
			rs[i]  /= gamma;
			rs[i-1] = c[i-1]*rs[i-1];
			// determine residual norm 
			hh[idx(i-1,i-1, k_dim+1)] = s[i-1]*hh[idx(i,i-1, k_dim+1)] + c[i-1]*hh[idx(i-1,i-1, k_dim+1)];
			r_norm = fabs(rs[i]);
			printf("r_norm %f \n", r_norm);	
			time2 = MPI_Wtime();
			linSolveTime  += (time2-time1);
			/* print ? */
			time1 = MPI_Wtime();  
			if ( print_level>0 )
			{
				norms[iter] = r_norm;
				if ( print_level>1 && my_id == 0 )
				{
					if (b_norm > 0.0)
						hypre_printf("% 5d    %e    %f   %e\n", iter, 
								norms[iter],norms[iter]/norms[iter-1],
								norms[iter]/b_norm);
					else
						hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
								norms[iter]/norms[iter-1]);
				}
			}
			/*convergence factor tolerance */
			if (cf_tol > 0.0)
			{

printf("here 8\n");
				cf_ave_0 = cf_ave_1;
				cf_ave_1 = pow( r_norm / r_norm_0, 1.0/(2.0*iter));

				weight = fabs(cf_ave_1 - cf_ave_0);
				weight = weight / hypre_max(cf_ave_1, cf_ave_0);
				weight = 1.0 - weight;
#if 0
				hypre_printf("I = %d: cf_new = %e, cf_old = %e, weight = %e\n",
						i, cf_ave_1, cf_ave_0, weight );
#endif
				if (weight * cf_ave_1 > cf_tol) 
				{

printf("here 9\n");
					break_value = 1;
					break;
				}
			}
			/* should we exit the restart cycle? (conv. check) */
			if (r_norm <= epsilon && iter >= min_iter)
			{

printf("here 10\n");
				if (rel_change && !rel_change_passed)
				{

printf("here 11\n");
					/* To decide whether to break here: to actually
						 determine the relative change requires the approx
						 solution (so a triangular solve) and a
						 precond. solve - so if we have to do this many
						 times, it will be expensive...(unlike cg where is
						 is relatively straightforward)

						 previously, the intent (there was a bug), was to
						 exit the restart cycle based on the residual norm
						 and check the relative change outside the cycle.
						 Here we will check the relative here as we don't
						 want to exit the restart cycle prematurely */

					for (k=0; k<i; k++) /* extra copy of rs so we don't need
																 to change the later solve */
						rs_2[k] = rs[k];

					/* solve tri. system*/
					/*          rs_2[i-1] = rs_2[i-1]/hh[i-1][i-1];
											for (k = i-2; k >= 0; k--)
											{
											t = 0.0;
											for (j = k+1; j < i; j++)
											{
											t -= hh[k][j]*rs_2[j];
											}
											t+= rs_2[k];
											rs_2[k] = t/hh[k][k];
											}*/


					rs_2[i-1] = rs_2[i-1]/hh[idx(i-1,i-1, k_dim+1)];
					for (k = i-2; k >= 0; k--)
					{
						t = 0.0;
						for (j = k+1; j < i; j++)
						{
							t -= hh[idx(k,j, k_dim+1)]*rs_2[j];
						}
						t+= rs_2[k];
						rs_2[k] = t/hh[idx(k,k, k_dim+1)];
					}

					//					testError = cudaMemcpy (w->local_vector->data ,&p[(i-1)*sz], 
					//						sz*sizeof(HYPRE_Real),
					//					cudaMemcpyDeviceToDevice ); 

					//(*(cogmres_functions->ScaleVector))(rs_2[i-1],w);
					ScaleGPUonly(&p[(i-1)*sz],
							rs_2[i-1],
							sz);


					for (j = i-2; j >=0; j--){

						//	testError = cudaMemcpy (bb->local_vector->data,&p[(j)*sz],
						//		sz*sizeof(HYPRE_Real),
						//	cudaMemcpyDeviceToDevice ); 
						//			(*(cogmres_functions->Axpy))(rs_2[j], bb, w);
						AxpyGPUonly(&p[j*sz],  
								&p[(i-1)*sz],
								rs_2[j], 
								sz); 
					}
//					(*(cogmres_functions->ClearVector))(r);
					/* find correction (in r) */
				testError = cudaMemcpy (r_GPUonly ,&p[(i-1)*sz], 
							sz*sizeof(HYPRE_Real),
							cudaMemcpyDeviceToDevice ); 


				//	precond(precond_data, A, w, r);
					/* copy current solution (x) to w (don't want to over-write x)*/
				//	(*(cogmres_functions->CopyVector))(x,w);

					testError = cudaMemcpy (wtemp ,x_GPUonly, 
							sz*sizeof(HYPRE_Real),
							cudaMemcpyDeviceToDevice ); 
					/* add the correction */
			//		(*(cogmres_functions->Axpy))(1.0,r,w);

						AxpyGPUonly(r_GPUonly,  
								wtemp,
								1.0f, 
								sz); 
					/* now w is the approx solution  - get the norm*/
			//		x_norm = sqrt( (*(cogmres_functions->InnerProd))(w,w) );

	InnerProdGPUonly(wtemp,  
			wtemp, 
			&x_norm, 
			sz);
x_norm = sqrt(x_norm);
					if ( !(x_norm <= guard_zero_residual ))
						/* don't divide by zero */
					{  /* now get  x_i - x_i-1 */

printf("here 12\n");
						if (num_rel_change_check)
						{
printf("here 13\n");
							/* have already checked once so we can avoid another precond.
								 solve */
							(*(cogmres_functions->CopyVector))(w, r);
							(*(cogmres_functions->Axpy))(-1.0, w_2, r);
							/* now r contains x_i - x_i-1*/

							/* save current soln w in w_2 for next time */
							(*(cogmres_functions->CopyVector))(w, w_2);
						}
						else
						{
printf("here 14\n");
							/* first time to check rel change*/

							/* first save current soln w in w_2 for next time */
						//	(*(cogmres_functions->CopyVector))(w, w_2);

							/* for relative change take x_(i-1) to be 
								 x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
								 Now
								 x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
								 - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
								 = M^{-1} rs_{i-1}{p_{i-1}} */

						//	(*(cogmres_functions->ClearVector))(w);

	//						testError = cudaMemcpy (bb->local_vector->data,&p[(i-1)*sz],
		//							sz*sizeof(HYPRE_Real),
			//						cudaMemcpyDeviceToDevice ); 
				
							testError = cudaMemcpy (bbtemp,&p[(i-1)*sz],
									sz*sizeof(HYPRE_Real),
									cudaMemcpyDeviceToDevice ); 

		//	(*(cogmres_functions->Axpy))(rs_2[i-1], bb, w);
			
			AxpyGPUonly(bbtemp,r_GPUonly,	
					rs_2[i-1],
					sz);		
		//		(*(cogmres_functions->ClearVector))(r);
							/* apply the preconditioner */
			//				precond(precond_data, A, w, r);
							/* now r contains x_i - x_i-1 */          
						}
						/* find the norm of x_i - x_i-1 */          
						w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
						relative_error = w_norm/x_norm;
						if (relative_error <= r_tol)
						{

printf("here 15\n");
							rel_change_passed = 1;
							break;
						}
					}
					else
					{

printf("here 16\n");
						rel_change_passed = 1;
						break;

					}
					num_rel_change_check++;
				}
				else /* no relative change */
				{

printf("here 17\n");
					break;
				}
			}
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);

		} /*** end of restart cycle ***/
		//error somewhere below here
		/* now compute solution, first solve upper triangular system */
		time1 = MPI_Wtime();
		if (break_value) break;
		rs[i-1] = rs[i-1]/hh[idx(i-1,i-1, k_dim+1)];
		for (k = i-2; k >= 0; k--)
		{
			t = 0.0;
			for (j = k+1; j < i; j++)
			{
				t -= hh[idx(k,j, k_dim+1)]*rs[j];
			}
			t+= rs[k];
			rs[k] = t/hh[idx(k,k, k_dim+1)];
		}


		//		testError = cudaMemcpy (w->local_vector->data,&p[(i-1)*sz],
		//			sz*sizeof(HYPRE_Real),
		//		cudaMemcpyDeviceToDevice ); 
		//	(*(cogmres_functions->ScaleVector))(rs[i-1],w);
		testError = cudaMemcpy (tempV,&p[(i-1)*sz],
				sz*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice ); 
		//	(*(cogmres_functions->ScaleVector))(rs[i-1],w);
		//printf("ONE scaling by %f \n", rs[i-1]);	  
		ScaleGPUonly(tempV,
				rs[i-1],
				sz);	
		for (j = i-2; j >=0; j--){

			//	testError = cudaMemcpy (bb->local_vector->data,&p[(j)*sz],
			//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice ); 
			//	(*(cogmres_functions->Axpy))(rs[j],bb, w);
			//printf("TWO scaling by %f \n", rs[j]);	  

			AxpyGPUonly(&p[j*sz],tempV,	
					rs[j],
					sz);		
		}

//		testError = cudaMemcpy (w->local_vector->data,tempV,
	//			sz*sizeof(HYPRE_Real),
		//p\		cudaMemcpyDeviceToDevice ); 
		testError = cudaMemcpy (r_GPUonly,tempV,
				sz*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice ); 
		testError = cudaMemcpy (wtemp,tempV,
				sz*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice ); 
//		(*(cogmres_functions->ClearVector))(r);
		/* find correction (in r) */
	//	precond(precond_data, A, w, r);
		//printf("THREE, after precond %f, \n", sqrt( (*(cogmres_functions->InnerProd))(r,r) ));

		/* update current solution x (in x) */
//		(*(cogmres_functions->Axpy))(1.0,r,x);

						AxpyGPUonly(r_GPUonly,  
								x_GPUonly,
								1.0f, 
								sz); 
		//printf("FOUR, norm of new x  %f, \n", sqrt( (*(cogmres_functions->InnerProd))(x,x) ));

		/* check for convergence by evaluating the actual residual */
		if (r_norm  <= epsilon && iter >= min_iter)
		{

printf("here 18\n");
			if (skip_real_r_check)
			{
printf("here 19\n");
				(cogmres_data -> converged) = 1;
				break;
			}

			/* calculate actual residual norm*/
	//		(*(cogmres_functions->CopyVector))(b,r);
		//	(*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
		testError = cudaMemcpy (r_GPUonly,b_GPUonly,
				sz*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice ); 
//			real_r_norm_new = r_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
	//MATVEC
	

	cusparseDcsrmv(myHandle ,
			CUSPARSE_OPERATION_NON_TRANSPOSE,
			num_rows, num_cols, num_nonzeros,
			&minusone, myDescr,
			A_dataGPUonly,A_iGPUonly,A_jGPUonly,
			x_GPUonly, &one, r_GPUonly);
	InnerProdGPUonly(r_GPUonly,  
			r_GPUonly, 
			&r_norm, 
			sz);
r_norm = sqrt(r_norm);
 real_r_norm_new = r_norm;
		//printf("FIVE new norm %f \n", real_r_norm_new);
			if (r_norm <= epsilon)
			{
printf("here 20\n");
				if (rel_change && !rel_change_passed) /* calculate the relative change */
				{

printf("here 21\n");
					/* calculate the norm of the solution */
			//		x_norm = sqrt( (*(cogmres_functions->InnerProd))(x,x) );

	InnerProdGPUonly(x_GPUonly,  
			x_GPUonly, 
			&x_norm, 
			sz);
x_norm = sqrt(x_norm);
					if ( !(x_norm <= guard_zero_residual ))
						/* don't divide by zero */
					{

printf("here 22\n");
						/* for relative change take x_(i-1) to be 
							 x + M^{-1}[sum{j=0..i-2} rs_j p_j ]. 
							 Now
							 x_i - x_{i-1}= {x + M^{-1}[sum{j=0..i-1} rs_j p_j ]}
							 - {x + M^{-1}[sum{j=0..i-2} rs_j p_j ]}
							 = M^{-1} rs_{i-1}{p_{i-1}} */
				//		(*(cogmres_functions->ClearVector))(w);

			/*			testError = cudaMemcpy (bb->local_vector->data,&p[(i-1)*sz],
								sz*sizeof(HYPRE_Real),
								cudaMemcpyDeviceToDevice );*/
 
//						(*(cogmres_functions->Axpy))(rs[i-1], bb, w);
	
						AxpyGPUonly(&p[(i-1)*sz],  
								wtemp,
								rs[i-1], 
								sz); 
						testError = cudaMemcpy (r_GPUonly,wtemp,
								sz*sizeof(HYPRE_Real),
								cudaMemcpyDeviceToDevice );
				//	(*(cogmres_functions->ClearVector))(r);
						/* apply the preconditioner */
					//	precond(precond_data, A, w, r);
						/* find the norm of x_i - x_i-1 */          
				
//		w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
	
	InnerProdGPUonly(wtemp,  
			wtemp, 
			&w_norm, 
			sz);
w_norm = sqrt(w_norm);
					relative_error= w_norm/x_norm;
						if ( relative_error < r_tol )
						{
							(cogmres_data -> converged) = 1;
							if ( print_level>1 && my_id == 0 )
							{
								hypre_printf("\n\n");
								hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
							}
							break;
						}
					}
					else
					{

printf("here 23\n");
						(cogmres_data -> converged) = 1;
						if ( print_level>1 && my_id == 0 )
						{
printf("here 24\n");
							hypre_printf("\n\n");
							hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
						}
						break;
					}

				}
				else /* don't need to check rel. change */
				{

printf("here 25\n");
					if ( print_level>1 && my_id == 0 )
					{

						hypre_printf("\n\n");
						hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
					}
					(cogmres_data -> converged) = 1;
					break;
				}
			}
			else /* conv. has not occurred, according to true residual */
			{
				/* exit if the real residual norm has not decreased */
				if (real_r_norm_new >= real_r_norm_old)
				{
printf("here 26\n");
					if (print_level > 1 && my_id == 0)
					{
printf("here 27\n");
						hypre_printf("\n\n");
						hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
					}
					(cogmres_data -> converged) = 1;
					break;
				}

				/* report discrepancy between real/COGMRES residuals and restart */
				if ( print_level>0 && my_id == 0)
					hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
				//			(*(cogmres_functions->CopyVector))(r,&p[0]);
		//		hypre_ParVector * rrr = (hypre_ParVector *)r;
				//printf("SIX, norm of p[0] %f, ", sqrt( (*(cogmres_functions->InnerProd))(r,r) ));

				testError = cudaMemcpy (&p[0],r_GPUonly,
						sz*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 
				i = 0;
				real_r_norm_old = real_r_norm_new;
			}
		} /* end of convergence check */

		/* compute residual vector and continue loop */
		for (j=i ; j > 0; j--)
		{
			rs[j-1] = -s[j-1]*rs[j];
			rs[j] = c[j-1]*rs[j];
		}

		//	testError = cudaMemcpy (bb->local_vector->data,&p[(i)*sz],
		//		sz*sizeof(HYPRE_Real),
		//	cudaMemcpyDeviceToDevice ); 
		if (i)
{
// (*(cogmres_functions->Axpy))(rs[i]-1.0,bb,bb);
	AxpyGPUonly(&p[i*sz],  
			&p[i*sz], 
			rs[i]-1.0, 
			sz);
}
		for (j=i-1 ; j > 0; j--){

//			testError = cudaMemcpy (w->local_vector->data,&p[(j)*sz],
	//				sz*sizeof(HYPRE_Real),
		//			cudaMemcpyDeviceToDevice ); 
		//	(*(cogmres_functions->Axpy))(rs[j],w,bb);
	
					
	AxpyGPUonly(&p[j*sz],  
			&p[i*sz], 
			rs[j], 
			sz);
	}

//		testError = cudaMemcpy (&p[(i)*sz], bb->local_vector->data,
	//			sz*sizeof(HYPRE_Real),
		//		cudaMemcpyDeviceToDevice ); 


	//	testError = cudaMemcpy (w->local_vector->data,p,
		//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice ); 
		if (i)
		{

printf("here 28\n");
/*
 *     (*(gmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
 *                (*(gmres_functions->Axpy))(1.0,p[i],p[0]);
 * */
		//	(*(cogmres_functions->Axpy))(rs[0]-1.0,w,w);
	//		(*(cogmres_functions->Axpy))(1.0,bb,w);
	
	AxpyGPUonly(&p[0],  
			&p[0], 
			rs[0]-1.0, 
			sz);

	AxpyGPUonly(&p[i*sz],  
			&p[0], 
			1.0f, 
			sz);
	}

	//	testError = cudaMemcpy (&p[0], w->local_vector->data,
		//		sz*sizeof(HYPRE_Real),
			//	cudaMemcpyDeviceToDevice ); 
		//printf("THREE and a halfnorm od p[0] %f, \n", sqrt( (*(cogmres_functions->InnerProd))(w,w) ));

		time2 = MPI_Wtime();
		remainingTime += (time2-time1);

	} /* END of iteration while loop */


	if ( print_level>1 && my_id == 0 )
		hypre_printf("\n\n"); 
	if (my_id == 0){
		hypre_printf("TIME for CO-GMRES\n");
		hypre_printf("matvec+precon        = %16.16f \n", matvecPreconTime);
		hypre_printf("gram-schmidt (total) = %16.16f \n", gsTime);
		hypre_printf("linear solve         = %16.16f \n", linSolveTime);
		hypre_printf("all other            = %16.16f \n", remainingTime);
		hypre_printf("FINE times\n");
		hypre_printf("mass Inner Product   = %16.16f \n", massIPTime);
		hypre_printf("mass Axpy            = %16.16f \n", massAxpyTime);
		hypre_printf("precon multiply      = %16.16f \n", preconTime);
		hypre_printf("mv time              = %16.16f \n", mvTime);
	}
	(cogmres_data -> num_iterations) = iter;
	if (b_norm > 0.0)
		(cogmres_data -> rel_residual_norm) = r_norm/b_norm;
	if (b_norm == 0.0)
		(cogmres_data -> rel_residual_norm) = r_norm;

	if (iter >= max_iter && r_norm > epsilon) hypre_error(HYPRE_ERROR_CONV);

	hypre_TFreeF(c,cogmres_functions); 
	hypre_TFreeF(s,cogmres_functions); 
	hypre_TFreeF(rs,cogmres_functions);
	if (rel_change)  hypre_TFreeF(rs_2,cogmres_functions);

	/* for (i=0; i < k_dim+1; i++)
		 {  
		 hypre_TFreeF(hh[i],cogmres_functions);
		 }*/
	// hypre_TFreeF(hh,cogmres_functions); 

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetKDim, hypre_COGMRESGetKDim
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetKDim( void   *cogmres_vdata,
		HYPRE_Int   k_dim )
{
	hypre_COGMRESData *cogmres_data =(hypre_COGMRESData *) cogmres_vdata;


	(cogmres_data -> k_dim) = k_dim;

	return hypre_error_flag;

}

	HYPRE_Int
hypre_COGMRESGetKDim( void   *cogmres_vdata,
		HYPRE_Int * k_dim )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*k_dim = (cogmres_data -> k_dim);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetTol, hypre_COGMRESGetTol
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetTol( void   *cogmres_vdata,
		HYPRE_Real  tol       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> tol) = tol;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetTol( void   *cogmres_vdata,
		HYPRE_Real  * tol      )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*tol = (cogmres_data -> tol);

	return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSetAbsoluteTol, hypre_COGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetAbsoluteTol( void   *cogmres_vdata,
		HYPRE_Real  a_tol       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> a_tol) = a_tol;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetAbsoluteTol( void   *cogmres_vdata,
		HYPRE_Real  * a_tol      )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*a_tol = (cogmres_data -> a_tol);

	return hypre_error_flag;
}
/*--------------------------------------------------------------------------
 * hypre_COGMRESSetConvergenceFactorTol, hypre_COGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetConvergenceFactorTol( void   *cogmres_vdata,
		HYPRE_Real  cf_tol       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> cf_tol) = cf_tol;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetConvergenceFactorTol( void   *cogmres_vdata,
		HYPRE_Real * cf_tol       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*cf_tol = (cogmres_data -> cf_tol);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetMinIter, hypre_COGMRESGetMinIter
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetMinIter( void *cogmres_vdata,
		HYPRE_Int   min_iter  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> min_iter) = min_iter;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetMinIter( void *cogmres_vdata,
		HYPRE_Int * min_iter  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*min_iter = (cogmres_data -> min_iter);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetMaxIter, hypre_COGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetMaxIter( void *cogmres_vdata,
		HYPRE_Int   max_iter  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> max_iter) = max_iter;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetMaxIter( void *cogmres_vdata,
		HYPRE_Int * max_iter  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*max_iter = (cogmres_data -> max_iter);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetRelChange, hypre_COGMRESGetRelChange
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetRelChange( void *cogmres_vdata,
		HYPRE_Int   rel_change  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> rel_change) = rel_change;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetRelChange( void *cogmres_vdata,
		HYPRE_Int * rel_change  )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*rel_change = (cogmres_data -> rel_change);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetSkipRealResidualCheck, hypre_COGMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetSkipRealResidualCheck( void *cogmres_vdata,
		HYPRE_Int skip_real_r_check )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;

	(cogmres_data -> skip_real_r_check) = skip_real_r_check;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetSkipRealResidualCheck( void *cogmres_vdata,
		HYPRE_Int *skip_real_r_check)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;

	*skip_real_r_check = (cogmres_data -> skip_real_r_check);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetStopCrit, hypre_COGMRESGetStopCrit
 *
 *  OBSOLETE 
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetStopCrit( void   *cogmres_vdata,
		HYPRE_Int  stop_crit       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> stop_crit) = stop_crit;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetStopCrit( void   *cogmres_vdata,
		HYPRE_Int * stop_crit       )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*stop_crit = (cogmres_data -> stop_crit);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetPrecond
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetPrecond( void  *cogmres_vdata,
		HYPRE_Int  (*precond)(void*,void*,void*,void*),
		HYPRE_Int  (*precond_setup)(void*,void*,void*,void*),
		void  *precond_data )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;


	(cogmres_functions -> precond)        = precond;
	(cogmres_functions -> precond_setup)  = precond_setup;
	(cogmres_data -> precond_data)   = precond_data;

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetPrecond
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESGetPrecond( void         *cogmres_vdata,
		HYPRE_Solver *precond_data_ptr )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*precond_data_ptr = (HYPRE_Solver)(cogmres_data -> precond_data);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetPrintLevel, hypre_COGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetPrintLevel( void *cogmres_vdata,
		HYPRE_Int   level)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> print_level) = level;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetPrintLevel( void *cogmres_vdata,
		HYPRE_Int * level)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*level = (cogmres_data -> print_level);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetLogging, hypre_COGMRESGetLogging
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetLogging( void *cogmres_vdata,
		HYPRE_Int   level)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> logging) = level;

	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetLogging( void *cogmres_vdata,
		HYPRE_Int * level)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*level = (cogmres_data -> logging);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESGetNumIterations( void *cogmres_vdata,
		HYPRE_Int  *num_iterations )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*num_iterations = (cogmres_data -> num_iterations);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetConverged
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESGetConverged( void *cogmres_vdata,
		HYPRE_Int  *converged )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*converged = (cogmres_data -> converged);

	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESGetFinalRelativeResidualNorm( void   *cogmres_vdata,
		HYPRE_Real *relative_residual_norm )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	*relative_residual_norm = (cogmres_data -> rel_residual_norm);

	return hypre_error_flag;
}


HYPRE_Int hypre_COGMRESSetModifyPC(void *cogmres_vdata, 
		HYPRE_Int (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm))
{

	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;

	(cogmres_functions -> modify_pc)        = modify_pc;

	return hypre_error_flag;
} 

