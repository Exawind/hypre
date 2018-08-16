//NEW SIMPLIFIED VERSION, WRITTEN BY KS
// AUGUST 2018
//




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
// don't need this but will in the future so let's leave it alone
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


/*-----------------------------------------------------
 * Aux function for Hessenberg matrix storage
 *-----------------------------------------------------*/

HYPRE_Int idx(HYPRE_Int r, HYPRE_Int c, HYPRE_Int n){
	//n is the # el IN THE COLUMN
	return c*n+r;
}


/*--------------------------------------------------------------------------
 * hypre_COGMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int hypre_COGMRESSolve(void  *cogmres_vdata,
		void  *A,
		void  *b,
		void  *x)
{
	HYPRE_Real time1, time2, time3, time4;

	time1                                     = MPI_Wtime(); 
	hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
	HYPRE_Int               k_dim             = (cogmres_data -> k_dim);
	HYPRE_Int               min_iter          = (cogmres_data -> min_iter);
	HYPRE_Int               max_iter          = (cogmres_data -> max_iter);
	HYPRE_Real              r_tol             = (cogmres_data -> tol);
	HYPRE_Real              a_tol             = (cogmres_data -> a_tol);
	void                   *matvec_data       = (cogmres_data -> matvec_data);
	HYPRE_Real            *p            = (HYPRE_Real*) (cogmres_data -> p);


//debug to be removed

  // void             *r            = (cogmres_data -> r);
  // void             *w            = (cogmres_data -> w);
      (cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);
      (cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
      (cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);


	hypre_ParVector * r =  (hypre_ParVector*) (cogmres_data -> r) ;

	hypre_ParVector * w =  (hypre_ParVector*) (cogmres_data -> w) ;
//end of debug

	HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
	HYPRE_Int  *precond_data                      = (HYPRE_Int*)(cogmres_data -> precond_data);

	HYPRE_Real     *norms          = (cogmres_data -> norms);
	HYPRE_Int print_level = (cogmres_data -> print_level);
	HYPRE_Int logging     = (cogmres_data -> logging);

	HYPRE_Int  break_value = 0;
	HYPRE_Int  i, j, k;
	/*KS: rv is the norm history */
	HYPRE_Real *rs, *hh, *c, *s, *rs_2, *rv;
	HYPRE_Real *x_GPUonly, *b_GPUonly, *r_GPUonly;
	HYPRE_Int  iter; 
	HYPRE_Int  my_id, num_procs;
	HYPRE_Real epsilon, gamma, t, r_norm, b_norm, den_norm, x_norm;
	HYPRE_Real w_norm;

	HYPRE_Real epsmac = 1.e-16; 

	HYPRE_Real relative_error = 1.0;
	/* TIMERS */

	HYPRE_Real gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
	HYPRE_Real massAxpyTime =0.0; 
	HYPRE_Real massIPTime = 0.0f, preconTime = 0.0f, mvTime = 0.0f;    
	HYPRE_Real initTime = 0.0f;

	(cogmres_data -> converged) = 0;
	/*-----------------------------------------------------------------------
	 * With relative change convergence test on, it is possible to attempt
	 * another iteration with a zero residual. This causes the parameter
	 * alpha to go NaN. The guard_zero_residual parameter is to circumvent
	 * this. Perhaps it should be set to something non-zero (but small).
	 *-----------------------------------------------------------------------*/

	(*(cogmres_functions->CommInfo))(A,&my_id,&num_procs);
	if ( logging>0 || print_level>0 )
	{
		norms          = (cogmres_data -> norms);
	}

	/* initialize work arrays */
	rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	c  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	s  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);

	rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_functions, HYPRE_MEMORY_HOST);
	/* KS copy matrix to GPU once */

	hypre_CSRMatrix * AA = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A);

	HYPRE_Int         num_rows = AA->num_rows;
	HYPRE_Int         num_cols = AA->num_cols;
	HYPRE_Int         num_nonzeros = AA->num_nonzeros;
	HYPRE_Real * A_dataGPUonly;
	HYPRE_Int  * A_iGPUonly, *A_jGPUonly;
	//allocate
	cudaMalloc ( &A_dataGPUonly, num_nonzeros*sizeof(HYPRE_Real)); 
	cudaMalloc ( &A_jGPUonly, num_nonzeros*sizeof(HYPRE_Int)); 
	cudaMalloc ( &A_iGPUonly, (num_rows+1)*sizeof(HYPRE_Int)); 

	cudaMemcpy (A_dataGPUonly,AA->data, 
			num_nonzeros*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice );
	cudaMemcpy (A_iGPUonly,AA->i, 
			(num_rows+1)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 
	cudaMemcpy (A_jGPUonly,AA->j, 
			(num_nonzeros)*sizeof(HYPRE_Int),
			cudaMemcpyDeviceToDevice ); 


	HYPRE_Int sz = num_cols;	
	HYPRE_Real * tempH;

	/* allocate Krylov space */
	cudaMalloc ( &cogmres_data -> p, sz*sizeof(HYPRE_Real)*(k_dim+1)); 
	p = (HYPRE_Real*) cogmres_data->p;

	cudaMalloc ( &tempH, sizeof(HYPRE_Real)*(k_dim+1)); 
	HYPRE_Real * tempV;
	cudaMalloc ( &tempV, sizeof(HYPRE_Real)*(sz)); 


	hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);

	HYPRE_Real *bbtemp, *wtemp;

	/* copy x and b to the GPU - ONCE */
	hypre_ParVector * cc =  (hypre_ParVector*) b;

	cudaMalloc ( &b_GPUonly, sizeof(HYPRE_Real)*(sz)); 
	cudaMemcpy (b_GPUonly,
			cc->local_vector->data, 
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 

	InnerProdGPUonly(b_GPUonly,  
			b_GPUonly, 
			&b_norm, 
			sz);
	hypre_ParVector * xx =  (hypre_ParVector*) x;

	cudaMalloc ( &x_GPUonly, sizeof(HYPRE_Real)*(sz)); 
	cudaMemcpy (x_GPUonly,
			xx->local_vector->data, 
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 

	//cudaMalloc ( &r_GPUonly, sizeof(HYPRE_Real)*(sz)); 


	cusparseHandle_t myHandle;
	cusparseMatDescr_t myDescr;
	cusparseCreate(&myHandle);

	cusparseCreateMatDescr(&myDescr); 
	cusparseSetMatType(myDescr,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(myDescr,CUSPARSE_INDEX_BASE_ZERO);

	/* compute initial residual */
	HYPRE_Real one = 1.0f, minusone = -1.0f, zero = 0.0f;

	/* so now our stop criteria is |r_i| <= epsilon */

	if ( print_level>1 && my_id == 0 )
	{
		hypre_printf("=============================================\n\n");
		hypre_printf("Iters     resid.norm     conv.rate  \n");
		hypre_printf("-----    ------------    ---------- \n");
	}


	/* once the rel. change check has passed, we do not want to check it again */

	time2 = MPI_Wtime();
	if (my_id == 0){
		hypre_printf("GMRES INIT TIME: %16.16f \n", time2-time1); 
		initTime += time2-time1;
	} 
	/*outer loop */

	iter = 0;

	while (iter < max_iter)
	{

		printf("iter %d \n", iter);
		if (iter == 0){	
			cudaMemcpy (tempV,b_GPUonly, 
					(sz)*sizeof(HYPRE_Real),
					cudaMemcpyDeviceToDevice ); 

			cusparseDcsrmv(myHandle ,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					num_rows, num_cols, num_nonzeros,
					&minusone, myDescr,
					A_dataGPUonly,A_iGPUonly,A_jGPUonly,
					x_GPUonly, &one, tempV);

cudaDeviceSynchronize();
printf("num_rows %d num_cols %d nnz %d \n", num_rows, num_cols, num_nonzeros);
double *temp1 = (double *) calloc(sz, sizeof(double));
double *temp2 = (double *) calloc(sz, sizeof(double));

	cudaMemcpy(temp1,
			tempV,
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToHost ); 

//debug

	cudaMemcpy(r->local_vector->data,
			b_GPUonly,
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 
  
	cudaMemcpy(w->local_vector->data,
			x_GPUonly,
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 

 (*(cogmres_functions->Matvec))(matvec_data,-1.0, A, w, 1.0, r);
	cudaMemcpy (p,r->local_vector->data,	
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 

	cudaMemcpy(temp2,
			r->local_vector->data,
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToHost ); 
for (i =0; i<sz; i++){
if (temp1[i] != temp2[i]){
printf("i= %d, cusparse %16.16f hypre %16.16f difference %16.16f \n", i, temp1[i], temp2[i], temp1[i]-temp2[i]);
}

}

//end of debug


			InnerProdGPUonly(p,  
					p, 
					&r_norm, 
					sz);
			r_norm = sqrt(r_norm);
		}

		//otherwise, we already have p[0] from previous cycle

		printf("r norm at the start of the cycle %16.16f \n", r_norm);

		if ( logging>0 || print_level > 0)
		{
			norms[iter] = r_norm;
			if ( print_level>1 && my_id == 0 )
			{

				hypre_printf("L2 norm of b: %e\n", b_norm);
				hypre_printf("Initial L2 norm of residual: %e\n", r_norm);

			}
		}


		t = 1.0f/r_norm;
		/* scale the initial vector */

		ScaleGPUonly(p, 
				t, 
				sz);

		if (iter == 0){


			/* conv criteria */

			epsilon = hypre_max(a_tol,r_tol*r_norm);
			printf("conv crit %f \n", epsilon);		
		}
		i = 0; 

		rs[0] = r_norm;
		//debug, to be removed
		InnerProdGPUonly(p,  
				p, 
				&t, 
				sz);
		printf("beginning of cycle, norm of the first vector is %16.16f \n", t);
		rv[0] = t;
		while (i < k_dim && iter < max_iter)
		{

			time1 = MPI_Wtime();
			i++;
			iter++;

			printf("i = %d, will be computing p[%d] \n", i, i);

cudaDeviceSynchronize();
			cusparseDcsrmv(myHandle ,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					num_rows, num_cols, num_nonzeros,
					&one, myDescr,
					A_dataGPUonly,A_iGPUonly,A_jGPUonly,
					&p[(i-1)*sz], &zero, &p[i*sz]);

			time2 = MPI_Wtime();

			mvTime += (time2-time1);
			matvecPreconTime += (time2-time1);   

			InnerProdGPUonly(&p[i*sz],  
					&p[i*sz], 
					&t, 
					sz);
			t = sqrt(t);
			printf("initial norm of p[%d] = %16.16f\n", i, t);
			/* GRAM SCHMIDT */

			time1=MPI_Wtime();


			MassInnerProdGPUonly(&p[i*sz],
					p,
					tempH,				
					i,
					sz);
			cudaMemcpy ( &hh[idx(0, i-1,k_dim+1)],tempH,
					i*sizeof(HYPRE_Real),
					cudaMemcpyDeviceToHost );


			time3 = MPI_Wtime();
			massIPTime += time3-time1;

			HYPRE_Real t2 = 0.0;
			for (j=0; j<i; j++){
				HYPRE_Int id = idx(j, i-1,k_dim+1);
				printf("hh[%d, %d] = %16.16f, scaling by %16.16f, which gives %16.16f  \n",j, i-1, hh[id], (2.0f-rv[j]), (2.0f-rv[j])*hh[id]);		
				hh[id]       = (2.0f-rv[j])*hh[id];
				t2          += (hh[id]*hh[id]);        
			}

			cudaMemcpy ( tempH,&hh[idx(0, i-1,k_dim+1)],
					i*sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );


			time4 = MPI_Wtime();
			MassAxpyGPUonly(sz,  i,
					p,				
					&p[sz*i],
					tempH);	

			time3 = MPI_Wtime();
			massAxpyTime += time3-time4;      


			t2 = sqrt(t2)*sqrt(rv[i-1]);

			printf("t = %f t2 = %f \n", t, t2);
			hh[idx(i, i-1,k_dim+1)] = sqrt(t-t2)*sqrt(t2+t);

			if (hh[idx(i,i-1,k_dim+1)] != 0.0)
			{
				t = 1.0/hh[idx(i,i-1,k_dim+1)]; 

				ScaleGPUonly(&p[sz*i], 
						t, 
						sz);

				InnerProdGPUonly(&p[i*sz],  
						&p[i*sz], 
						&rv[i], 
						sz);
				printf("AFTER  IP  IS %f \n", rv[i]);
			}//if

			for (j = 1; j < i; j++)
			{
				t = hh[idx(j-1,i-1,k_dim+1)];
				printf("t = %f \n", t);
				hh[idx(j-1,i-1,k_dim+1)] = s[j-1]*hh[idx(j,i-1,k_dim+1)] + c[j-1]*t;
				hh[idx(j,i-1, k_dim+1)]  = -s[j-1]*t + c[j-1]*hh[idx(j,i-1,k_dim+1)];
				printf("h[%d, %d] = %f and h[%d, %d ]= %f \n", j-1, i-1, hh[idx(j-1,i-1,k_dim+1)],j, i-1,  hh[idx(j,i-1, k_dim+1)] );
			}
			t     = hh[idx(i, i-1, k_dim+1)]*hh[idx(i,i-1, k_dim+1)];
			t    += hh[idx(i-1,i-1, k_dim+1)]*hh[idx(i-1,i-1, k_dim+1)];
			gamma = sqrt(t);
			if (gamma == 0.0) gamma = epsmac;


			c[i-1]  = hh[idx(i-1,i-1, k_dim+1)]/gamma;
			s[i-1]  = hh[idx(i,i-1, k_dim+1)]/gamma;
			rs[i]   = -hh[idx(i,i-1, k_dim+1)]*rs[i-1];
			rs[i]  /= gamma;
			rs[i-1] = c[i-1]*rs[i-1];
			// determine residual norm 
			hh[idx(i-1,i-1, k_dim+1)] = s[i-1]*hh[idx(i,i-1, k_dim+1)] + c[i-1]*hh[idx(i-1,i-1, k_dim+1)];
			r_norm = fabs(rs[i]);
			printf("r_norm %f \n", r_norm);	

			time1 = MPI_Wtime();  
			if ( print_level>0 )
			{
				norms[iter] = r_norm;
				if ( print_level>1 && my_id == 0 )
				{
					hypre_printf("% 5d    %e    %f\n", iter, norms[iter],
							norms[iter]/norms[iter-1]);
				}
			}// if (printing)
			if (r_norm <epsilon){

				(cogmres_data -> converged) = 1;
				break;
			}//conv check
		}//while (inner)
		/*compute solution */
		printf("computing solution, i = %d \n", i);
		/*rs[i-1] = rs[i-1]/hh[idx(i-1,i-1, k_dim+1)];
			for (int ii = 1; ii<=i; ii++){
			k = i-ii+1;
			int k1 = k+1;
			t = rs[k];

			for (j=k1; j<i; ++j){
			t -= hh[idx(k,j, k_dim+1)]*rs[j];
			}
			rs[k] = t/hh[idx(k,k, k_dim+1)];
			printf("rs[%d] = %f \n", k, rs[k]);		
			}	
			*/

		rs[i-1] = rs[i-1]/hh[idx(i-1,i-1, k_dim+1)];
		printf("rs[%d] = %f  \n", i-1, rs[i-1]);	
		for (k = i-2; k >= 0; k--)
		{
			t = 0.0;
			for (j = k+1; j < i; j++)
			{
				t -= hh[idx(k,j, k_dim+1)]*rs[j];
			}
			t+= rs[k];
			rs[k] = t/hh[idx(k,k, k_dim+1)];
			printf("rs[%d] = %f \n", k, rs[k]);
		}
		/*	for (k = i-2; k >= 0; k--)
				{
				t = 0.0;
				for (j = k+1; j < i; j++)
				{
				t -= hh[idx(k,j, k_dim+1)]*rs[j];
				}
				t+= rs[k];
				rs[k] = t/hh[idx(k,k, k_dim+1)];
				}*/

		printf("running AXPY!  \n");
		for (j = i-1; j >=0; j--){
			printf("using vector p[%d] \n", j);		
			AxpyGPUonly(&p[j*sz],x_GPUonly,	
					rs[j],
					sz);

			InnerProdGPUonly(x_GPUonly,  
					x_GPUonly, 
					&t, 
					sz);
			printf("norm of x is %f \n", sqrt(t));
		}

		//	AxpyGPUonly(&p[(i-1)*sz],x_GPUonly,	
		//				rs[i-1],
		//			sz);
		/*test solution */
		//debug - to be remopved
		cudaMemcpy (tempV,b_GPUonly, 
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice ); 

cudaDeviceSynchronize();
		cusparseDcsrmv(myHandle ,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				num_rows, num_cols, num_nonzeros,
				&minusone, myDescr,
				A_dataGPUonly,A_iGPUonly,A_jGPUonly,
				x_GPUonly, &one, tempV);

		InnerProdGPUonly(tempV,  
				tempV, 
				&t, 
				sz);
		printf("norm of r is %f \n", sqrt(t));
		if (r_norm < epsilon){

			(cogmres_data -> converged) = 1;
			break;
		}
		//update p if appropriate

		for (j=i ; j > 0; j--)
		{
			rs[j-1] = -s[j-1]*rs[j];
			rs[j] = c[j-1]*rs[j];
		}

		if (i){ 

			AxpyGPUonly(&p[i*sz],&p[i*sz],	
					rs[i]-1.0f,
					sz);
		}
		for (j=i-1 ; j > 0; j--){

			AxpyGPUonly(&p[j*sz],&p[i*sz],	
					rs[j],
					sz);
		}
		//   (*(gmres_functions->Axpy))(rs[j],p[j],p[i]);

		if (i)
		{
			// (*(gmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
			// (*(gmres_functions->Axpy))(1.0,p[i],p[0]);

			AxpyGPUonly(&p[0*sz],&p[0*sz],	
					rs[0]-1.0f,
					sz);

			AxpyGPUonly(&p[i*sz],&p[0*sz],	
					1.0f,
					sz);
		}


		/* final tolerance */



		//printf("Iters = %d, residue = %e\n", iter, r_norm*a_tol/epsilon);
	}//while (outer)

	cudaMemcpy (xx->local_vector->data,x_GPUonly, 
			(sz)*sizeof(HYPRE_Real),
			cudaMemcpyDeviceToDevice ); 
}//Solve




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

