//NEW SIMPLIFIED VERSION, WRITTEN BY KS
// AUGUST 2018
// BASED ON RUIPENG'S LI code and gmres.c
//

#define solverTimers 1
#define usePrecond 0


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

/*======
 * KS: THIS IS for using various orth method; user can choose some options 
 * ============================*/

//#define GSoption 1
// 0 is modified gram-schmidt as in "normal" GMRES straight from SAADs book
// 1 is co-gmres, version 0 by ST and KS
// allocate spaces for GPU (copying does not happen inside) prior to calling

void GramSchmidt (HYPRE_Int option,
		HYPRE_Int i,
		HYPRE_Int sz, 
		HYPRE_Int k_dim,
		HYPRE_Real * Vspace, 
		HYPRE_Real * w, 
		HYPRE_Real * Hcolumn,
		HYPRE_Real * HcolumnGPU,  
		HYPRE_Real* rv, 
		HYPRE_Real * rvGPU,
		HYPRE_Real * L){

	HYPRE_Int j;
	HYPRE_Real t;
	if (option == 0){
		for (j=0; j<i; ++j){ 
			InnerProdGPUonly(&Vspace[j*sz],  
					w, 
					&Hcolumn[idx(j, i-1, k_dim+1)], 
					sz);
			//printf("h[%d] = %f \n",j, Hcolumn[idx(j, i-1, k_dim+1)]);
			AxpyGPUonly(&Vspace[j*sz],w,	
					(-1.0)*Hcolumn[idx(j, i-1, k_dim+1)],
					sz);
		}

		InnerProdGPUonly(w,  
				w, 
				&t, 
				sz);
		t = sqrt(t);
		//printf("h[%d] = %f \n",i, Hcolumn[idx(i, i-1, k_dim+1)]);
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (t != 0){
			t = 1/t;

			ScaleGPUonly(w, 
					t, 
					sz);
			//printf("scaling by %f \n", t);
			/*InnerProdGPUonly(w,  
				w, 
				&t, 
				sz);
				printf("vector %d has norm %f \n", i, sqrt(t));
				*/
		}

	}
	if (option == 1){

		InnerProdGPUonly(w,  
				w, 
				&t, 
				sz);
		t = sqrt(t);
		MassInnerProdWithScalingGPUonly(w,
				Vspace,
				rvGPU, 
				HcolumnGPU,				
				i,
				sz);

		cudaMemcpy ( &Hcolumn[idx(0, i-1,k_dim+1)],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		MassAxpyGPUonly(sz,  i,
				Vspace,				
				w,
				HcolumnGPU);	

		HYPRE_Real t2= 0.0f;

		for (j=0; j<i; j++){
			HYPRE_Int id = idx(j, i-1,k_dim+1);
			t2          += (Hcolumn[id]*Hcolumn[id]);        
		}
		t2 = sqrt(t2)*sqrt(rv[i-1]);

		Hcolumn[idx(i, i-1, k_dim+1)] = sqrt(t-t2)*sqrt(t2+t);
		//printf("t = %f t2 = %f Hcol = %f \n", t, t2,Hcolumn[idx(i, i-1, k_dim+1)]  ); 
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			//      printf("scaling by 1/%f = %f\n", Hcolumn[idx(i, i-1, k_dim+1)], t);
			ScaleGPUonly(w, 
					t, 
					sz);

			InnerProdGPUonly(w,  
					w, 
					&rv[i], 
					sz);
			//   printf("NORM OF vector %d is %f \n", i, rv[i]);
			double dd = 2.0f - rv[i];
			cudaMemcpy (&rvGPU[i], &dd,
					sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );
		}//if

	}

	if (option == 2){
		//CGS2 -- straight from Ruhe

		MassInnerProdGPUonly(w,
				Vspace,
				HcolumnGPU,				
				i,
				sz);

		cudaMemcpy ( &Hcolumn[idx(0, i-1,k_dim+1)],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		MassAxpyGPUonly(sz,  i,
				Vspace,				
				w,
				HcolumnGPU);	
		//k=2
		//do it again
		MassInnerProdGPUonly(w,
				Vspace,
				HcolumnGPU,				
				i,
				sz);

		cudaMemcpy ( &rv[0],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		for (j=0; j<i; j++){
			HYPRE_Int id = idx(j, i-1,k_dim+1);
			Hcolumn[id]+=rv[j];        
		}

		MassAxpyGPUonly(sz,  i,
				Vspace,				
				w,
				HcolumnGPU);	


		InnerProdGPUonly(w,  
				w, 
				&t, 
				sz);
		t = sqrt(t);
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			ScaleGPUonly(w, 
					t, 
					sz);

		}//if

	}
	if (option == 3){
		//Alg 3 from paper
		//C version of orth_cgs2_new by ST
		//remember: U NEED L IN THIS CODE AND NEED TO KEEP IT!! 
		//L(1:i, i) = V(:, 1:i)'*V(:,i);
		//
		//printf("option = %d, i = %d \n", option, i );
		MassInnerProdGPUonly(&Vspace[(i-1)*sz],
				Vspace,
				rvGPU,				
				i,
				sz);

		//copy rvGPU to L

		//printf("test 2\n");
		cudaMemcpy ( &L[(i-1)*(k_dim+1)],rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//aux = V(:,i)'*w;

		//printf("test 3\n");
		MassInnerProdGPUonly(w,
				Vspace,
				rvGPU,				
				i,
				sz);

		//printf("test 4\n");
		cudaMemcpy ( rv,rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//H(1:i, i) = D*aux - Lp*aux - Lp'*aux
		/*for (int j=0; j<i; ++j){
			printf("L[%d, %d] = %16f \n",j, i-1,L[(i-1)*(k_dim+1)+j]);
			}
			printf("L matrix as is!!\n");

			for (int j=0; j<i; ++j){
			for (int k=0; k<i; ++k){

			printf(" %16.16f ", L[k*(k_dim+1) +j]);

			}
			printf("\n");
			}*/

		//printf("rv as is \n");

		for (int j=0; j<i; ++j){
			//printf(" %16.16f \n ", rv[j]);
			Hcolumn[(i-1)*(k_dim+1)+j] = 0.0f;
		}

		//printf("test 5, i = %d\n", i);
		for (int j=0; j<i; ++j){
			for (int k=0; k<i; ++k){
				//we are processing H[j*(k_dim+1)+k]
				if (j==k){Hcolumn[(i-1)*(k_dim+1)+k] += L[j*(k_dim+1)+k]*rv[j];
					//        printf("DIAGONAL L(%d, %d) = %16.16f , aux[%d] = %f, adding to: H(%d,%d) \n",k, j,L[j*(k_dim+1)+k], j, rv[j], k, i-1);

				} 

				if (j<k){
					Hcolumn[(i-1)*(k_dim+1)+j] -= L[k*(k_dim+1)+j]*rv[k];
					//printf("LpT L(%d, %d) = %16.16f , aux[%d] = %f, adding to: H(%d,%d) \n", j,k, L[k*(k_dim+1)+j], k, rv[k], j, i -1);
				}
				if (j>k){

					Hcolumn[(i-1)*(k_dim+1)+j] -= L[j*(k_dim+1)+k]*rv[k];
					//        printf("Lp L(%d, %d) = %16.16f , aux[%d] = %f adding to: H(%d,%d) \n",j, k, L[j*(k_dim+1)+k], k, rv[k], j, i-1 );
				}
			}//for k 
		}//for j

		/*printf("Hcolumn AFTER \n");

			for (int j=0; j<i; ++j){
			printf(" %16.16f \n ", Hcolumn[(i-1)*(k_dim+1)+j]);
			}
		//z = z -H*V

		printf("test 6\n");
		*/
		cudaMemcpy ( HcolumnGPU,&Hcolumn[(i-1)*(k_dim+1)],
				i*sizeof(HYPRE_Real),
				cudaMemcpyHostToDevice);


		//printf("test 7\n");
		MassAxpyGPUonly(sz,  i,
				Vspace,				
				w,
				HcolumnGPU);	
		//normalize
		InnerProdGPUonly(w,  
				w, 
				&t, 
				sz);
		t = sqrt(t);

		//printf("test 8, t = %f\n", t);
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			ScaleGPUonly(w, 
					t, 
					sz);

		}//if

	}
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
	if (solverTimers)
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
	hypre_ParVector * w ;
//	hypre_ParVector * w2 ;
	//  void             *w            = (cogmres_data -> w);


	HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
	HYPRE_Int  *precond_data                      = (HYPRE_Int*)(cogmres_data -> precond_data);

	HYPRE_Real     *norms          = (cogmres_data -> norms);
	HYPRE_Int print_level = (cogmres_data -> print_level);
	HYPRE_Int logging     = (cogmres_data -> logging);

	HYPRE_Int GSoption = (cogmres_data -> GSoption);
	HYPRE_Int  break_value = 0;
	HYPRE_Int  i, j, k;
	/*KS: rv is the norm history */
	HYPRE_Real *rs, *hh, *c, *s, *rs_2, *rv, *L;
	HYPRE_Real *x_GPUonly, *b_GPUonly;
	HYPRE_Int  iter; 
	HYPRE_Int  my_id, num_procs;
	HYPRE_Real epsilon, gamma, t, r_norm, b_norm, x_norm;

	HYPRE_Real epsmac = 1.e-16; 

	HYPRE_Real relative_error = 1.0;
	/* TIMERS */

	HYPRE_Real gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
	HYPRE_Real massAxpyTime =0.0; 
	HYPRE_Real gsOtherTime =0.0f;
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
	if (usePrecond) { 
		
    (cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
//Initialize
	//	w2=  (hypre_ParVector*) (cogmres_data->w_2);
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
  //hypre_CTAllocF(HYPRE_Real, num_nonzeros,cogmres_functions, HYPRE_MEMORY_DEVICE);
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
	HYPRE_Real * tempRV;

	/* allocate Krylov space */
	cudaMalloc ( &cogmres_data -> p, sz*sizeof(HYPRE_Real)*(k_dim+1)); 
	p = (HYPRE_Real*) cogmres_data->p;
	if (GSoption != 0){
		//printf("allocating tempRV and tempH \n");
		cudaMalloc ( &tempRV, sizeof(HYPRE_Real)*(k_dim+1)); 
		cudaMalloc ( &tempH, sizeof(HYPRE_Real)*(k_dim+1)); 
	}
	HYPRE_Real * tempV;
	cudaMalloc ( &tempV, sizeof(HYPRE_Real)*(sz)); 


	hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
	if (GSoption >=3){
		L = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
	}
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
  b_norm = sqrt(b_norm);
	
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

	if (solverTimers){
		time2 = MPI_Wtime();
		initTime += time2-time1;
	}
	if (my_id == 0){
		hypre_printf("GMRES INIT TIME: %16.16f \n", time2-time1); 
	} 
	/*outer loop */

	iter = 0;

	while (iter < max_iter)
	{

		if (solverTimers)
			time1 = MPI_Wtime();

		if (iter == 0){
			//precon here

			if (usePrecond){
				//p[0] =  M*x_GPUonly 

   (*(cogmres_functions->ClearVector))(w);
   (*(cogmres_functions->ClearVector))(xx);
				cudaMemcpy (w->local_vector->data,
						x_GPUonly, 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 
				if (solverTimers){
					time3 = MPI_Wtime();
				}
				precond(precond_data, A, w, xx);

				if (solverTimers){
					time4 = MPI_Wtime();
					preconTime +=(time4-time3);
					matvecPreconTime += (time4-time3);
				}
				cudaMemcpy (x_GPUonly,
						xx->local_vector->data, 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 
			}

			printf("done 1\n");


			cudaMemcpy (p,b_GPUonly, 
					(sz)*sizeof(HYPRE_Real),
					cudaMemcpyDeviceToDevice ); 
			printf("done 0\n");
			if (solverTimers){
				time3 = MPI_Wtime();
			}
/*			cusparseDcsrmv(myHandle ,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					num_rows, num_cols, num_nonzeros,
					&minusone, myDescr,
					A_dataGPUonly,A_iGPUonly,A_jGPUonly,
					x_GPUonly, &one, p);
*/

    w =  (hypre_ParVector*) (cogmres_data->w);
printf("about to initialize! \n");
hypre_ParVectorInitialize(w);

	//	(cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
		
printf("about to grab w local\n");
  hypre_Vector *w_local = hypre_ParVectorLocalVector(w);
printf("I have w local\n");
   (*(cogmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, w);

			if (solverTimers){
				time4 = MPI_Wtime();
				matvecPreconTime+=(time4-time3);
				mvTime += (time4-time3);
			}

			//copy BACK
			//
			/*
if (usePrecond){
				cudaMemcpy (x_GPUonly,
						w->local_vector->data, 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 
			}*/
			InnerProdGPUonly(p,  
					p, 
					&r_norm, 
					sz);
			r_norm = sqrt(r_norm);

			if ( logging>0 || print_level > 0)
			{
				norms[iter] = r_norm;
				if ( print_level>1 && my_id == 0 ){

					hypre_printf("L2 norm of b: %e\n", b_norm);
					hypre_printf("Initial L2 norm of residual: %e\n", r_norm);

				}
			}

			/* conv criteria */

			epsilon = hypre_max(a_tol,r_tol*r_norm);
			printf("eps = %f \n", epsilon);
		}//if

		//otherwise, we already have p[0] from previous cycle


		t = 1.0f/r_norm;
		/* scale the initial vector */
		ScaleGPUonly(p, 
				t, 
				sz);

		i = 0; 

		rs[0] = r_norm;
		rv[0] = 1.0;
		if (GSoption >=3){  
			L[0] = 1.0f;
		}
		if (GSoption != 0){
			cudaMemcpy (&tempRV[0], &rv[0],
					sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );
		}

		if (solverTimers){
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
		}

		while (i < k_dim && iter < max_iter)
		{
			i++;
			iter++;

			if (usePrecond){
				//x = M * p[i-1]
				//p[i] = A*x

   (*(cogmres_functions->ClearVector))(w);
   (*(cogmres_functions->ClearVector))(xx);
				cudaMemcpy (w->local_vector->data,
						&p[(i-1)*sz], 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 

				if (solverTimers){
					time3 = MPI_Wtime();
				}
				precond(precond_data, A, w, xx);

				if (solverTimers){
					time4 = MPI_Wtime();

					preconTime += (time4-time3);
					matvecPreconTime += (time4-time3);
				}
				//printf("norm after Precond! %16.16f \n ", (*(cogmres_functions->InnerProd))(x,x));
				cudaMemcpy (tempV,
						xx->local_vector->data, 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 

				if (solverTimers)
					time1 = MPI_Wtime();
				cusparseDcsrmv(myHandle ,
						CUSPARSE_OPERATION_NON_TRANSPOSE,
						num_rows, num_cols, num_nonzeros,
						&one, myDescr,
						A_dataGPUonly,A_iGPUonly,A_jGPUonly,
						tempV, &zero, &p[i*sz]);
				if (solverTimers){

					time2 = MPI_Wtime();
					mvTime += (time2-time1);
					matvecPreconTime += (time2-time1);   
				}
			}

			else{

				if (solverTimers)
					time1 = MPI_Wtime();
				cusparseDcsrmv(myHandle ,
						CUSPARSE_OPERATION_NON_TRANSPOSE,
						num_rows, num_cols, num_nonzeros,
						&one, myDescr,
						A_dataGPUonly,A_iGPUonly,A_jGPUonly,
						&p[(i-1)*sz], &zero, &p[i*sz]);
			}
			time2 = MPI_Wtime();
			if (solverTimers){
				time2 = MPI_Wtime();
				mvTime += (time2-time1);
				matvecPreconTime += (time2-time1);   
			}


			/* GRAM SCHMIDT */
			if (solverTimers)
				time1=MPI_Wtime();
			if (GSoption == 0){

				GramSchmidt (0, 
						i, 
						sz,
						k_dim, 
						p, 
						&p[i*sz], 
						hh, 
						NULL,  
						rv, 
						NULL, NULL );
			}

			if ((GSoption >0) &&(GSoption <3)){
				GramSchmidt (GSoption, 
						i, 
						sz, 
						k_dim,
						p, 
						&p[i*sz], 
						hh, 
						tempH,  
						rv, 
						tempRV, NULL );
			}



			if ((GSoption >=3)){
				//printf("starting Alg 3!I have  i = %d vectors in space and ONE new\n", i);		
				GramSchmidt (GSoption, 
						i, 
						sz, 
						k_dim,
						p, 
						&p[i*sz], 
						hh, 
						tempH,  
						rv, 
						tempRV, L );
			}
			// CALL IT HERE
			if (solverTimers){
				time2 = MPI_Wtime();
				//gsOtherTime +=  time2-time3;
				gsTime += (time2-time1);
			}
			for (j = 1; j < i; j++)
			{
				t = hh[idx(j-1,i-1,k_dim+1)];
				hh[idx(j-1,i-1,k_dim+1)] = s[j-1]*hh[idx(j,i-1,k_dim+1)] + c[j-1]*t;
				hh[idx(j,i-1, k_dim+1)]  = -s[j-1]*t + c[j-1]*hh[idx(j,i-1,k_dim+1)];
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

			if (solverTimers){
				time4 = MPI_Wtime();
				linSolveTime += time4-time2;
			}

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

		if (solverTimers)
			time1 = MPI_Wtime();  
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

		for (j = i-1; j >=0; j--){
			//printf("using vector p[%d] \n", j);		
			AxpyGPUonly(&p[j*sz],x_GPUonly,	
					rs[j],
					sz);

		}

		/*test solution */
		//debug - to be remopved
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
		if (i)
		{

			AxpyGPUonly(&p[0*sz],&p[0*sz],	
					rs[0]-1.0f,
					sz);

			AxpyGPUonly(&p[i*sz],&p[0*sz],	
					1.0f,
					sz);
		}


		/* final tolerance */


		if (solverTimers){
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
		}

		//printf("Iters = %d, residue = %e\n", iter, r_norm*a_tol/epsilon);
	}//while (outer)

	if (solverTimers){
		time1 = MPI_Wtime();
	}
	if (usePrecond){
	
   (*(cogmres_functions->ClearVector))(w);
   (*(cogmres_functions->ClearVector))(xx);
	cudaMemcpy (w->local_vector->data,x_GPUonly,
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );

		if (solverTimers){
			time3 = MPI_Wtime();
		}

		precond(precond_data, A, w, xx);

/*		cudaMemcpy (p, b_GPUonly,
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );
		cudaMemcpy (x_GPUonly, xx->local_vector->data,
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );

		cusparseDcsrmv(myHandle ,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				num_rows, num_cols, num_nonzeros,
				&minusone, myDescr,
				A_dataGPUonly,A_iGPUonly,A_jGPUonly,
				x_GPUonly, &one, p);
double ttt;
		InnerProdGPUonly(p,  
				p, 
				&ttt, 
				sz);

		printf("Norm of residual AFTER prec %16.16f \n", sqrt(ttt)/b_norm);
*/


		if (solverTimers){
			time4 = MPI_Wtime();
			preconTime+=(time4-time3);
			matvecPreconTime += (time4-time3);
		}
	}
	else{
	/*	double ttt;
		cudaMemcpy (p, b_GPUonly,
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );
		cudaMemcpy (x_GPUonly, xx->local_vector->data,
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );

		cusparseDcsrmv(myHandle ,
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				num_rows, num_cols, num_nonzeros,
				&minusone, myDescr,
				A_dataGPUonly,A_iGPUonly,A_jGPUonly,
				x_GPUonly, &one, p);

		InnerProdGPUonly(p,  
				p, 
				&ttt, 
				sz);

		printf("No-precond Norm of residual %16.16f \n", sqrt(ttt));

*/
		cudaMemcpy (xx->local_vector->data,x_GPUonly, 
				(sz)*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToDevice );
	}
	/* clean up */
	cudaFree(A_dataGPUonly);
	cudaFree(A_iGPUonly);
	cudaFree(A_jGPUonly);

	cudaFree(p);
	cudaFree(tempV);

	if (GSoption != 0){
		cudaFree(tempH);

		cudaFree(tempRV);
	}

	cudaFree(x_GPUonly);
	cudaFree(b_GPUonly);

	if (solverTimers){
		time2 = MPI_Wtime();
		remainingTime += (time2-time1);
	}


	if ((my_id == 0)&& (solverTimers)){
		hypre_printf("itersAll(%d,%d)                = %d \n", GSoption+1, k_dim/5, iter);
		hypre_printf("timeGSAll(%d,%d)                = %16.16f \n",GSoption+1, k_dim/5, gsTime);

		hypre_printf("TIME for CO-GMRES\n");
		hypre_printf("init cost            = %16.16f \n", initTime);
		hypre_printf("matvec+precon        = %16.16f \n", matvecPreconTime);
		hypre_printf("gram-schmidt (total) = %16.16f \n", gsTime);
		hypre_printf("linear solve         = %16.16f \n", linSolveTime);
		hypre_printf("all other            = %16.16f \n", remainingTime);
		hypre_printf("FINE times\n");
		hypre_printf("mass Inner Product   = %16.16f \n", massIPTime);
		hypre_printf("mass Axpy            = %16.16f \n", massAxpyTime);
		hypre_printf("Gram-Schmidt: other  = %16.16f \n", gsOtherTime);
		hypre_printf("precon multiply      = %16.16f \n", preconTime);
		hypre_printf("mv time              = %16.16f \n", mvTime);
		hypre_printf("MATLAB timers \n");
		hypre_printf("gsTime(%d) = %16.16f \n", k_dim/5, gsTime);	
		hypre_printf("timeAll(%d,%d)                =  ",GSoption+1, k_dim/5);
	}
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
hypre_COGMRESSetGSoption( void *cogmres_vdata,
		HYPRE_Int   level)
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;


	(cogmres_data -> GSoption) = level;

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

