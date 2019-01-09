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

	//printf("COGMRES setup: data and functions retrieved!\n");
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

	//printf("COGMRES setup: creating b!\n");
	if ((cogmres_data -> r) == NULL)
		(cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);



	//printf("COGMRES setup: creating matvec data!\n");
	if ((cogmres_data -> matvec_data) == NULL)
		(cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);

	precond_setup(precond_data, A, b, x);

	/*-----------------------------------------------------
	 * Allocate space for log info
	 *-----------------------------------------------------*/

	//printf("COGMRES setup: creating log INFO!\n");
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
		hypre_ParVector * Vspace,
		HYPRE_Real * Hcolumn,
		HYPRE_Real * HcolumnGPU,  
		HYPRE_Real* rv, 
		HYPRE_Real * rvGPU,
		HYPRE_Real * L){

	HYPRE_Int j;
	HYPRE_Real t;
	if (option == 0){
		for (j=0; j<i; ++j){ 
			/*			InnerProdGPUonly(&Vspace[j*sz],  
							w, 
							&Hcolumn[idx(j, i-1, k_dim+1)], 
							sz);
			//printf("h[%d] = %f \n",j, Hcolumn[idx(j, i-1, k_dim+1)]);
			AxpyGPUonly(&Vspace[j*sz],w,	
			(-1.0)*Hcolumn[idx(j, i-1, k_dim+1)],
			sz);*/

			//printf("inside GS, first IP, i = %d, j= %d \n", i, j);
			Hcolumn[idx(j, i-1, k_dim+1)] = hypre_ParKrylovInnerProdOneOfMult(Vspace,j,Vspace, i);

			//printf("I GOT %16.16f \n", Hcolumn[idx(j, i-1, k_dim+1)]);
			hypre_ParKrylovAxpyOneOfMult((-1.0)*Hcolumn[idx(j, i-1, k_dim+1)], Vspace, j, Vspace, i);		
		}

		/*		InnerProdGPUonly(w,  
					w, 
					&t, 
					sz);
					t = sqrt(t);*/

		t = sqrt(hypre_ParKrylovInnerProdOneOfMult(Vspace,i,Vspace, i));
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (t != 0){
			t = 1/t;

			/*			ScaleGPUonly(w, 
							t, 
							sz);*/

			hypre_ParKrylovScaleVectorOneOfMult(t,Vspace, i);
			//			hypre_ParKrylovAxpyOneOfMult(t, auxV, 0, Vspace, i);		
			//	printf("scaling by %f \n", t);
		}

	}
	if (option == 1){
		//	printf("GS = 1 \n");
		/*		InnerProdGPUonly(w,  
					w, 
					&t, 
					sz);*/

		t  = sqrt(hypre_ParKrylovInnerProdOneOfMult(Vspace,i,Vspace, i));

		/*	MassInnerProdWithScalingGPUonly(w,
				Vspace,
				rvGPU, 
				HcolumnGPU,				
				i,
				sz);*/


		/*--------------------------------------------------------------------------
		 * hypre_ParKrylovMassInnerProdWithScalingMult // written by KS //for multivectors
		 * x is the space, y is the single vector 
		 *--------------------------------------------------------------------------*/
		hypre_ParKrylovMassInnerProdWithScalingMult(Vspace,i, Vspace, i,rvGPU, HcolumnGPU);
		cudaMemcpy ( &Hcolumn[idx(0, i-1,k_dim+1)],
				HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		/*		MassAxpyGPUonly(sz,  i,
					Vspace,				
					w,
					HcolumnGPU);	
					*/

		/*--------------------------------------------------------------------------
		 * hypre_ParKrylovMassAxpyMult (for multivectors, x is a multivector)
		 *--------------------------------------------------------------------------*/

		hypre_ParKrylovMassAxpyMult( HcolumnGPU,Vspace,i, Vspace, i);

		HYPRE_Real t2= 0.0f;

		for (j=0; j<i; j++){
			HYPRE_Int id = idx(j, i-1,k_dim+1);
			t2          += (Hcolumn[id]*Hcolumn[id]);        
		}
		t2 = sqrt(t2)*sqrt(rv[i-1]);

		Hcolumn[idx(i, i-1, k_dim+1)] = sqrt(t-t2)*sqrt(t2+t);
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			/*			ScaleGPUonly(w, 
							t, 
							sz);*/

			hypre_ParKrylovScaleVectorOneOfMult(t,Vspace, i);

			/*			InnerProdGPUonly(w,  
							w, 
							&rv[i], 
							sz);*/

			rv[i]  = hypre_ParKrylovInnerProdOneOfMult(Vspace,i,Vspace, i);
			double dd = 2.0f - rv[i];
			cudaMemcpy (&rvGPU[i], &dd,
					sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );
		}//if

	}
	if (option == 2){
		//CGS2 -- straight from Ruhe
		/*		MassInnerProdGPUonly(w,
					Vspace,
					HcolumnGPU,				
					i,
					sz);*/

		hypre_ParKrylovMassInnerProdMult(Vspace,i, Vspace, i, HcolumnGPU);
		/*
		 *
		 cudaMemcpy ( &Hcolumn[idx(0, i-1,k_dim+1)],
		 HcolumnGPU,
		 i*sizeof(HYPRE_Real),
		 cudaMemcpyDeviceToHost );
		 * */
		cudaMemcpy ( &Hcolumn[idx(0, i-1,k_dim+1)],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		/*		MassAxpyGPUonly(sz,  i,
					Vspace,				
					w,
					HcolumnGPU);	*/

		hypre_ParKrylovMassAxpyMult( HcolumnGPU,Vspace,i, Vspace, i);
		//k=2
		//do it again
		/*		MassInnerProdGPUonly(w,
					Vspace,
					HcolumnGPU,				
					i,
					sz);*/

		hypre_ParKrylovMassInnerProdMult(Vspace,i, Vspace, i, HcolumnGPU);
		cudaMemcpy ( &rv[0],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		for (j=0; j<i; j++){
			HYPRE_Int id = idx(j, i-1,k_dim+1);
			Hcolumn[id]+=rv[j];        
		}

		/*		MassAxpyGPUonly(sz,  i,
					Vspace,				
					w,
					HcolumnGPU);	*/

		hypre_ParKrylovMassAxpyMult( HcolumnGPU,Vspace,i, Vspace, i);


		/*		InnerProdGPUonly(w,  
					w, 
					&t, 
					sz);
					t = sqrt(t);*/

		t  = sqrt(hypre_ParKrylovInnerProdOneOfMult(Vspace,i,Vspace, i));
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			/*			ScaleGPUonly(w, 
							t, 
							sz);*/

			hypre_ParKrylovScaleVectorOneOfMult(t,Vspace, i);
		}//if

	}

	if (option == 3){
		//Alg 3 from paper
		//C version of orth_cgs2_new by ST
		//remember: U NEED L IN THIS CODE AND NEED TO KEEP IT!! 
		//L(1:i, i) = V(:, 1:i)'*V(:,i);
		//
		//printf("option = %d, i = %d \n", option, i );
		/*		MassInnerProdGPUonly(&Vspace[(i-1)*sz],
					Vspace,
					rvGPU,				
					i,
					sz);*/

		hypre_ParKrylovMassInnerProdMult(Vspace,i, Vspace, i-1, rvGPU);
		//copy rvGPU to L

		//printf("test 2\n");
		cudaMemcpy ( &L[(i-1)*(k_dim+1)],rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//aux = V(:,i)'*w;

		//printf("test 3\n");
		/*		MassInnerProdGPUonly(w,
					Vspace,
					rvGPU,				
					i,
					sz);*/

		hypre_ParKrylovMassInnerProdMult(Vspace,i, Vspace, i, rvGPU);

		//printf("test 4\n");
		cudaMemcpy ( rv,rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//H(1:i, i) = D*aux - Lp*aux - Lp'*aux
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

		cudaMemcpy ( HcolumnGPU,&Hcolumn[(i-1)*(k_dim+1)],
				i*sizeof(HYPRE_Real),
				cudaMemcpyHostToDevice);


		//printf("test 7\n");
		/*		MassAxpyGPUonly(sz,  i,
					Vspace,				
					w,
					HcolumnGPU);*/

		hypre_ParKrylovMassAxpyMult( HcolumnGPU,Vspace,i, Vspace, i);
		//normalize
		/*		InnerProdGPUonly(w,  
					w, 
					&t, 
					sz);
					t = sqrt(t);*/

		t  = sqrt(hypre_ParKrylovInnerProdOneOfMult(Vspace,i,Vspace, i));
		//printf("test 8, t = %f\n", t);
		Hcolumn[idx(i, i-1, k_dim+1)] = t;
		if (Hcolumn[idx(i, i-1, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i, i-1, k_dim+1)]; 
			/*			ScaleGPUonly(w, 
							t, 
							sz);*/

			hypre_ParKrylovScaleVectorOneOfMult(t,Vspace, i);

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
	hypre_ParVector * w, *p, *r ;


	HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
	HYPRE_Int  *precond_data                      = (HYPRE_Int*)(cogmres_data -> precond_data);

	HYPRE_Real     *norms          = (cogmres_data -> norms);
	HYPRE_Int print_level = (cogmres_data -> print_level);
	HYPRE_Int logging     = (cogmres_data -> logging);

	HYPRE_Int GSoption = (cogmres_data -> GSoption);
	HYPRE_Int  break_value = 0;
	HYPRE_Int  i, j, k;
	//KS: rv is the norm history 
	HYPRE_Real *rs, *hh, *c, *s, *rs_2, *rv, *L;
	HYPRE_Int  iter; 
	HYPRE_Int  my_id, num_procs;
	HYPRE_Real epsilon, gamma, t, r_norm, b_norm, x_norm;

	HYPRE_Real epsmac = 1.e-16; 

	HYPRE_Real relative_error = 1.0;
	// TIMERS/

	HYPRE_Real gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
	HYPRE_Real massAxpyTime =0.0; 
	HYPRE_Real gsOtherTime =0.0f;
	HYPRE_Real massIPTime = 0.0f, preconTime = 0.0f, mvTime = 0.0f;    
	HYPRE_Real initTime = 0.0f;

	(cogmres_data -> converged) = 0;

	(*(cogmres_functions->CommInfo))(A,&my_id,&num_procs);
	if ( logging>0 || print_level>0 )
	{
		norms          = (cogmres_data -> norms);
	}

	(cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
	r = (hypre_ParVector * )(*(cogmres_functions->CreateVector))(b);
	w = (hypre_ParVector*) cogmres_data -> w;
	hypre_ParVectorInitialize(w);	
	// initialize work arrays 
	rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	c  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	s  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);

	rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_functions, HYPRE_MEMORY_HOST);

	hypre_CSRMatrix * AA = hypre_ParCSRMatrixDiag((hypre_ParCSRMatrix *)A);

	HYPRE_Int         num_rows = AA->num_rows;
	HYPRE_Int         num_cols = AA->num_cols;
	HYPRE_Int         num_nonzeros = AA->num_nonzeros;
	//Mod Oct 2018

	HYPRE_Int sz = num_cols;	
	HYPRE_Real * tempH;
	HYPRE_Real * tempRV;

	hypre_ParVector * bb =  (hypre_ParVector*) b; 
	hypre_ParVector * xx =  (hypre_ParVector*) x;

	p = (hypre_ParVector*) (cogmres_data->p);
	p =(hypre_ParVector*)  hypre_ParMultiVectorCreate( hypre_ParVectorComm(bb),
			sz,
			hypre_ParVectorPartitioning(bb),
			k_dim+1 );

	hypre_ParVectorInitialize(p);

	if (GSoption != 0){
		cudaMalloc ( &tempRV, sizeof(HYPRE_Real)*(k_dim+1)); 
		cudaMalloc ( &tempH, sizeof(HYPRE_Real)*(k_dim+1)); 
	}
	HYPRE_Real * tempV;
	cudaMalloc ( &tempV, sizeof(HYPRE_Real)*(sz)); 


	hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);

	if (GSoption >=3){
		L = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
	}
	b_norm = sqrt(hypre_ParKrylovInnerProdOneOfMult(bb,0,bb, 0));
//	b_norm = sqrt(b_norm);
	HYPRE_Real one = 1.0f, minusone = -1.0f, zero = 0.0f;


	if ( print_level>1 && my_id == 0 )
	{
		//printf("initial norm of b is %16.16f \n", b_norm);	
		hypre_printf("=============================================\n\n");
		hypre_printf("Iters     resid.norm     conv.rate  \n");
		hypre_printf("-----    ------------    ---------- \n");
	}


	// once the rel. change check has passed, we do not want to check it again

	if (solverTimers){
		time2 = MPI_Wtime();
		initTime += time2-time1;
	}
	if (my_id == 0){
		hypre_printf("GMRES INIT TIME: %16.16f \n", time2-time1); 
	} 
	//outer loop 

	iter = 0;
	//works
	//printf("ABOUT TO START OUTER WHILE \n");
	while (iter < max_iter)
	{
		if (solverTimers)
			time1 = MPI_Wtime();

		if (iter == 0){

			if (usePrecond){
				//p[0] =  M*x_GPUonly 

				if (solverTimers){
					time3 = MPI_Wtime();
				}

				hypre_ParKrylovCopyVectorOneOfMult(xx, 0, w, 0);
				precond(precond_data, A, w, xx);


				if (solverTimers){
					time4 = MPI_Wtime();
					preconTime +=(time4-time3);
					matvecPreconTime += (time4-time3);
				}
			}// if usePrecond

			hypre_ParKrylovCopyVectorOneOfMult(bb, 0, p, 0);

			if (solverTimers){
				time3 = MPI_Wtime();
			}

			hypre_ParKrylovMatvecMult(matvec_data,
					minusone,
					A,
					xx,
					0,
					one,
					p, 0);

			hypre_ParVectorCopyDataGPUtoCPU(p);

			if (solverTimers){
				time4 = MPI_Wtime();
				matvecPreconTime+=(time4-time3);
				mvTime += (time4-time3);
			}


			if (usePrecond){
				//xx = w //

				hypre_ParKrylovCopyVectorOneOfMult(w, 0, xx, 0);
			}

			r_norm = sqrt(hypre_ParKrylovInnerProdOneOfMult(p,0,p, 0));
//printf("current r_norm %16.16f \n", r_norm);		
	if ( logging>0 || print_level > 0)
			{
				norms[iter] = r_norm;
				if ( print_level>1 && my_id == 0 ){

			//		hypre_printf("L2 norm of b: %16.16f\n", b_norm);
				//	hypre_printf("Initial L2 norm of residual: %16.16f\n", r_norm);

				}
			}

			// conv criteria 

			epsilon = hypre_max(a_tol,r_tol*r_norm);
		}//if

		//otherwise, we already have p[0] from previous cycle


		t = 1.0f/r_norm;

		hypre_ParKrylovScaleVectorOneOfMult(t,p, 0);
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


#if 1	
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
				cudaMemcpy (tempV,
						xx->local_vector->data, 
						(sz)*sizeof(HYPRE_Real),
						cudaMemcpyDeviceToDevice ); 

				if (solverTimers)
					time1 = MPI_Wtime();
				if (solverTimers){

					time2 = MPI_Wtime();
					mvTime += (time2-time1);
					matvecPreconTime += (time2-time1);   
				}
			}
			else{

// not using preconddd
				if (solverTimers)
					time1 = MPI_Wtime();

				hypre_ParKrylovCopyVectorOneOfMult(p, i-1, w, 0);
				hypre_ParKrylovMatvecMult(matvec_data,
						one,
						A,
						w,
						0,
						zero,
						p, i);

			}
			time2 = MPI_Wtime();
			if (solverTimers){
				time2 = MPI_Wtime();
				mvTime += (time2-time1);
				matvecPreconTime += (time2-time1);   
			}


			// GRAM SCHMIDT 
			if (solverTimers)
				time1=MPI_Wtime();
			if (GSoption == 0){

				GramSchmidt (0, 
						i, 
						sz,
						k_dim, 
						p,
						hh, 
						NULL,  
						rv, 
						NULL, NULL );
			}

			if ((GSoption >0) &&(GSoption <3)){
				//printf("starting GS\n");
				GramSchmidt (GSoption, 
						i, 
						sz, 
						k_dim,
						p, 
						hh, 
						tempH,  
						rv, 
						tempRV, NULL );
			}



			if ((GSoption >=3)){
				GramSchmidt (GSoption, 
						i, 
						sz, 
						k_dim,
						p, 
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
					hypre_printf("ITER: % 5d    %e    %f\n", iter, norms[iter],
							norms[iter]/norms[iter-1]);
				}
			}// if (printing)
			if (r_norm <epsilon){

				(cogmres_data -> converged) = 1;
				break;
			}//conv check
		}//while (inner)

#endif
		//compute solution 

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

			hypre_ParKrylovAxpyOneOfMult(rs[j], p, j, x, 0);		
		}

		//test solution 
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

			hypre_ParKrylovAxpyOneOfMult(rs[i]-1.0f, p, i, p, i);		
		}
		for (j=i-1 ; j > 0; j--){

			hypre_ParKrylovAxpyOneOfMult(rs[j], p, j, p, i);		
		}
		if (i)
		{


			hypre_ParKrylovAxpyOneOfMult(rs[0]-1.0f, p, 0, p, 0);		

			hypre_ParKrylovAxpyOneOfMult(1.0, p, i, p, 0);		
		}


		// final tolerance 


		if (solverTimers){
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
		}
	}
	if (solverTimers){
		time1 = MPI_Wtime();
	}
	if (usePrecond){

		(*(cogmres_functions->ClearVector))(w);
		(*(cogmres_functions->ClearVector))(xx);
		//	cudaMemcpy (w->local_vector->data,x_GPUonly,
		//	(sz)*sizeof(HYPRE_Real),
		//	cudaMemcpyDeviceToDevice );

		if (solverTimers){
			time3 = MPI_Wtime();
		}

		precond(precond_data, A, w, xx);

		//	cudaMemcpy (p, b_GPUonly,
		//		(sz)*sizeof(HYPRE_Real),
		//			cudaMemcpyDeviceToDevice );
		//			cudaMemcpy (x_GPUonly, xx->local_vector->data,
		//				(sz)*sizeof(HYPRE_Real),
		//					cudaMemcpyDeviceToDevice );

		//				cusparseDcsrmv(myHandle ,
		//				CUSPARSE_OPERATION_NON_TRANSPOSE,
		//				num_rows, num_cols, num_nonzeros,
		//					&minusone, myDescr,
		//				A_dataGPUonly,A_iGPUonly,A_jGPUonly,
		//			x_GPUonly, &one, p);
		double ttt;
		/*
			 dd	InnerProdGPUonly(p,  
			 dd		p, 
			 dd	&ttt, 
			 dd		sz);
			 dd	printf("Norm of residual AFTER prec %16.16f \n", sqrt(ttt)/b_norm);

*/


		if (solverTimers){
			time4 = MPI_Wtime();
			preconTime+=(time4-time3);
			matvecPreconTime += (time4-time3);
		}
	}//if use precond
	else{

		hypre_ParKrylovCopyVectorOneOfMult(bb, 0, p, 0);
		hypre_ParKrylovMatvecMult(matvec_data,
				minusone,
				A,
				xx,
				0,
				one,
				p, 0);
	}//else use precond

	if (GSoption != 0){
		//	cudaFree(tempH);

		//	cudaFree(tempRV);
	}

	//	cudaFree(x_GPUonly);
	//	cudaFree(b_GPUonly);

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

