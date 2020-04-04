#define solverTimers 1
#define usePrecond 1
#define leftPrecond 0
#define flexiblePrecond 0 // for flexible
#define GPUVcycle 1
/******************************************************************************
 *
 * COGMRES cogmres
 *
 *****************************************************************************/



#include "krylov.h"
#include "_hypre_utilities.h"
//#include "_hypre_parcsr_ls.h"
//#include "_hypre_parcsr_ls.h"
#ifdef HYPRE_USING_GPU
#include "../seq_mv/gpukernels.h"
#endif
static HYPRE_Int HegedusTrick=0; 
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
			void *       (*CreateMultiVector)  (void *vectors, HYPRE_Int size ),
			void *       (*UpdateVectorCPU)  ( void *vector ),
			HYPRE_Int    (*DestroyVector) ( void *vector ),
			void *       (*MatvecCreate)  ( void *A, void *x ),
			HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
				void *x,HYPRE_Int k1, HYPRE_Complex beta, void *y, HYPRE_Int k2 ),
			HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
			HYPRE_Real   (*InnerProd)     ( void *x,HYPRE_Int i1,  void *y, HYPRE_Int i2 ),
			HYPRE_Int    (*MassInnerProd) (void *x,HYPRE_Int k1, void *y, HYPRE_Int k2, void *result),
			HYPRE_Int    (*MassInnerProdTwoVectors) ( void *x,HYPRE_Int k, void *y1, HYPRE_Int k1, void *y2, HYPRE_Int k2, void *result),
			HYPRE_Int    (*MassInnerProdWithScaling)   ( void *x, HYPRE_Int i1, void *y,HYPRE_Int i2, void *scaleFactors, void *result),
			HYPRE_Int   (*DoubleInnerProd)     ( void *x,HYPRE_Int i1,  void *y, HYPRE_Int i2, void * res ),
			HYPRE_Int    (*CopyVector)    ( void *x,HYPRE_Int i1, void *y, HYPRE_Int i2 ),
			HYPRE_Int    (*ClearVector)   ( void *x ),
			HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x, HYPRE_Int i1 ),
			HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, HYPRE_Int k1, void *y, HYPRE_Int k2 ),      
			HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void *x, HYPRE_Int k1, void *y, HYPRE_Int k2),   
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
	cogmres_functions->CreateMultiVector = CreateMultiVector; /* not in PCGFunctionsCreate */
	cogmres_functions->UpdateVectorCPU = UpdateVectorCPU; /* not in PCGFunctionsCreate */
	cogmres_functions->DestroyVector     = DestroyVector;
	cogmres_functions->MatvecCreate      = MatvecCreate;
	cogmres_functions->Matvec            = Matvec;
	cogmres_functions->MatvecDestroy     = MatvecDestroy;
	cogmres_functions->InnerProd         = InnerProd;
	cogmres_functions->MassInnerProd     = MassInnerProd;	
	cogmres_functions->MassInnerProdTwoVectors     = MassInnerProdTwoVectors;	
	cogmres_functions->MassInnerProdWithScaling       = MassInnerProdWithScaling;
	cogmres_functions->DoubleInnerProd       = DoubleInnerProd;
	cogmres_functions->CopyVector        = CopyVector;
	cogmres_functions->ClearVector       = ClearVector;
	cogmres_functions->ScaleVector       = ScaleVector;
	cogmres_functions->Axpy              = Axpy;
	cogmres_functions->MassAxpy          = MassAxpy;
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

	(cogmres_data -> GSoption)  = 4;
	return (void *) cogmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESDestroy
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESDestroy( void *cogmres_vdata )
{

	size_t mf, ma;
	//cudaMemGetInfo(&mf, &ma);
	//printf("starting de-allocation, free memory %zu allocated memory %zu \n", mf, ma);
	//printf("DESTROYING COGMRES, free memory %zu total memory %zu percentage of allocated %f \n", mf, ma, (double)mf/ma);

	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;

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
			//printf("freeing p\n");
			// cudaFree(cogmres_data -> p);
			(*(cogmres_functions->DestroyVector))(cogmres_data -> p);
		}

		if ( (cogmres_data -> z) != NULL )
		{
			//printf("freeing p\n");
			// cudaFree(cogmres_data -> p);
			(*(cogmres_functions->DestroyVector))(cogmres_data -> z);
		}
		hypre_TFreeF( cogmres_data, cogmres_functions );
		hypre_TFreeF( cogmres_functions, cogmres_functions );
	}

	//cudaMemGetInfo(&mf, &ma);
	//printf("starting de-allocation, free memory %zu allocated memory %zu \n", mf, ma);
	//printf("\n ENDED COGMRES, free memory %zu total memory %zu percentage of allocated %f \n", mf, ma, (double)mf/ma);
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
	HYPRE_Int max_iter                                  = (cogmres_data -> max_iter);
	HYPRE_Int (*precond_setup)(void*,void*,void*,void*) = (cogmres_functions->precond_setup);
	void       *precond_data                            = (cogmres_data -> precond_data);



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

	PUSH_RANGE("cogmres precon SETUP", 2);
	precond_setup(precond_data, A, b, x);

	POP_RANGE;
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
	//
	//#define IDX2C(i,j,ld) (((i)*(ld))+(j))
	return r*n+c;
}

/*======
 * KS: THIS IS for using various orth method; user can choose some options 
 * ============================*/

//#define GSoption 1
// 0 is modified gram-schmidt as in "normal" GMRES straight from Saad's book
// 1 is CGS-1 with Ghysells norm (stabiity cannot be assured)
// 3 is CGS-2 (reorhogonalized Classical Gram Schmidti, 3 synch version)
// 4 is 2-synch Modified Gram Schmidt (triangular solve version but without (important) delayed norm
// 5 is 1-synch Modifid Gram Schmidt (norm is delayed)
// <6 is 1 synch Classical Gram Schmidt with delayed norm 
// allocate spaces for GPU (copying does not happen inside) prior to calling

void GramSchmidt (HYPRE_Int option,
		HYPRE_Int i,
		HYPRE_Int k_dim,
		void * Vspace,
		HYPRE_Real * Hcolumn,
		HYPRE_Real * HcolumnGPU,  
		HYPRE_Real* rv, 
		HYPRE_Real * rvGPU,
		HYPRE_Real * L,
		hypre_COGMRESFunctions *cf){

	HYPRE_Int j;
	HYPRE_Real t;
	// MODIFIED GRAM SCHMIDT, (i-1) synchs (textbook) version, 
	if (option == 0){
		for (j=0; j<i; ++j){ 
			Hcolumn[idx( i-1,j, k_dim+1)] =(*(cf->InnerProd))(Vspace, 
					j, 
					Vspace, 
					i);
			//printf("H[%d, %d] =H[%d] =  %16.16f \n ",j,i-1,idx( i-1,j, k_dim+1), Hcolumn[idx( i-1,j, k_dim+1)]  ); 
			(*(cf->Axpy))((-1.0)*Hcolumn[idx( i-1,j, k_dim+1)], Vspace, j, Vspace, i);		
		}

		t = sqrt((*(cf->InnerProd))(Vspace,i,Vspace, i));
		Hcolumn[idx( i-1,i, k_dim+1)] = t;

		//printf("H[%d, %d] =H[%d] =  %16.16f \n ",i-1,i,idx( i-1,i, k_dim+1), Hcolumn[idx( i-1,i, k_dim+1)]  ); 
		if (t != 0){
			t = 1/t;
			(*(cf->ScaleVector))(t,Vspace, i);
		}

	}

	/*CGS-1 with Ghysels norm estimate */
	if (option == 1){
		//t = \|V_i\|

		if (i>0) {
			//first synch
			(*(cf->DoubleInnerProd)) (Vspace,i-1, Vspace, i, &rv[i-1]);

			double dd = 2.0f - rv[i-1];
			cudaMemcpy (&rvGPU[i-1], &dd,
					sizeof(HYPRE_Real),
					cudaMemcpyHostToDevice );
			t= sqrt(rv[i]);
		}
		else{

			t  = sqrt((*(cf->InnerProd))(Vspace,i,Vspace, i));
		}
		// H_i = rvGPU^T*(V_{i-1}^Tv_i)
		//second synch    
		(*(cf->MassInnerProdWithScaling)) (Vspace,i, Vspace, i,rvGPU, HcolumnGPU);

		cudaMemcpy ( &Hcolumn[idx( i-1,0,k_dim+1)],
				HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		// v_i = v_i - V_{i-1}^TH_i = v_i - rvGPU^TV_{i-1}^T v_i = (I-rvGPU^T V_{i-1}^T) v_i 

		(*(cf->MassAxpy))( HcolumnGPU,Vspace,i, Vspace, i);


		HYPRE_Real t2= 0.0f;
		// t2 = \|H_i\|
		for (j=0; j<i; j++){
			HYPRE_Int id = idx( i-1,j,k_dim+1);
			t2          += (Hcolumn[id]*Hcolumn[id]);        
		}
		t2 = sqrt(t2)*sqrt(rv[i-1]);

		Hcolumn[idx(i-1, i, k_dim+1)] = sqrt(t-t2)*sqrt(t2+t);
		if (Hcolumn[idx( i-1,i, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i-1, i, k_dim+1)]; 
			(*(cf->ScaleVector))(t,Vspace, i);

			//  rv[i]  = (*(cf->InnerProd))(Vspace,i,Vspace, i);
			//rv[i] = 1.0/t;     
		}//if
	}

	if (option == 2){
		//HcolumnGPU =V_{i-1}^Tv_i = V_{i-1}^Ta (first synch)
		(*(cf->MassInnerProd))(Vspace,i, Vspace, i, HcolumnGPU);

		cudaMemcpy ( &Hcolumn[idx( i-1,0,k_dim+1)],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//v_i = v_i - V_{i-1}HcolumnGPU = v_i - V_{i-1}V_{i-1}^Tv_i

		(*(cf->MassAxpy))( HcolumnGPU,Vspace,i, Vspace, i);
		//second orth
		//HcolumnGPU = = V_{i-1}^Tv_i = V_{i-1}^Ta (second synch)
		(*(cf->MassInnerProd))(Vspace,i, Vspace, i, HcolumnGPU);
		cudaMemcpy ( &rv[0],HcolumnGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );
		//update Hcolumn a.k.a R in QR factorization
		for (j=0; j<i; j++){
			HYPRE_Int id = idx( i-1,j,k_dim+1);
			Hcolumn[id]+=rv[j];        
		}

		//v_i = v_i - V_{i-1}HcolumnGPU = v_i - V_{i-1}V_{i-1}^Tv_i
		(*(cf->MassAxpy))( HcolumnGPU,Vspace,i, Vspace, i);

		//norm (third synch)
		t  = sqrt((*(cf->InnerProd))(Vspace,i,Vspace, i));
		Hcolumn[idx(i-1,i, k_dim+1)] = t;
		if (Hcolumn[idx( i-1,i, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx(i-1,i, k_dim+1)]; 
			(*(cf->ScaleVector))(t,Vspace, i);
		}//if

	}

	if (option == 3){
		//Tri solve MGS WITHOUT delayed norm (so 2 synchs, not 1)
		//C version of orth_cgs2_new by ST
		//remember: U NEED L IN THIS CODE AND NEED TO KEEP IT!!
		//
		//L(1:i, i) = V(:, 1:i)'*V(:,i); (first synch)
		// [rvGPU] = [V_{i-1} a]^T [v_{i-1} a]
		(*(cf->MassInnerProdTwoVectors))(Vspace,i, Vspace, i-1, Vspace, i, rvGPU);

		//copy rvGPU to L

		cudaMemcpy ( &L[(i-1)*(k_dim+1)],rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		cudaMemcpy ( rv,&rvGPU[i],
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		for (int j=0; j<i; ++j){
			Hcolumn[(i-1)*(k_dim+1)+j] = 0.0f;
		}
		//triangular solve
		for (int j=0; j<i; ++j){
			for (int k=0; k<i; ++k){
				//we are processing H[j*(k_dim+1)+k]
				if (j==k){Hcolumn[(i-1)*(k_dim+1)+k] += L[j*(k_dim+1)+k]*rv[j];
				} 

				if (j<k){
					Hcolumn[(i-1)*(k_dim+1)+j] -= L[k*(k_dim+1)+j]*rv[k];
				}
				if (j>k){
					Hcolumn[(i-1)*(k_dim+1)+j] -= L[j*(k_dim+1)+k]*rv[k];
				}
			}//for k 
		}//for j

		cudaMemcpy ( HcolumnGPU,&Hcolumn[(i-1)*(k_dim+1)],
				i*sizeof(HYPRE_Real),
				cudaMemcpyHostToDevice);

		(*(cf->MassAxpy))( HcolumnGPU,Vspace,i, Vspace, i);
		//normalize (second synch)
		t  = sqrt((*(cf->InnerProd))(Vspace,i,Vspace, i));
		Hcolumn[idx(i-1,i, k_dim+1)] = t;
		if (Hcolumn[idx( i-1,i, k_dim+1)] != 0.0)
		{
			t = 1.0/Hcolumn[idx( i-1,i, k_dim+1)]; 
			(*(cf->ScaleVector))(t,Vspace, i);
		}//if

	}


	if (option == 4){
		//one synch triang solve MGS version

		//the ONLY synch

		(*(cf->MassInnerProdTwoVectors))(Vspace,i, Vspace, i-1, Vspace, i, rvGPU);

		//FIRST COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j-1), goes to COLUMN of L  in original code 
		cudaMemcpy ( &L[(i-1)*(k_dim+1)],rvGPU,
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		//SECOND COLUMN OF rvGPU, i.e., Q(:,1:j-1)'*Q(:,j), goes to r_i in original code
		//while r is a matrix and R=L^T, we need only to operate on one column at a time. 
		cudaMemcpy ( rv,&rvGPU[i],
				i*sizeof(HYPRE_Real),
				cudaMemcpyDeviceToHost );

		double t = sqrt(L[(i-1)*(k_dim+1)+(i-1)]);
		L[(i-1)*(k_dim+1)+(i-1)] = t;
		//scale 
		if ((i-2)>=0){
			Hcolumn[(i-2)*(k_dim+1) +i-1] = t ;
			//printf("putting %f in H(%d,%d) \n", i-1, i-2 );
		}    
		for (int j=0; j<i; ++j){
			rv[j] = rv[j]/t;
			if (j<i-1)
				L[(i-1)*(k_dim+1)+j] /= t;


		}
		rv[i-1]=rv[i-1]/t;
		L[(i-1)*(k_dim+1)+i-1] =1.0f;

		for (int j=0; j<i; ++j){
			Hcolumn[(i-1)*(k_dim+1)+j] = 0.0f;
		}
		t=1.0/t;
		if(i-1>0){
#if 0
			printf("scaling V(%d) by %16.16f where 1/t = %16.16f norm of V(%d) is %16.16f (before scaling)\n",i-1,t, 1/t, i-1,  (*(cf->InnerProd))(Vspace, 
						i-1, 
						Vspace, 
						i-1));
#endif
			(*(cf->ScaleVector))(t,Vspace, i-1);

		}
#if 0
		printf("scaling CURRENT vector %d; before scaling the norm is %16.16f, sclaing gfactor %f \n", i, (*(cf->InnerProd))(Vspace, 
					i, 
					Vspace, 
					i),t);
#endif
		(*(cf->ScaleVector))(t,Vspace, i);
		//triangular solve
		for (int j=0; j<i; ++j){
			for (int k=0; k<i; ++k){
				//we are processing H[j*(k_dim+1)+k]
				if (j==k){Hcolumn[(i-1)*(k_dim+1)+k] += L[j*(k_dim+1)+k]*rv[j];
				} 

				if (j<k){
					Hcolumn[(i-1)*(k_dim+1)+j] -= L[k*(k_dim+1)+j]*rv[k];
				}
				if (j>k){
					Hcolumn[(i-1)*(k_dim+1)+j] -= L[j*(k_dim+1)+k]*rv[k];
				}
			}//for k 
		}//for j
		// for (int j=0; j<i; ++j)
		cudaMemcpy ( HcolumnGPU,&Hcolumn[(i-1)*(k_dim+1)],
				i*sizeof(HYPRE_Real),
				cudaMemcpyHostToDevice);

		(*(cf->MassAxpy))( HcolumnGPU,Vspace,i, Vspace, i);

	}//if

	if (option == 5){
		//one synch CGS-2 version
	}//if
}


/*--------------------------------------------------------------------------
 * hypre_COGMRESSolve
 *-------------------------------------------------------------------------*/



HYPRE_Int hypre_COGMRESSolve(void  *cogmres_vdata,
		void  *A,
		void  *b,
		void  *x)
{


	size_t mf, ma;
	//cudaMemGetInfo(&mf, &ma);
	//printf("starting de-allocation, free memory %zu allocated memory %zu \n", mf, ma);
	//printf("STARTING COGMRES, free memory %zu total memory %zu percentage of allocated %f \n", mf, ma, (double)mf/ma);
	HYPRE_Real time1, time2, time3, time4;
	HYPRE_Real gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
	HYPRE_Real massAxpyTime = 0.0; 
	HYPRE_Real gsOtherTime  = 0.0f;
	HYPRE_Real massIPTime   = 0.0f, preconTime = 0.0f, mvTime = 0.0f;    
	HYPRE_Real initTime     = 0.0f;
	if (solverTimers)
		time1                                     = MPI_Wtime(); 

	hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
	hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
	HYPRE_Int               k_dim             = (cogmres_data -> k_dim);
	HYPRE_Int               max_iter          = (cogmres_data -> max_iter);
	HYPRE_Real              r_tol             = (cogmres_data -> tol);
	HYPRE_Real              a_tol             = (cogmres_data -> a_tol);
	void                   *matvec_data       = (cogmres_data -> matvec_data);
	//hypre_ParVector * w, *p, *r ;

	//printf("Starting VERY BEGINNING norm x %16.16f \n", sqrt((*(cogmres_functions->InnerProd))(x,0,x, 0)));
	//printf("Starting VERY BEGINNING norm b %16.16f \n", sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0)));
	void         *w                 = (cogmres_data -> w);
	void         *w_2               = (cogmres_data -> w_2); 

	void        *p                 = (cogmres_data -> p);
	void        *z                 = (cogmres_data -> z);

	HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
	HYPRE_Int  *precond_data                      = (HYPRE_Int*)(cogmres_data -> precond_data);

	HYPRE_Real     *norms          = (cogmres_data -> norms);
	HYPRE_Int print_level = (cogmres_data -> print_level);
	HYPRE_Int logging     = (cogmres_data -> logging);

	HYPRE_Int GSoption = (cogmres_data -> GSoption);
	HYPRE_Int  i, j, k;
	//KS: rv is the norm history 
	HYPRE_Real *rs, *hh, *c, *s, *rv, *L;
	HYPRE_Int  iter; 
	HYPRE_Int  my_id, num_procs;
	HYPRE_Real epsilon, gamma, t, r_norm, b_norm, b_norm_original;

	HYPRE_Real epsmac = 1.e-16; 
	HYPRE_Int doNotSolve=0;
	// TIMERS/


	(cogmres_data -> converged) = 0;

	(*(cogmres_functions->CommInfo))(A,&my_id,&num_procs);
	if ( logging>0 || print_level>0 )
	{
		norms          = (cogmres_data -> norms);
	}

	//  if (usePrecond){
	(cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
	w_2 = cogmres_data -> w_2;
	// }

	(cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);
	w = cogmres_data -> w;
	//CreateVecrtie in this case will initialize it too  
	//r = (*(cogmres_functions->CreateVector))(b);
	rs = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST);
	c  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);
	s  = hypre_CTAllocF(HYPRE_Real,k_dim,cogmres_functions, HYPRE_MEMORY_HOST);

	rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_functions, HYPRE_MEMORY_HOST);


	HYPRE_Real * tempH;
	HYPRE_Real * tempRV;

	(cogmres_data -> p) = (*(cogmres_functions->CreateMultiVector))(b, k_dim+1);
	p = (cogmres_data->p);
	if (flexiblePrecond) {

		(cogmres_data -> z) = (*(cogmres_functions->CreateMultiVector))(b, k_dim+1);
		z = (cogmres_data->z);
	}
	if (GSoption != 0){
		if (GSoption <3){   
			cudaMalloc ( &tempRV, sizeof(HYPRE_Real)*(k_dim+1)); 
			cudaMalloc ( &tempH, sizeof(HYPRE_Real)*(k_dim+1)); 
		}
		else {

			cudaMalloc ( &tempRV, 2*sizeof(HYPRE_Real)*(k_dim+1)); 
			cudaMalloc ( &tempH, 2*sizeof(HYPRE_Real)*(k_dim+1)); 
		}
	}


	hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);

	if (GSoption >=3){
		L = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
	}
	HYPRE_Real one = 1.0f, minusone = -1.0f, zero = 0.0f;

#if 1

	if ( print_level>1 && my_id == 0 )
	{
		hypre_printf("=============================================\n\n");
		hypre_printf("Iters     resid.norm     conv.rate  \n");
		hypre_printf("-----    ------------    ---------- \n");
	}



	if (solverTimers){
		time2 = MPI_Wtime();
		initTime += (time2-time1);
	}
	//outer loop 

	iter = 0;
	//works

	while (iter < max_iter)
	{

		if (iter == 0){
			if (solverTimers){
				time1 = MPI_Wtime();
			}
			b_norm_original =  sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0));
			b_norm = b_norm_original;

			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime += (time2-time1);
			}
			if ((usePrecond) && (leftPrecond)){
				if (solverTimers){
					time1 = MPI_Wtime();
				}
				(*(cogmres_functions->UpdateVectorCPU))(b);
				(*(cogmres_functions->ClearVector))(w_2);
				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime += (time2-time1); 
					time1 = MPI_Wtime();
				}
				precond(precond_data, A, b,w_2 );

				if (solverTimers){
					time2 = MPI_Wtime();
					matvecPreconTime+=(time2-time1);
					preconTime += (time2-time1);
					time1 = MPI_Wtime();
				}
				(*(cogmres_functions->UpdateVectorCPU))(w_2);
				(*(cogmres_functions->CopyVector))(w_2, 0, b, 0);
				(*(cogmres_functions->UpdateVectorCPU))(b);
				b_norm =  sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0));
				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime += (time2-time1);
				}
			}

		}
		/*******************************************************************
		 * RESTART, PRECON IS RIGHT
		 * *****************************************************************/    

		if ((usePrecond)&&(!leftPrecond)){
			if(!flexiblePrecond){
				if (solverTimers)
					time1 = MPI_Wtime();

				(*(cogmres_functions->ClearVector))(w);
				//(*(cogmres_functions->ClearVector))(w_2);
				//KS: if iter == 0, x has the right CPU data, no need to copy	
				//not true if restarting
				if(iter!=0){
					(*(cogmres_functions->UpdateVectorCPU))(x);
				}	
				//w = Mx
				(*(cogmres_functions->CopyVector))(x, 0, w_2, 0);
				//	if (iter==0) printf("ITER 0, norm of x  (before precon) %16.16f\n ",sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)));
				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime += (time2-time1);
					time3 = MPI_Wtime();
				}

				PUSH_RANGE("cogmres precon", 0);
				precond(precond_data, A, w_2,w );
				POP_RANGE;

				//	if (iter==0) printf("ITER 0, norm after applying precon %16.16f\n ",sqrt((*(cogmres_functions->InnerProd))(w,0,w, 0)));

				if (solverTimers){
					time4 = MPI_Wtime();
					preconTime +=(time4-time3);
					matvecPreconTime += (time4-time3);

					time1 = MPI_Wtime();
				}
				//w_2 = AMx = Aw   

				(*(cogmres_functions->Matvec))(matvec_data,
						one,
						A,
						w,
						0,
						zero,
						w_2, 0);

				//	if (iter==0) printf("ITER 0, norm after applying matvec %16.16f\n ",sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)));
				if (solverTimers){
					time2 = MPI_Wtime();
					mvTime +=(time2-time1);
					matvecPreconTime += (time2-time1);
					time1 = MPI_Wtime();      
				}
				//use Hegedus, if indicated AND first cycle

				HYPRE_Real part2 = (*(cogmres_functions->InnerProd))(w_2,0,w_2, 0);
				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime +=(time2-time1);
				}
				if (part2 == 0.0f){
					//safety check - cant divide by 0
					HegedusTrick = 0;
				}      
				if ((HegedusTrick)&&(iter==0)){

					if (solverTimers){
						time1 = MPI_Wtime();
						remainingTime +=(time2-time1);
					}
					HYPRE_Real part1 = (*(cogmres_functions->InnerProd))(w_2,0,b, 0);



					(*(cogmres_functions->ScaleVector))(part1/part2,x, 0);
					(*(cogmres_functions->UpdateVectorCPU))(x);

					//w = Mx_0

					(*(cogmres_functions->ClearVector))(w);

					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime +=(time2-time1);
						time1 = MPI_Wtime();     
					}
					precond(precond_data, A, x,w );

					if (solverTimers){
						time2 = MPI_Wtime();
						preconTime +=(time2-time1);
						matvecPreconTime += (time2-time1);
						time1 = MPI_Wtime();      
					}
					(*(cogmres_functions->CopyVector))(b, 0, p, 0);
					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime +=(time2-time1);
						time1 = MPI_Wtime();     
					}
					(*(cogmres_functions->Matvec))(matvec_data,
							minusone,
							A,
							w,
							0,
							one,
							p, 0);
					if (solverTimers){
						time2 = MPI_Wtime();
						mvTime +=(time2-time1);
						matvecPreconTime += (time2-time1);
					}
				}//if Hegedus
				else {
					//not using Hegedus, compute the right residual

					if (solverTimers){
						time1 = MPI_Wtime();
					}
					(*(cogmres_functions->CopyVector))(b, 0, p, 0);
					(*(cogmres_functions->Axpy))(-1.0f, w_2, 0, p, 0);
					//printf("Starting with NON FLEXIBLE  norm p(0) %16.16f \n", sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0)));
					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime +=(time2-time1);
					}


				}    
			}// not flexible precon
			else {//flexible precon
				//no Hegedus here

				if (solverTimers)
					time1 = MPI_Wtime();

				(*(cogmres_functions->ClearVector))(w);
				(*(cogmres_functions->CopyVector))(b, 0, p, 0);
				if(iter!=0){
					(*(cogmres_functions->UpdateVectorCPU))(x);
				}	


				if (solverTimers){
					time4 = MPI_Wtime();
					remainingTime +=(time1-time4);

					time1 = MPI_Wtime();
				}
				//p(0) = 1*p(0)-A*x = b-Ax   

				(*(cogmres_functions->Matvec))(matvec_data,
						minusone,
						A,
						x,
						0,
						one,
						p, 0);

				//printf("Starting with FLEXIBLE  norm p(0) %16.16f \n", sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0)));
				if (solverTimers){
					time2 = MPI_Wtime();
					mvTime +=(time2-time1);
					matvecPreconTime += (time2-time1);
					time1 = MPI_Wtime();      
				}
			}//flexible    


		}// if RIGHT Precond

		/*******************************************************************
		 * RESTART, PRECON IS LEFT
		 * *****************************************************************/    

		if ((usePrecond)&&(leftPrecond)){
			if (solverTimers){
				time1 = MPI_Wtime();
			}
			(*(cogmres_functions->Matvec))(matvec_data,
					one,
					A,
					x,
					0,
					zero,
					w, 0);

			if (solverTimers){
				time2 = MPI_Wtime();
				mvTime +=(time2-time1);
				matvecPreconTime += (time2-time1);
				time1 = MPI_Wtime();	
			}

			(*(cogmres_functions->UpdateVectorCPU))(w);
			(*(cogmres_functions->ClearVector))(w_2);



			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime +=(time2-time1);
				time1 = MPI_Wtime();
			}
			precond(precond_data, A, w,w_2 );

			if (solverTimers){
				time2 = MPI_Wtime();
				preconTime +=(time2-time1);
				matvecPreconTime += (time2-time1);
				time1 = MPI_Wtime();
			}
			HYPRE_Real part2 = (*(cogmres_functions->InnerProd))(w_2,0,w_2, 0);
			if (part2 == 0.0f){HegedusTrick = 0;}     
			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime +=(time2-time1);
			}
			if ((HegedusTrick)&&(iter==0)){
				//(Mb)'*(MAx)/\|MAx\|^2 = b'w_2/[(w_2)'*(w_2)] <-- scaling factor

				if (solverTimers){
					time2 = MPI_Wtime();
				}
				HYPRE_Real part1 = (*(cogmres_functions->InnerProd))(b,0,w_2, 0);



				(*(cogmres_functions->ScaleVector))(part1/part2,x, 0);
				(*(cogmres_functions->UpdateVectorCPU))(x);
				//update w_2 = MAx_0

				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime +=(time2-time1);
					time1 = MPI_Wtime();
				}
				(*(cogmres_functions->Matvec))(matvec_data,
						one,
						A,
						x,
						0,
						zero,
						w, 0);

				if (solverTimers){
					time2 = MPI_Wtime();
					mvTime +=(time2-time1);
					matvecPreconTime += (time2-time1);
					time1 = MPI_Wtime();	
				}

				(*(cogmres_functions->UpdateVectorCPU))(w);
				(*(cogmres_functions->ClearVector))(w_2);



				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime +=(time2-time1);
					time1 = MPI_Wtime();
				}
				precond(precond_data, A, w,w_2 );

				if (solverTimers){
					time2 = MPI_Wtime();
					preconTime +=(time2-time1);
					matvecPreconTime += (time2-time1);
				}

			}//Hegedus




			if (solverTimers){
				time1 = MPI_Wtime();
			}
			(*(cogmres_functions->CopyVector))(b, 0, p, 0);
			(*(cogmres_functions->Axpy))(-1.0f, w_2, 0, p, 0);		



			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime +=(time2-time1);
			}
		}//LEFT precond




		/*******************************************************************
		 * RESTART, PRECON IS NONE
		 * *****************************************************************/    
		if (!usePrecond){
			if (solverTimers){
				time1 = MPI_Wtime();
			}
			(*(cogmres_functions->CopyVector))(b, 0, p, 0);
			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime +=(time2-time1);
				time1 = MPI_Wtime();
			}

			(*(cogmres_functions->Matvec))(matvec_data,
					minusone,
					A,
					x,
					0,
					one,
					p, 0);
			if (solverTimers){
				time2 = MPI_Wtime();
				mvTime +=(time2-time1);
				matvecPreconTime += (time2-time1);
			}
		}//no precon

		if (solverTimers){
			time1 = MPI_Wtime();
		}
		r_norm = sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0));
		if (iter == 0){      
			epsilon = r_tol*b_norm;
			//hypre_max(a_tol,r_tol*r_norm);
		}   
		if ( logging>0 || print_level > 0)
		{
			norms[iter] = r_norm;
		}
		if (my_id == 0  ){
			if (iter == 0){
#if 1
				hypre_printf("Orthogonalization variant: %d \n", GSoption);
				hypre_printf("L2 norm of b: %16.16f\n", b_norm);
				hypre_printf("Initial L2 norm of (current) residual: %16.16f\n", r_norm);
#endif
			}
		}


		// conv criteria 
		if (solverTimers){
			time2 = MPI_Wtime();

			remainingTime += (time2-time1);
		}

		if (r_norm <epsilon){

			(cogmres_data -> converged) = 1;
			break;
		}//conv check
		//iter++;

		if (solverTimers)
			time1 = MPI_Wtime();
#if 0
		printf("BEFORE scaling by %f, the norm of the first vector is %f \n",1.0f/r_norm, (*(cogmres_functions->InnerProd))(p, 
					0, 
					p, 
					0));
		printf("residual norm %16.16f \n", r_norm);
#endif   
		t = 1.0f/r_norm;

		(*(cogmres_functions->ScaleVector))(t,p, 0);
#if 0
		printf("after scaling by %f, the norm of the first vector is %f \n",1.0/r_norm, (*(cogmres_functions->InnerProd))(p, 
					0, 
					p, 
					0));
#endif
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

		//inner loop 
		i = -1; 
		while (i+1 < k_dim && iter < max_iter)
		{
			i++;
			iter++;
			if ((usePrecond) && !(leftPrecond)){
				//x = M * p[i-1]
				//p[i] = A*x

				if (solverTimers)
					time1 = MPI_Wtime();
				(*(cogmres_functions->CopyVector))(p, i, w_2, 0);
				//clear vector is absolutely necessary
				(*(cogmres_functions->ClearVector))(w);

				if (solverTimers){
					time2 = MPI_Wtime();
					remainingTime += (time2-time1);
					time3 = MPI_Wtime();
				}

				PUSH_RANGE("cogmres precon", 1);
				//printf("BEFORE PRECOND, norm w2(%d) %16.16f norm w (output) %16.16f \n",i, sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)),  sqrt((*(cogmres_functions->InnerProd))(w,0,w, 0)));
				precond(precond_data, A, w_2, w);
				if (flexiblePrecond){ 
					(*(cogmres_functions->CopyVector))(w, 0, z, i);
					//printf("FLEXIBLE  norm z(%d) %16.16f \n",i, sqrt((*(cogmres_functions->InnerProd))(z,i,z, i)));

				}
				POP_RANGE;
				if (solverTimers){
					time4 = MPI_Wtime();

					preconTime += (time4-time3);
					matvecPreconTime += (time4-time3);
					time1 = MPI_Wtime();
				}

				//printf("iter %d, putting new stuff in  p %d \n", i, i+1);
				(*(cogmres_functions->Matvec))(matvec_data,
						one,
						A,
						w,
						0,
						zero,
						p, i+1);
				//printf("FLEXIBLE  norm p(%d) %16.16f \n",i+1, sqrt((*(cogmres_functions->InnerProd))(p,i+1,p, i+1)));
				if (solverTimers){

					time2 = MPI_Wtime();
					mvTime += (time2-time1);
					matvecPreconTime += (time2-time1);   
				}
			}
			else{
				if ((usePrecond) && (leftPrecond)){

					if (solverTimers)
						time1 = MPI_Wtime();

					(*(cogmres_functions->ClearVector))(w);

					(*(cogmres_functions->CopyVector))(p, i, w_2, 0);
					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime += (time2-time1);
						time1 = MPI_Wtime();
					}
					(*(cogmres_functions->Matvec))(matvec_data,
							one,
							A,
							w_2,
							0,
							zero,
							w, 0);

					if (solverTimers){
						time2 = MPI_Wtime();
						mvTime += (time2-time1);
						matvecPreconTime += (time2-time1);   
						time1 = MPI_Wtime();
					}

					if (solverTimers)
						time1 = MPI_Wtime();
					(*(cogmres_functions->UpdateVectorCPU))(w);

					(*(cogmres_functions->ClearVector))(w_2);


					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime += (time2-time1);
						time1 = MPI_Wtime();
					}
					PUSH_RANGE("cogmres precon", 2);
					precond(precond_data, A, w, w_2);
					POP_RANGE;
					if (solverTimers){
						time2 = MPI_Wtime();
						preconTime += (time2-time1);
						matvecPreconTime += (time2-time1);   
						time1 = MPI_Wtime();
					}



					(*(cogmres_functions->CopyVector))(w_2, 0, p, i+1);
					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime += (time2-time1);
					}
				}
				else{
					// not using preconddd
					if (solverTimers)
						time1 = MPI_Wtime();

					(*(cogmres_functions->CopyVector))(p, i, w, 0);
					if (solverTimers){
						time2 = MPI_Wtime();
						remainingTime += (time2-time1);
						time1 = MPI_Wtime();
					}

					(*(cogmres_functions->Matvec))(matvec_data,
							one,
							A,
							w,
							0,
							zero,
							p, i+1);
					if (solverTimers){
						time2 = MPI_Wtime();
						mvTime += (time2-time1);
						matvecPreconTime += (time2-time1);   
					}

				}
			}
			if (solverTimers){

				time1=MPI_Wtime();
			}


			// GRAM SCHMIDT 
			if (GSoption == 0){

				GramSchmidt (0, 
						i+1, 
						k_dim, 
						p,
						hh, 
						NULL,  
						rv, 
						NULL, NULL, cogmres_functions );
			}

			if ((GSoption >0) &&(GSoption <3)){
				GramSchmidt (GSoption, 
						i+1, 
						k_dim,
						p, 
						hh, 
						tempH,  
						rv, 
						tempRV, NULL, cogmres_functions );
			}


			if ((GSoption ==3)){

				GramSchmidt (GSoption, 
						i+1, 
						k_dim,
						p, 
						hh, 
						tempH,  
						rv, 
						tempRV, L, cogmres_functions );

			}

			if ((GSoption >3)){

				GramSchmidt (GSoption, 
						i+1, 
						k_dim,
						p, 
						hh, 
						tempH,  
						rv, 
						tempRV, L, cogmres_functions );
				if (flexiblePrecond) { 
					if (i>0){
						//printf("FLEXIBLE  norm z(%d) before scaling %16.16f scaling factor: 1/%16.16ff taken from h(%d) \n",i, sqrt((*(cogmres_functions->InnerProd))(z,i,z, i)), hh[(i-1)*(k_dim+1) +i], (i-1)*(k_dim+1) +i);

						(*(cogmres_functions->ScaleVector))(1.0f/hh[(i-1)*(k_dim+1) +i],z, i);
						//printf("FLEXIBLE  norm z(%d) after scaling %16.16f \n",i, sqrt((*(cogmres_functions->InnerProd))(z,i,z, i)));
					}
					//divide i
				}

				if(i==0) {doNotSolve=1;iter--;}
				else doNotSolve = 0;

			}

			// CALL IT HERE
			if (solverTimers){
				time2 = MPI_Wtime();
				//gsOtherTime +=  time2-time3;
				gsTime += (time2-time1);
				time1 = MPI_Wtime();
			}
			if (!doNotSolve){
				if (GSoption >3) i--;
				for (j = 1; j <= i; j++)
				{
					HYPRE_Int j1=j-1;
					t = hh[idx(i,j1,k_dim+1)];
					hh[idx(i,j1,k_dim+1)] = s[j1]*hh[idx(i,j,k_dim+1)] + c[j1]*t;
					hh[idx(i,j, k_dim+1)]  = -s[j1]*t + c[j1]*hh[idx(i,j,k_dim+1)];
				}

				t     = hh[idx(i, i+1, k_dim+1)]*hh[idx(i,i+1, k_dim+1)];//Hii1
				t    += hh[idx(i,i, k_dim+1)]*hh[idx(i,i, k_dim+1)];//Hii
				gamma = sqrt(t);
				if (gamma == 0.0) gamma = epsmac;


				c[i]  = hh[idx(i,i, k_dim+1)]/gamma;
				s[i]  = hh[idx(i,i+1, k_dim+1)]/gamma;
				rs[i+1]   = -s[i]*rs[i];
				rs[i] = c[i]*rs[i];  

				// determine residual norm 
				hh[idx(i,i, k_dim+1)] = s[i]*hh[idx(i,i+1, k_dim+1)] + c[i]*hh[idx(i,i, k_dim+1)];
				r_norm = fabs(rs[i+1]);
				if (solverTimers){
					time4 = MPI_Wtime();
					linSolveTime += (time4-time1);
				}
#if 1
				if ( print_level>0 )
				{
					norms[iter] = r_norm;
					if ( print_level>1 && my_id == 0  )
					{
						HYPRE_Real sc;

						if (i == 0) {sc = 1.0;}
						else {sc = norms[iter-1];}
						hypre_printf("ITER: % 5d    %e    %f\n", iter, norms[iter],
								norms[iter]/sc);
					}
				}// if (printing)
#else

				norms[iter] = r_norm;
				HYPRE_Real sc;

				if (i == 0) {sc = 1.0;}
				else {sc = norms[iter-1];}

				if (my_id == 0 )hypre_printf("ITER: % 5d    %e    %f\n", iter, norms[iter],
						norms[iter]/sc);


#endif
				//printf("r_norm = %16.16e epsilon = %16.16e\n", r_norm, epsilon);
				if (r_norm <epsilon){

					(cogmres_data -> converged) = 1;
					break;

				}//conv check

				if (GSoption >3) i++;
			}//doNotSolve
		}//while (inner)

		if (solverTimers){
			time1 = MPI_Wtime();
		}
		int k1, ii;
		rs[i] = rs[i]/hh[idx(i,i,k_dim+1)];
		for (ii=2; ii<=i+1; ii++)
		{
			k  = i-ii+1;
			k1 = k+1;
			t  = rs[k];
			for (j=k1; j<=i; j++)
				t = t - hh[idx(j,k,k_dim+1)]*rs[j];

			rs[k] = t / hh[idx(k,k,k_dim+1)];
		}
		if (!flexiblePrecond){
			//printf("normal: computing sol. will be going from %d to %d \n", 0, i);
			for (j = 0; j <=i; j++){
				(*(cogmres_functions->Axpy))(rs[j], p, j, x, 0);		
			}}	
		else{
			// printf("flexible: computing sol. will be going from %d to %d \n", 0, i);
			for (j = 0; j <=i; j++){
				//printf("multiplying z(%d) by %16.16f \n", j, rs[j]);	
				(*(cogmres_functions->Axpy))(rs[j], z, j, x, 0);		
			}

		}	
		(*(cogmres_functions->UpdateVectorCPU))(x);

		if (solverTimers){
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
		}
#if 0

		(*(cogmres_functions->CopyVector))(b, 0, w_2, 0);
		(*(cogmres_functions->Matvec))(matvec_data,
				minusone,
				A,
				x,
				0,
				one,
				w_2, 0);
		printf("END of cycle: norm of residual: %16.16f \n",sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)));
#endif 

		/* debug mode */
#if 0
		//  (*(cogmres_functions->CopyVector))(x, 0, w, 0);
		//    (*(cogmres_functions->ClearVector))(w_2);
		//  precond(precond_data, A, w, w_2);

		(*(cogmres_functions->Matvec))(matvec_data,
				one,
				A,
				x,
				0,
				zero,
				w_2, 0);

		printf("Norm of w_2  %16.16f \n",  sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)));
		(*(cogmres_functions->UpdateVectorCPU))(w_2);
		(*(cogmres_functions->ClearVector))(w);
		if (usePrecond)    
			precond(precond_data, A, w_2, w);
		else{ 

			(*(cogmres_functions->CopyVector))(w_2, 0, w, 0);
			printf("Norm of w  %16.16f \n",  sqrt((*(cogmres_functions->InnerProd))(w,0,w, 0)));
			printf("Norm of b  %16.16f \n",  sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0)));
		}
		(*(cogmres_functions->Axpy))(1.0,b , 0, w_2, 0);		
		(*(cogmres_functions->Axpy))(-1.0,b , 0, w, 0);		
		printf("End of cycle!  norm of true res  %16.16f other way %16.16f norm of x %16.16f \n", sqrt((*(cogmres_functions->InnerProd))(w_2,0,w_2, 0)),sqrt((*(cogmres_functions->InnerProd))(w,0,w, 0)),  sqrt((*(cogmres_functions->InnerProd))(x,0,x, 0)));

		/*debug mode */
		hypre_ParVectorSetConstantValues((hypre_ParVector*) w_2, 1.0f);
		HYPRE_Real t1 = (*(cogmres_functions->InnerProd))(w_2,0,x, 0);

		if ( print_level>1 && my_id == 0 ){
			printf("END: IP of x and 1s is %16.16f \n", t1); 
		}

#endif
		/*end of debug mode */
		//test solution 
		if (r_norm < epsilon){

			(cogmres_data -> converged) = 1;
			break;
		}

		// final tolerance 


		if (solverTimers){
			time2 = MPI_Wtime();
			remainingTime += (time2-time1);
		}
	}
	if ((usePrecond)&&(!leftPrecond)){

		if (!flexiblePrecond){
			if (solverTimers){
				time1 = MPI_Wtime();
			}
			(*(cogmres_functions->CopyVector))(x, 0, w_2, 0);
			(*(cogmres_functions->ClearVector))(w);
			if (solverTimers){
				time2 = MPI_Wtime();
				remainingTime += (time2-time1);
				time1 = MPI_Wtime();
			}

			PUSH_RANGE("cogmres precon", 2);
			precond(precond_data, A, w_2, w);
			POP_RANGE;
			if (solverTimers){
				time2 = MPI_Wtime();
				matvecPreconTime+=(time2-time1);
				preconTime += (time2-time1);
			}
			//debug code
			(*(cogmres_functions->CopyVector))(w, 0, x, 0);
		}
#if 0
		(*(cogmres_functions->CopyVector))(b, 0, p, 0);
		(*(cogmres_functions->Matvec))(matvec_data,
				minusone,
				A,
				x,
				0,
				one,
				p, 0);
		printf("END: norm of residual: %16.16f \n",sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0)));
#endif   
		//end of debug code
	}//done only for right precond
#if 0
	double ttt;

	if (solverTimers){
		time4 = MPI_Wtime();
	}

	(*(cogmres_functions->Matvec))(matvec_data,
			one,
			A,
			x,
			0,
			zero,
			w_2, 0);

	(*(cogmres_functions->ClearVector))(w);

	(*(cogmres_functions->UpdateVectorCPU))(w_2);
	if (usePrecond) precond(precond_data, A, w_2, w);
	else 
		(*(cogmres_functions->CopyVector))(w_2, 0, w, 0);
	(*(cogmres_functions->Axpy))(-1.0f, b, 0, w, 0);		
	(*(cogmres_functions->Axpy))(1.0f, b, 0, w_2, 0);		
#endif

	if (GSoption != 0){
		cudaFree(tempH);
		cudaFree(tempRV);
	}

	//	cudaFree(x_GPUonly);
	//	cudaFree(b_GPUonly);



	if ((my_id == 0)&& (solverTimers)){
		hypre_printf("\n\nitersAll(%d,%d)                = %d \n", GSoption+1, k_dim/5, iter);
		hypre_printf("timeGSAll(%d,%d)                = %16.16f \n",GSoption+1, k_dim/5, gsTime);

		hypre_printf("TIME for CO-GMRES\n");
		hypre_printf("init cost            = %16.16f \n", initTime);
		hypre_printf("matvec+precon        = %16.16f \n", matvecPreconTime);
		hypre_printf("mv time              = %16.16f \n", mvTime);
		hypre_printf("precon multiply      = %16.16f \n", preconTime);
		hypre_printf("gram-schmidt (total) = %16.16f \n", gsTime);
		hypre_printf("linear solve         = %16.16f \n", linSolveTime);
		hypre_printf("all other            = %16.16f \n", remainingTime);

		hypre_printf("TOTAL:               = %16.16f \n", initTime+matvecPreconTime+gsTime+linSolveTime+remainingTime);
#if 0 
		hypre_printf("FINE times\n");
		hypre_printf("mass Inner Product   = %16.16f \n", massIPTime);
		hypre_printf("mass Axpy            = %16.16f \n", massAxpyTime);
		hypre_printf("Gram-Schmidt: other  = %16.16f \n", gsOtherTime);
		hypre_printf("precon multiply      = %16.16f \n", preconTime);
		hypre_printf("mv time              = %16.16f \n", mvTime);
		hypre_printf("MATLAB timers \n");
		hypre_printf("gsTime(%d) = %16.16f \n", k_dim/5, gsTime);	
		hypre_printf("timeAll(%d,%d)                =  \n\n",GSoption+1, k_dim/5);
#endif
	}
#endif
	if ((HegedusTrick == 0))
		HegedusTrick=1;


	//cudaMemGetInfo(&mf, &ma);
	//printf("starting de-allocation, free memory %zu allocated memory %zu \n", mf, ma);
	//printf("\n ENDING COGMRES(solve), free memory %zu total memory %zu percentage of allocated %f \n", mf, ma, (double)mf/ma);

   return 0;

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
 * hypre_COGMRESSetUnroll, hypre_COGMRESGetUnroll
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetUnroll( void   *cogmres_vdata,
		HYPRE_Int   unroll )
{
	hypre_COGMRESData *cogmres_data =(hypre_COGMRESData *) cogmres_vdata;
	(cogmres_data -> unroll) = unroll;
	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetUnroll( void   *cogmres_vdata,
		HYPRE_Int * unroll )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
	*unroll = (cogmres_data -> unroll);
	return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESSetCGS, hypre_COGMRESGetCGS
 *--------------------------------------------------------------------------*/

	HYPRE_Int
hypre_COGMRESSetCGS( void   *cogmres_vdata,
		HYPRE_Int   cgs )
{
	hypre_COGMRESData *cogmres_data =(hypre_COGMRESData *) cogmres_vdata;
	(cogmres_data -> cgs) = cgs;
	return hypre_error_flag;
}

	HYPRE_Int
hypre_COGMRESGetCGS( void   *cogmres_vdata,
		HYPRE_Int * cgs )
{
	hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
	*cgs = (cogmres_data -> cgs);
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




