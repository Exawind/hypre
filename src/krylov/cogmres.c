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
#ifdef HYPRE_NREL_CUDA
#include "_hypre_parcsr_ls.h"
#endif

#ifdef HYPRE_NREL_CUDA
#ifdef HYPRE_USING_GPU
#include "gpukernels.h"
#endif
static HYPRE_Int HegedusTrick=0;
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
      HYPRE_Int    (*DestroyVector) ( void *vector ),
      void *       (*MatvecCreate)  ( void *A, void *x ),
      HYPRE_Int    (*MatvecDestroy) ( void *matvec_data ),
      HYPRE_Int    (*ClearVector)   ( void *x ),
#ifdef HYPRE_NREL_CUDA
      void *       (*CreateMultiVector)  (void *vectors, HYPRE_Int num_vectors ),	  
      void *       (*UpdateVectorCPU)  ( void *vector ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A,
                               void *x,HYPRE_Int k1, HYPRE_Complex beta, void *y, HYPRE_Int k2 ),
      HYPRE_Real   (*InnerProd)     ( void *x,HYPRE_Int k1, void *y, HYPRE_Int k2 ),
      HYPRE_Int    (*MassInnerProd) ( void *x, HYPRE_Int k1, void *y, HYPRE_Int k2, void *result),
      HYPRE_Int    (*MassInnerProdTwoVectors) ( void *x,HYPRE_Int k, void *y1, HYPRE_Int k1, void *y2, HYPRE_Int k2, void *result),
      HYPRE_Int    (*MassInnerProdWithScaling)   ( void *x, HYPRE_Int i1, void *y,HYPRE_Int i2, void *scaleFactors, void *result),
      HYPRE_Int    (*DoubleInnerProd)     ( void *x,HYPRE_Int k1, void *y, HYPRE_Int k2, void * res ),
      HYPRE_Int    (*CopyVector)    ( void *x,HYPRE_Int i1,  void *y, HYPRE_Int i2 ),
      HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x, HYPRE_Int k1 ),
      HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x,HYPRE_Int k1, void *y, HYPRE_Int k2 ),
      HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void *x,HYPRE_Int i1, void *y, HYPRE_Int i2),
#else
      void *       (*CreateVectorArray)  ( HYPRE_Int size, void *vectors ),
      HYPRE_Int    (*Matvec)        ( void *matvec_data, HYPRE_Complex alpha, void *A, void *x, HYPRE_Complex beta, void *y ),
      HYPRE_Real   (*InnerProd)     ( void *x, void *y ),
      HYPRE_Int    (*MassInnerProd) ( void *x, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result),
      HYPRE_Int    (*MassDotpTwo)   ( void *x, void *y, void **p, HYPRE_Int k, HYPRE_Int unroll, void *result_x, void *result_y),
      HYPRE_Int    (*CopyVector)    ( void *x, void *y ),
      HYPRE_Int    (*ScaleVector)   ( HYPRE_Complex alpha, void *x ),
      HYPRE_Int    (*Axpy)          ( HYPRE_Complex alpha, void *x, void *y ),
      HYPRE_Int    (*MassAxpy)      ( HYPRE_Complex *alpha, void **x, void *y, HYPRE_Int k, HYPRE_Int unroll),
#endif
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
#ifdef HYPRE_NREL_CUDA
   cogmres_functions->CreateMultiVector = CreateMultiVector; /* not in PCGFunctionsCreate */
   cogmres_functions->UpdateVectorCPU = UpdateVectorCPU; /* not in PCGFunctionsCreate */
#else
   cogmres_functions->CreateVectorArray = CreateVectorArray; /* not in PCGFunctionsCreate */
#endif
   cogmres_functions->DestroyVector     = DestroyVector;
   cogmres_functions->MatvecCreate      = MatvecCreate;
   cogmres_functions->Matvec            = Matvec;
   cogmres_functions->MatvecDestroy     = MatvecDestroy;
   cogmres_functions->InnerProd         = InnerProd;
#ifdef HYPRE_NREL_CUDA
   cogmres_functions->MassInnerProd     = MassInnerProd;	
   cogmres_functions->MassInnerProdTwoVectors     = MassInnerProdTwoVectors;	
   cogmres_functions->MassInnerProdWithScaling       = MassInnerProdWithScaling;
   cogmres_functions->DoubleInnerProd       = DoubleInnerProd;
#else
   cogmres_functions->MassInnerProd     = MassInnerProd;
   cogmres_functions->MassDotpTwo       = MassDotpTwo;
#endif
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
#ifndef HYPRE_NREL_CUDA
   (cogmres_data -> cgs)            = 1; /* if 2 performs reorthogonalization */
#endif
   (cogmres_data -> tol)            = 1.0e-06; /* relative residual tol */
   (cogmres_data -> cf_tol)         = 0.0;
   (cogmres_data -> a_tol)          = 0.0; /* abs. residual tol */
   (cogmres_data -> min_iter)       = 0;
   (cogmres_data -> max_iter)       = 1000;
   (cogmres_data -> rel_change)     = 0;
   (cogmres_data -> skip_real_r_check) = 0;
#ifdef HYPRE_NREL_CUDA
   (cogmres_data -> stop_crit)      = 0; /* rel. residual norm  - this is obsolete!*/
#endif
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
#ifndef HYPRE_NREL_CUDA
   (cogmres_data -> unroll)         = 0;
#endif

   return (void *) cogmres_data;
}

/*--------------------------------------------------------------------------
 * hypre_COGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESDestroy( void *cogmres_vdata )
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
#ifndef HYPRE_NREL_CUDA
   HYPRE_Int i;
#endif

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
#ifdef HYPRE_NREL_CUDA
         cudaFree(cogmres_data -> p);
#else
         for (i = 0; i < (cogmres_data -> k_dim+1); i++)
         {
            if ( (cogmres_data -> p)[i] != NULL )
            (*(cogmres_functions->DestroyVector))( (cogmres_data -> p) [i]);
         }
         hypre_TFreeF( cogmres_data->p, cogmres_functions );
#endif
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

#ifdef HYPRE_NREL_CUDA
   HYPRE_Int max_iter                                  = (cogmres_data -> max_iter);
#else
   HYPRE_Int k_dim            = (cogmres_data -> k_dim);
   HYPRE_Int max_iter         = (cogmres_data -> max_iter);
#endif
   HYPRE_Int (*precond_setup)(void*,void*,void*,void*) = (cogmres_functions->precond_setup);
   void       *precond_data   = (cogmres_data -> precond_data);
#ifndef HYPRE_NREL_CUDA
   HYPRE_Int rel_change       = (cogmres_data -> rel_change);
#endif

   (cogmres_data -> A) = A;

   /*--------------------------------------------------
    * The arguments for NewVector are important to
    * maintain consistency between the setup and
    * compute phases of matvec and the preconditioner.
    *--------------------------------------------------*/

#ifndef HYPRE_NREL_CUDA
   if ((cogmres_data -> p) == NULL)
      (cogmres_data -> p) = (void**)(*(cogmres_functions->CreateVectorArray))(k_dim+1,x);
#endif
   if ((cogmres_data -> r) == NULL)
      (cogmres_data -> r) = (*(cogmres_functions->CreateVector))(b);
#ifndef HYPRE_NREL_CUDA
   if ((cogmres_data -> w) == NULL)
      (cogmres_data -> w) = (*(cogmres_functions->CreateVector))(b);

   if (rel_change)
   {  
      if ((cogmres_data -> w_2) == NULL)
         (cogmres_data -> w_2) = (*(cogmres_functions->CreateVector))(b);
   }


   if ((cogmres_data -> matvec_data) == NULL)
      (cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);
#endif

#ifdef HYPRE_NREL_CUDA
  if ((cogmres_data -> matvec_data) == NULL)
    (cogmres_data -> matvec_data) = (*(cogmres_functions->MatvecCreate))(A, x);
#endif

   precond_setup(precond_data, A, b, x);

   /*-----------------------------------------------------
    * Allocate space for log info
    *-----------------------------------------------------*/

   if ( (cogmres_data->logging)>0 || (cogmres_data->print_level) > 0 )
   {
      if ((cogmres_data -> norms) == NULL)
         (cogmres_data -> norms) = hypre_CTAllocF(HYPRE_Real, max_iter + 1,cogmres_functions, HYPRE_MEMORY_HOST);
   }
   if ( (cogmres_data->print_level) > 0 ) 
   {
      if ((cogmres_data -> log_file_name) == NULL)
         (cogmres_data -> log_file_name) = (char*)"cogmres.out.log";
   }

   return hypre_error_flag;
}
#ifndef HYPRE_NREL_CUDA
/*--------------------------------------------------------------------------
 * hypre_COGMRESSolve
 *-------------------------------------------------------------------------*/

HYPRE_Int
hypre_COGMRESSolve(void  *cogmres_vdata,
                   void  *A,
                   void  *b,
                   void  *x)
{

   hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   HYPRE_Int     k_dim             = (cogmres_data -> k_dim);
   HYPRE_Int     unroll            = (cogmres_data -> unroll);
   HYPRE_Int     cgs               = (cogmres_data -> cgs);
   HYPRE_Int     min_iter          = (cogmres_data -> min_iter);
   HYPRE_Int     max_iter          = (cogmres_data -> max_iter);
   HYPRE_Int     rel_change        = (cogmres_data -> rel_change);
   HYPRE_Int     skip_real_r_check = (cogmres_data -> skip_real_r_check);
   HYPRE_Real    r_tol             = (cogmres_data -> tol);
   HYPRE_Real    cf_tol            = (cogmres_data -> cf_tol);
   HYPRE_Real    a_tol             = (cogmres_data -> a_tol);
   void         *matvec_data       = (cogmres_data -> matvec_data);

   void         *r                 = (cogmres_data -> r);
   void         *w                 = (cogmres_data -> w);
   /* note: w_2 is only allocated if rel_change = 1 */
   void         *w_2               = (cogmres_data -> w_2); 

   void        **p                 = (cogmres_data -> p);

   HYPRE_Int (*precond)(void*,void*,void*,void*) = (cogmres_functions -> precond);
   HYPRE_Int  *precond_data       = (HYPRE_Int*)(cogmres_data -> precond_data);

   HYPRE_Int print_level = (cogmres_data -> print_level);
   HYPRE_Int logging     = (cogmres_data -> logging);

   HYPRE_Real     *norms          = (cogmres_data -> norms);
  /* not used yet   char           *log_file_name  = (cogmres_data -> log_file_name);*/
  /*   FILE           *fp; */

   HYPRE_Int  break_value = 0;
   HYPRE_Int  i, j, k;
  /*KS: rv is the norm history */
   HYPRE_Real *rs, *hh, *uu, *c, *s, *rs_2, *rv;
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
   HYPRE_Int    itmp = 0;

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
   if (rel_change) rs_2 = hypre_CTAllocF(HYPRE_Real,k_dim+1,cogmres_functions, HYPRE_MEMORY_HOST); 

   rv = hypre_CTAllocF(HYPRE_Real, k_dim+1, cogmres_functions, HYPRE_MEMORY_HOST);
  
   hh = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);
   uu = hypre_CTAllocF(HYPRE_Real, (k_dim+1)*k_dim, cogmres_functions, HYPRE_MEMORY_HOST);

   (*(cogmres_functions->CopyVector))(b,p[0]);

   /* compute initial residual */
   (*(cogmres_functions->Matvec))(matvec_data,-1.0, A, x, 1.0, p[0]);

   b_norm = sqrt((*(cogmres_functions->InnerProd))(b,b));
   real_r_norm_old = b_norm;

   /* Since it is does not diminish performance, attempt to return an error flag
      and notify users when they supply bad input. */
   if (b_norm != 0.) ieee_check = b_norm/b_norm; /* INF -> NaN conversion */
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
         hypre_printf("User probably placed non-numerics in supplied b.\n");
         hypre_printf("Returning error flag += 101.  Program not terminated.\n");
         hypre_printf("ERROR detected by Hypre ... END\n\n\n");
      }
      hypre_error(HYPRE_ERROR_GENERIC);
      return hypre_error_flag;
   }

   r_norm   = sqrt((*(cogmres_functions->InnerProd))(p[0],p[0]));
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
      den_norm = b_norm;
   }
   else
   {
      /* convergence criterion |r_i|/|r0| <= accuracy if |b| = 0 */
      den_norm = r_norm;
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
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate  rel.res.norm\n");
         hypre_printf("-----    ------------    ---------- ------------\n");

      }
      else
      {
         hypre_printf("=============================================\n\n");
         hypre_printf("Iters     resid.norm     conv.rate\n");
         hypre_printf("-----    ------------    ----------\n");
      };
   }


   /* once the rel. change check has passed, we do not want to check it again */
   rel_change_passed = 0;

   while (iter < max_iter)
   {
      /* initialize first term of hessenberg system */
      rs[0] = r_norm;
      if (r_norm == 0.0)
      {
         hypre_TFreeF(c,cogmres_functions); 
         hypre_TFreeF(s,cogmres_functions); 
         hypre_TFreeF(rs,cogmres_functions);
         hypre_TFreeF(rv,cogmres_functions);
         if (rel_change)  hypre_TFreeF(rs_2,cogmres_functions);
         hypre_TFreeF(hh,cogmres_functions); 
         hypre_TFreeF(uu,cogmres_functions); 
         return hypre_error_flag;
      }

      /* see if we are already converged and 
         should print the final norm and exit */

      if (r_norm  <= epsilon && iter >= min_iter) 
      {
         if (!rel_change) /* shouldn't exit after no iterations if
                           * relative change is on*/
         {
            (*(cogmres_functions->CopyVector))(b,r);
            (*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
            r_norm = sqrt((*(cogmres_functions->InnerProd))(r,r));
            if (r_norm  <= epsilon)
            {
               if ( print_level>1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               break;
            }
            else if ( print_level>0 && my_id == 0)
               hypre_printf("false convergence 1\n");
         }
      }



      t = 1.0 / r_norm;
      (*(cogmres_functions->ScaleVector))(t,p[0]);
      i = 0;
      /***RESTART CYCLE (right-preconditioning) ***/
      while (i < k_dim && iter < max_iter)
      {
         i++;
         iter++;
         itmp = (i-1)*(k_dim+1);

         (*(cogmres_functions->ClearVector))(r);
        
         precond(precond_data, A, p[i-1], r);
         (*(cogmres_functions->Matvec))(matvec_data, 1.0, A, r, 0.0, p[i]);
         for (j=0; j<i; j++)
            rv[j]  = 0;

         if (cgs > 1)
         {
            (*(cogmres_functions->MassDotpTwo))((void *) p[i], p[i-1], p, i, unroll, &hh[itmp], &uu[itmp]);
            for (j=0; j<i-1; j++) uu[j*(k_dim+1)+i-1] = uu[itmp+j];
            for (j=0; j<i; j++) rv[j] = hh[itmp+j];
            for (k=0; k < i; k++)
            {
               for (j=0; j < i; j++)
               {
                  hh[itmp+j] -= (uu[k*(k_dim+1)+j]*rv[j]);
               }
            }
            for (j=0; j<i; j++)
               hh[itmp+j]  = -rv[j]-hh[itmp+j];
         }
         else
         {
            (*(cogmres_functions->MassInnerProd))((void *) p[i], p, i, unroll, &hh[itmp]);
            for (j=0; j<i; j++)
               hh[itmp+j]  = -hh[itmp+j];
         }

         (*(cogmres_functions->MassAxpy))(&hh[itmp],p,p[i], i, unroll);
         for (j=0; j<i; j++)
            hh[itmp+j]  = -hh[itmp+j];
         t = sqrt( (*(cogmres_functions->InnerProd))(p[i],p[i]) );
         hh[itmp+i] = t;

         if (hh[itmp+i] != 0.0)
         {
            t = 1.0/t;
            (*(cogmres_functions->ScaleVector))(t,p[i]);
         }
         for (j = 1; j < i; j++)
         {
            t = hh[itmp+j-1];
            hh[itmp+j-1] = s[j-1]*hh[itmp+j] + c[j-1]*t;
            hh[itmp+j] = -s[j-1]*t + c[j-1]*hh[itmp+j];
         }
         t= hh[itmp+i]*hh[itmp+i];
         t+= hh[itmp+i-1]*hh[itmp+i-1];
         gamma = sqrt(t);
         if (gamma == 0.0) gamma = epsmac;
         c[i-1] = hh[itmp+i-1]/gamma;
         s[i-1] = hh[itmp+i]/gamma;
         rs[i] = -hh[itmp+i]*rs[i-1];
         rs[i] /=  gamma;
         rs[i-1] = c[i-1]*rs[i-1];
         // determine residual norm 
         hh[itmp+i-1] = s[i-1]*hh[itmp+i] + c[i-1]*hh[itmp+i-1];
         r_norm = fabs(rs[i]);
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
               break_value = 1;
               break;
            }
         }
         /* should we exit the restart cycle? (conv. check) */
         if (r_norm <= epsilon && iter >= min_iter)
         {
            if (rel_change && !rel_change_passed)
            {
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
               rs_2[i-1] = rs_2[i-1]/hh[itmp+i-1];
               for (k = i-2; k >= 0; k--)
               {
                  t = 0.0;
                  for (j = k+1; j < i; j++)
                  {
                     t -= hh[j*(k_dim+1)+k]*rs_2[j];
                  }
                  t+= rs_2[k];
                  rs_2[k] = t/hh[k*(k_dim+1)+k];
               }
               (*(cogmres_functions->CopyVector))(p[i-1],w);
               (*(cogmres_functions->ScaleVector))(rs_2[i-1],w);
               for (j = i-2; j >=0; j--)
                  (*(cogmres_functions->Axpy))(rs_2[j], p[j], w);

               (*(cogmres_functions->ClearVector))(r);
               /* find correction (in r) */
               precond(precond_data, A, w, r);
               /* copy current solution (x) to w (don't want to over-write x)*/
               (*(cogmres_functions->CopyVector))(x,w);

               /* add the correction */
               (*(cogmres_functions->Axpy))(1.0,r,w);

               /* now w is the approx solution  - get the norm*/
               x_norm = sqrt( (*(cogmres_functions->InnerProd))(w,w) );

               if ( !(x_norm <= guard_zero_residual ))
                  /* don't divide by zero */
               {  /* now get  x_i - x_i-1 */
                  if (num_rel_change_check)
                  {
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
                     /* first time to check rel change*/
                     /* first save current soln w in w_2 for next time */
                     (*(cogmres_functions->CopyVector))(w, w_2);

                     (*(cogmres_functions->ClearVector))(w);
                     (*(cogmres_functions->Axpy))(rs_2[i-1], p[i-1], w);
                     (*(cogmres_functions->ClearVector))(r);
                     /* apply the preconditioner */
                     precond(precond_data, A, w, r);
                     /* now r contains x_i - x_i-1 */          
                  }
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
                  relative_error = w_norm/x_norm;
                  if (relative_error <= r_tol)
                  {
                     rel_change_passed = 1;
                     break;
                  }
               }
               else
               {
                  rel_change_passed = 1;
                  break;
               }
               num_rel_change_check++;
            }
            else /* no relative change */
            {
               break;
            }
         }
      } /*** end of restart cycle ***/

      /* now compute solution, first solve upper triangular system */
      if (break_value) break;
     
      rs[i-1] = rs[i-1]/hh[itmp+i-1];
      for (k = i-2; k >= 0; k--)
      {
         t = 0.0;
         for (j = k+1; j < i; j++)
         {
            t -= hh[j*(k_dim+1)+k]*rs[j];
         }
         t+= rs[k];
         rs[k] = t/hh[k*(k_dim+1)+k];
      }

      (*(cogmres_functions->CopyVector))(p[i-1],w);
      (*(cogmres_functions->ScaleVector))(rs[i-1],w);
      for (j = i-2; j >=0; j--)
         (*(cogmres_functions->Axpy))(rs[j], p[j], w);

      (*(cogmres_functions->ClearVector))(r);
      /* find correction (in r) */
      precond(precond_data, A, w, r);

      /* update current solution x (in x) */
      (*(cogmres_functions->Axpy))(1.0,r,x);


      /* check for convergence by evaluating the actual residual */
      if (r_norm  <= epsilon && iter >= min_iter)
      {
         if (skip_real_r_check)
         {
            (cogmres_data -> converged) = 1;
            break;
         }

         /* calculate actual residual norm*/
         (*(cogmres_functions->CopyVector))(b,r);
         (*(cogmres_functions->Matvec))(matvec_data,-1.0,A,x,1.0,r);
         real_r_norm_new = r_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );

         if (r_norm <= epsilon)
         {
            if (rel_change && !rel_change_passed) /* calculate the relative change */
            {
               /* calculate the norm of the solution */
               x_norm = sqrt( (*(cogmres_functions->InnerProd))(x,x) );

               if ( !(x_norm <= guard_zero_residual ))
               /* don't divide by zero */
               {
                  (*(cogmres_functions->ClearVector))(w);
                  (*(cogmres_functions->Axpy))(rs[i-1], p[i-1], w);
                  (*(cogmres_functions->ClearVector))(r);
                  /* apply the preconditioner */
                  precond(precond_data, A, w, r);
                  /* find the norm of x_i - x_i-1 */          
                  w_norm = sqrt( (*(cogmres_functions->InnerProd))(r,r) );
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
                  (cogmres_data -> converged) = 1;
                  if ( print_level>1 && my_id == 0 )
                  {
                     hypre_printf("\n\n");
                     hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
                  }
                  break;
               }
            }
            else /* don't need to check rel. change */
            {
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
               if (print_level > 1 && my_id == 0)
               {
                  hypre_printf("\n\n");
                  hypre_printf("Final L2 norm of residual: %e\n\n", r_norm);
               }
               (cogmres_data -> converged) = 1;
               break;
            }
            /* report discrepancy between real/COGMRES residuals and restart */
            if ( print_level>0 && my_id == 0)
               hypre_printf("false convergence 2, L2 norm of residual: %e\n", r_norm);
            (*(cogmres_functions->CopyVector))(r,p[0]);
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

      if (i) (*(cogmres_functions->Axpy))(rs[i]-1.0,p[i],p[i]);
      for (j=i-1 ; j > 0; j--)
         (*(cogmres_functions->Axpy))(rs[j],p[j],p[i]);

      if (i)
      {
         (*(cogmres_functions->Axpy))(rs[0]-1.0,p[0],p[0]);
         (*(cogmres_functions->Axpy))(1.0,p[i],p[0]);
      }

   } /* END of iteration while loop */


   (cogmres_data -> num_iterations) = iter;
   if (b_norm > 0.0)
      (cogmres_data -> rel_residual_norm) = r_norm/b_norm;
   if (b_norm == 0.0)
      (cogmres_data -> rel_residual_norm) = r_norm;

   if (iter >= max_iter && r_norm > epsilon && epsilon > 0) hypre_error(HYPRE_ERROR_CONV);

   hypre_TFreeF(c,cogmres_functions); 
   hypre_TFreeF(s,cogmres_functions); 
   hypre_TFreeF(rs,cogmres_functions);
   hypre_TFreeF(rv,cogmres_functions);
   if (rel_change)  hypre_TFreeF(rs_2,cogmres_functions);

   /*for (i=0; i < k_dim+1; i++)
   {  
      hypre_TFreeF(hh[i],cogmres_functions);
      hypre_TFreeF(uu[i],cogmres_functions);
   }*/
   hypre_TFreeF(hh,cogmres_functions); 
   hypre_TFreeF(uu,cogmres_functions);

   return hypre_error_flag;
}
#endif

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


HYPRE_Int 
hypre_COGMRESSetModifyPC(void *cogmres_vdata, 
      HYPRE_Int (*modify_pc)(void *precond_data, HYPRE_Int iteration, HYPRE_Real rel_residual_norm))
{
   hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
   hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
   (cogmres_functions -> modify_pc)        = modify_pc;
   return hypre_error_flag;
} 

#ifdef HYPRE_NREL_CUDA

#define solverTimers 1
#define usePrecond 1
#define leftPrecond 0

/*-----------------------------------------------------
 * Aux function for Hessenberg matrix storage
 *-----------------------------------------------------*/

HYPRE_Int idx(HYPRE_Int r, HYPRE_Int c, HYPRE_Int n){
  //n is the # el IN THE COLUMN
  //
  //#define IDX2C(i,j,ld) (((i)*(ld))+(j))
  return r*n+c;
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

HYPRE_Int
hypre_COGMRESSetGSoption( void *cogmres_vdata,
    HYPRE_Int   level)
{
  hypre_COGMRESData *cogmres_data = (hypre_COGMRESData *)cogmres_vdata;
  (cogmres_data -> GSoption) = level;
  return hypre_error_flag;
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
      (*(cf->Axpy))((-1.0)*Hcolumn[idx( i-1,j, k_dim+1)], Vspace, j, Vspace, i);		
    }

    t = sqrt((*(cf->InnerProd))(Vspace,i,Vspace, i));
    Hcolumn[idx( i-1,i, k_dim+1)] = t;
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
(*(cf->ScaleVector))(t,Vspace, i-1);

}
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
    for (int j=0; j<i; ++j)
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

HYPRE_Int
hypre_COGMRESSolve(void  *cogmres_vdata,
                   void  *A,
                   void  *b,
                   void  *x)
{
  HYPRE_Real time1, time2, time3, time4;
  HYPRE_Real gsTime = 0.0, matvecPreconTime = 0.0, linSolveTime= 0.0, remainingTime = 0.0; 
  HYPRE_Real massAxpyTime = 0.0; 
  HYPRE_Real gsOtherTime  = 0.0f;
  HYPRE_Real massIPTime   = 0.0f, preconTime = 0.0f, mvTime = 0.0f;    
  HYPRE_Real initTime     = 0.0f;
  //if (solverTimers)
  //  time1                                     = MPI_Wtime(); 

  hypre_COGMRESData      *cogmres_data      = (hypre_COGMRESData *)cogmres_vdata;
  hypre_COGMRESFunctions *cogmres_functions = cogmres_data->functions;
  HYPRE_Int               k_dim             = (cogmres_data -> k_dim);
  HYPRE_Int               max_iter          = (cogmres_data -> max_iter);
  HYPRE_Real              r_tol             = (cogmres_data -> tol);
  HYPRE_Real              a_tol             = (cogmres_data -> a_tol);
  void                   *matvec_data       = (cogmres_data -> matvec_data);
  //hypre_ParVector * w, *p, *r ;
  void         *w                 = (cogmres_data -> w);
  void         *w_2               = (cogmres_data -> w_2); 
  void        *p                 = (cogmres_data -> p);

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

  if ( print_level>1 && my_id == 0 )
  {
    hypre_printf("=============================================\n\n");
    hypre_printf("Iters     resid.norm     conv.rate  \n");
    hypre_printf("-----    ------------    ---------- \n");
  }

  //if (solverTimers){
  //  time2 = MPI_Wtime();
  //  initTime += (time2-time1);
  //}
  //outer loop 

  iter = 0;

  while (iter < max_iter)
  {
    if (iter == 0){
      //if (solverTimers){
      //  time1 = MPI_Wtime();
      //}
      b_norm_original =  sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0));
      b_norm = b_norm_original;

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime += (time2-time1);
      //}
      if ((usePrecond) && (leftPrecond)){
	//if (solverTimers){
	//  time1 = MPI_Wtime();
	//}
	(*(cogmres_functions->UpdateVectorCPU))(b);
	(*(cogmres_functions->ClearVector))(w_2);
	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime += (time2-time1); 
	//  time1 = MPI_Wtime();
	//}
	precond(precond_data, A, b,w_2 );

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  matvecPreconTime+=(time2-time1);
	//  preconTime += (time2-time1);
	//  time1 = MPI_Wtime();
	//}
	(*(cogmres_functions->UpdateVectorCPU))(w_2);
	(*(cogmres_functions->CopyVector))(w_2, 0, b, 0);
	(*(cogmres_functions->UpdateVectorCPU))(b);
	b_norm =  sqrt((*(cogmres_functions->InnerProd))(b,0,b, 0));
	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime += (time2-time1);
	//}
      }
    }

    /*******************************************************************
     * RESTART, PRECON IS RIGHT
     * *****************************************************************/    
    if ((usePrecond)&&(!leftPrecond)){
      //if (solverTimers)
      //  time1 = MPI_Wtime();

      (*(cogmres_functions->ClearVector))(w);
      //KS: if iter == 0, x has the right CPU data, no need to copy	
      //not true if restarting
      if(iter!=0){
	(*(cogmres_functions->UpdateVectorCPU))(x);
      }	
      //w = Mx
      (*(cogmres_functions->CopyVector))(x, 0, w_2, 0);
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime += (time2-time1);
      //  time3 = MPI_Wtime();
      //}

      PUSH_RANGE("cogmres precon", 0);
      precond(precond_data, A, w_2,w );
      POP_RANGE;

      //if (solverTimers){
      //  time4 = MPI_Wtime();
      //  preconTime +=(time4-time3);
      //  matvecPreconTime += (time4-time3);

      //  time1 = MPI_Wtime();
      //}
      //w_2 = AMx = Aw   

      (*(cogmres_functions->Matvec))(matvec_data,
	  one,
	  A,
	  w,
	  0,
	  zero,
	  w_2, 0);

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  mvTime +=(time2-time1);
      //  matvecPreconTime += (time2-time1);
      //  time1 = MPI_Wtime();      
      //}
      //use Hegedus, if indicated AND first cycle

      HYPRE_Real part2 = (*(cogmres_functions->InnerProd))(w_2,0,w_2, 0);
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime +=(time2-time1);
      //}
      if (part2 == 0.0f){
	//safety check - cant divide by 0
	HegedusTrick = 0;
      }      
      if ((HegedusTrick)&&(iter==0)){
	//if (solverTimers){
	//  time1 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//}
	HYPRE_Real part1 = (*(cogmres_functions->InnerProd))(w_2,0,b, 0);

	(*(cogmres_functions->ScaleVector))(part1/part2,x, 0);
	(*(cogmres_functions->UpdateVectorCPU))(x);
	//w = Mx_0
	(*(cogmres_functions->ClearVector))(w);

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//  time1 = MPI_Wtime();     
	//}
	precond(precond_data, A, x,w );

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  preconTime +=(time2-time1);
	//  matvecPreconTime += (time2-time1);
	//  time1 = MPI_Wtime();      
	//}
	(*(cogmres_functions->CopyVector))(b, 0, p, 0);
	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//  time1 = MPI_Wtime();     
	//}
	(*(cogmres_functions->Matvec))(matvec_data,
	    minusone,
	    A,
	    w,
	    0,
	    one,
	    p, 0);
	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  mvTime +=(time2-time1);
	//  matvecPreconTime += (time2-time1);
	//}
      }//if Hegedus
      else {
	//not using Hegedus, compute the right residual
	//if (solverTimers){
	//  time1 = MPI_Wtime();
	//}
	(*(cogmres_functions->CopyVector))(b, 0, p, 0);
	(*(cogmres_functions->Axpy))(-1.0f, w_2, 0, p, 0);
	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//}
      }    
    }// if RIGHT Precond

    /*******************************************************************
     * RESTART, PRECON IS LEFT
     * *****************************************************************/    
    if ((usePrecond)&&(leftPrecond)){
      //if (solverTimers){
      //  time1 = MPI_Wtime();
      //}
      (*(cogmres_functions->Matvec))(matvec_data,
	  one,
	  A,
	  x,
	  0,
	  zero,
	  w, 0);

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  mvTime +=(time2-time1);
      //  matvecPreconTime += (time2-time1);
      //  time1 = MPI_Wtime();	
      //}

      (*(cogmres_functions->UpdateVectorCPU))(w);
      (*(cogmres_functions->ClearVector))(w_2);

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime +=(time2-time1);
      //  time1 = MPI_Wtime();
      //}
      precond(precond_data, A, w,w_2 );

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  preconTime +=(time2-time1);
      //  matvecPreconTime += (time2-time1);
      //  time1 = MPI_Wtime();
      //}
      HYPRE_Real part2 = (*(cogmres_functions->InnerProd))(w_2,0,w_2, 0);
      if (part2 == 0.0f){HegedusTrick = 0;}     
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime +=(time2-time1);
      //}
      if ((HegedusTrick)&&(iter==0)){
	//(Mb)'*(MAx)/\|MAx\|^2 = b'w_2/[(w_2)'*(w_2)] <-- scaling factor

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//}
	HYPRE_Real part1 = (*(cogmres_functions->InnerProd))(b,0,w_2, 0);

	(*(cogmres_functions->ScaleVector))(part1/part2,x, 0);
	(*(cogmres_functions->UpdateVectorCPU))(x);
	//update w_2 = MAx_0

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//  time1 = MPI_Wtime();
	//}
	(*(cogmres_functions->Matvec))(matvec_data,
	    one,
	    A,
	    x,
	    0,
	    zero,
	    w, 0);

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  mvTime +=(time2-time1);
	//  matvecPreconTime += (time2-time1);
	//  time1 = MPI_Wtime();	
	//}

	(*(cogmres_functions->UpdateVectorCPU))(w);
	(*(cogmres_functions->ClearVector))(w_2);

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime +=(time2-time1);
	//  time1 = MPI_Wtime();
	//}
	precond(precond_data, A, w,w_2 );

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  preconTime +=(time2-time1);
	//  matvecPreconTime += (time2-time1);
	//}

      }//Hegedus

      //if (solverTimers){
      //  time1 = MPI_Wtime();
      //}
      (*(cogmres_functions->CopyVector))(b, 0, p, 0);
      (*(cogmres_functions->Axpy))(-1.0f, w_2, 0, p, 0);		

      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime +=(time2-time1);
      //}
    }//LEFT precond

    /*******************************************************************
     * RESTART, PRECON IS NONE
     * *****************************************************************/    
    if (!usePrecond){
      //if (solverTimers){
      //  time1 = MPI_Wtime();
      //}
      (*(cogmres_functions->CopyVector))(b, 0, p, 0);
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  remainingTime +=(time2-time1);
      //  time1 = MPI_Wtime();
      //}

      (*(cogmres_functions->Matvec))(matvec_data,
	  minusone,
	  A,
	  x,
	  0,
	  one,
	  p, 0);
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  mvTime +=(time2-time1);
      //  matvecPreconTime += (time2-time1);
      //}
    }//no precon

    //if (solverTimers){
    //  time1 = MPI_Wtime();
    //}
    r_norm = sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0));
    if (iter == 0){      
      epsilon = r_tol*b_norm;
      //hypre_max(a_tol,r_tol*r_norm);
    }   
    if ( logging>0 || print_level > 0)
    {
      norms[iter] = r_norm;
      if ( print_level>1 && my_id == 0  ){
	if (iter == 0)
	  hypre_printf("Orthogonalization variant: %d \n", GSoption);
	hypre_printf("L2 norm of b: %16.16f\n", b_norm);
	hypre_printf("Initial L2 norm of (current) residual: %16.16f\n", r_norm);

      }
    }

    // conv criteria 
    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  remainingTime += (time2-time1);
    //}

    if (r_norm <epsilon){
      (cogmres_data -> converged) = 1;
      break;
    }//conv check
    //iter++;

    //if (solverTimers)
    //  time1 = MPI_Wtime();

    t = 1.0f/r_norm;

    (*(cogmres_functions->ScaleVector))(t,p, 0);

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

    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  remainingTime += (time2-time1);
    //}

    //inner loop 
    i = -1; 
    while (i+1 < k_dim && iter < max_iter)
    {
      i++;
      iter++;
      if ((usePrecond) && !(leftPrecond)){
	//x = M * p[i-1]
	//p[i] = A*x

	//if (solverTimers)
	//  time1 = MPI_Wtime();
	(*(cogmres_functions->CopyVector))(p, i, w_2, 0);
	//clear vector is absolutely necessary
	(*(cogmres_functions->ClearVector))(w);

	//if (solverTimers){
	//  time2 = MPI_Wtime();
	//  remainingTime += (time2-time1);
	//  time3 = MPI_Wtime();
	//}

	PUSH_RANGE("cogmres precon", 1);
	precond(precond_data, A, w_2, w);
	POP_RANGE;
	//if (solverTimers){
	//  time4 = MPI_Wtime();

	//  preconTime += (time4-time3);
	//  matvecPreconTime += (time4-time3);
	//  time1 = MPI_Wtime();
	//}

	(*(cogmres_functions->Matvec))(matvec_data,
	    one,
	    A,
	    w,
	    0,
	    zero,
	    p, i+1);
	//if (solverTimers){

	//  time2 = MPI_Wtime();
	//  mvTime += (time2-time1);
	//  matvecPreconTime += (time2-time1);   
	//}
      }
      else{
	if ((usePrecond) && (leftPrecond)){

	  //if (solverTimers)
	  //  time1 = MPI_Wtime();

	  (*(cogmres_functions->ClearVector))(w);

	  (*(cogmres_functions->CopyVector))(p, i, w_2, 0);
	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  remainingTime += (time2-time1);
	  //  time1 = MPI_Wtime();
	  //}
	  (*(cogmres_functions->Matvec))(matvec_data,
	      one,
	      A,
	      w_2,
	      0,
	      zero,
	      w, 0);

	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  mvTime += (time2-time1);
	  //  matvecPreconTime += (time2-time1);   
	  //  time1 = MPI_Wtime();
	  //}

	  //if (solverTimers)
	  //  time1 = MPI_Wtime();
	  (*(cogmres_functions->UpdateVectorCPU))(w);

	  (*(cogmres_functions->ClearVector))(w_2);

	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  remainingTime += (time2-time1);
	  //  time1 = MPI_Wtime();
	  //}
	  precond(precond_data, A, w, w_2);

	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  preconTime += (time2-time1);
	  //  matvecPreconTime += (time2-time1);   
	  //  time1 = MPI_Wtime();
	  //}

	  (*(cogmres_functions->CopyVector))(w_2, 0, p, i+1);
	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  remainingTime += (time2-time1);
	  //}
	}
	else{
	  // not using preconddd
	  //if (solverTimers)
	  //  time1 = MPI_Wtime();

	  (*(cogmres_functions->CopyVector))(p, i, w, 0);
	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  remainingTime += (time2-time1);
	  //  time1 = MPI_Wtime();
	  //}

	  (*(cogmres_functions->Matvec))(matvec_data,
	      one,
	      A,
	      w,
	      0,
	      zero,
	      p, i+1);
	  //if (solverTimers){
	  //  time2 = MPI_Wtime();
	  //  mvTime += (time2-time1);
	  //  matvecPreconTime += (time2-time1);   
	  //}

	}
      }
      //if (solverTimers){
      //  time1=MPI_Wtime();
      //}

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

      if ((GSoption >=3)){
	GramSchmidt (4, 
	    i+1, 
	    k_dim,
	    p, 
	    hh, 
	    tempH,  
	    rv, 
	    tempRV, L, cogmres_functions );
	if(i==0) {doNotSolve=1;iter--;}
	else doNotSolve = 0;
      }

      // CALL IT HERE
      //if (solverTimers){
      //  time2 = MPI_Wtime();
      //  //gsOtherTime +=  time2-time3;
      //  gsTime += (time2-time1);
      //  time1 = MPI_Wtime();
      //}
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
	//if (solverTimers){
	//  time4 = MPI_Wtime();
	//  linSolveTime += (time4-time1);
	//}

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
	if (r_norm <epsilon){
	  (cogmres_data -> converged) = 1;
	  break;
	}//conv check

	if (GSoption >3) i++;
      }//doNotSolve
    }//while (inner)

    //if (solverTimers){
    //  time1 = MPI_Wtime();
    //}
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

    for (j = 0; j <=i; j++){
      (*(cogmres_functions->Axpy))(rs[j], p, j, x, 0);		
    }	
    (*(cogmres_functions->UpdateVectorCPU))(x);

    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  remainingTime += (time2-time1);
    //}
    //test solution 
    if (r_norm < epsilon){
      (cogmres_data -> converged) = 1;
      break;
    }

    // final tolerance 
    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  remainingTime += (time2-time1);
    //}
  }
  if ((usePrecond)&&(!leftPrecond)){
    //if (solverTimers){
    //  time1 = MPI_Wtime();
    //}
    (*(cogmres_functions->CopyVector))(x, 0, w_2, 0);
    (*(cogmres_functions->ClearVector))(w);
    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  remainingTime += (time2-time1);
    //  time1 = MPI_Wtime();
    //}

    PUSH_RANGE("cogmres precon", 2);
    precond(precond_data, A, w_2, w);
    POP_RANGE;
    //if (solverTimers){
    //  time2 = MPI_Wtime();
    //  matvecPreconTime+=(time2-time1);
    //  preconTime += (time2-time1);
    //}
    //debug code

    (*(cogmres_functions->CopyVector))(w, 0, x, 0);
    (*(cogmres_functions->CopyVector))(b, 0, p, 0);
    (*(cogmres_functions->Matvec))(matvec_data,
	minusone,
	A,
	x,
	0,
	one,
	p, 0);
    printf("END: norm of residual: %16.16f \n",sqrt((*(cogmres_functions->InnerProd))(p,0,p, 0)));
    //end of debug code
  }//done only for right precond

  if (GSoption != 0){
    cudaFree(tempH);

    cudaFree(tempRV);
  }
/*
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
*/
  if ((HegedusTrick == 0))
    HegedusTrick=1;

  return 0;
}//Solve


#endif
