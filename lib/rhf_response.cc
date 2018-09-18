/*
    QC-DMET: a python implementation of density matrix embedding theory for ab initio quantum chemistry
    Copyright (C) 2015 Sebastian Wouters

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <cstdlib>
#include <mathimf.h>
#include <iostream>
#include <fstream>

extern "C" {
  
  void dgemm_(char * transA, char * transB, const int * m, const int * n,
	      const int * k, double * alpha, double * A, const int * lda,
	      double * B, const int * ldb, double * beta, double * C, const int * ldc);
  void dsyev_(char * jobz, char * uplo, const int * n, double * A,
	      const int * lda, double * W, double * work, int * lwork, int * info);
  void dcopy_(const int * n, double * x, int * incx, double * y, int * incy);
  
}

extern "C"{
  void rhf_response(const int Norb, const int Nterms, const int numPairs,
		    int * H1start, int * H1row, int * H1col, double * H0, double * rdm_deriv)
  {
    const int size = Norb * Norb;
    const int nVir = Norb - numPairs;
    
    double * eigvecs = (double *) malloc(sizeof(double)*size);
    double * eigvals = (double *) malloc(sizeof(double)*Norb);
    double * temp    = (double *) malloc(sizeof(double)*nVir*numPairs);
    
    // eigvecs and eigvals contain the eigenvectors and eigenvalues of H0
    {
      int inc = 1;
      dcopy_( &size, H0, &inc, eigvecs, &inc );
      char jobz = 'V';
      char uplo = 'U';
      int info;
      int lwork = 3*Norb-1;
      double * work = (double *) malloc(sizeof(double)*lwork);
      dsyev_( &jobz, &uplo, &Norb, eigvecs, &Norb, eigvals, work, &lwork, &info );
      free(work);
    }
    
    double * occ  = eigvecs;
    double * virt = eigvecs + numPairs * Norb;
    
    // H0 contains the 1-RDM of the RHF calculation: H0 = 2 * OCC * OCC.T
    {
      char tran = 'T';
      char notr = 'N';
      double alpha = 2.0;
      double beta  = 0.0;
      dgemm_( &notr, &tran, &Norb, &Norb, &numPairs, &alpha, occ, &Norb, occ, &Norb, &beta, H0, &Norb );
    }
    
    // temp[ vir + nVir * occ ] = - 1 / ( eps_vir - eps_occ )
    for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
      for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
	temp[ orb_vir + nVir * orb_occ ] = - 1.0 / ( eigvals[ numPairs + orb_vir ] - eigvals[ orb_occ ] );
	//if(abs( eigvals[ numPairs + orb_vir ] - eigvals[ orb_occ ] ) < 1E-6){
	//  printf("warning: gap is small\n");
	//}
      }
    }
    
# pragma omp parallel
    {
      double * work1 = (double *) malloc(sizeof(double)*size);
      double * work2 = (double *) malloc(sizeof(double)*Norb*numPairs);
      
# pragma omp for schedule(static)
      for ( int deriv = 0; deriv < Nterms; deriv++ ){
	
	// work1 = - VIRT.T * H1 * OCC / ( eps_vir - eps_occ )
	for ( int orb_vir = 0; orb_vir < nVir; orb_vir++ ){
	  for ( int orb_occ = 0; orb_occ < numPairs; orb_occ++ ){
	    double value = 0.0;
	    for ( int elem = H1start[ deriv ]; elem < H1start[ deriv + 1 ]; elem++ ){
	      value += virt[ H1row[ elem ] + Norb * orb_vir ] * occ[ H1col[ elem ] + Norb * orb_occ ];
	    }
	    work1[ orb_vir + nVir * orb_occ ] = value * temp[ orb_vir + nVir * orb_occ ];
	  }
	}
	
	// work1 = 2 * VIRT * work1 * OCC.T
	{
	  char notr = 'N';
	  double alpha = 2.0;
	  double beta = 0.0;
	  dgemm_( &notr, &notr, &Norb, &numPairs, &nVir, &alpha, virt,
		  &Norb, work1, &nVir, &beta, work2, &Norb ); // work2 = 2 * VIRT * work1
	  alpha = 1.0;
	  char tran = 'T';
	  dgemm_( &notr, &tran, &Norb, &Norb, &numPairs, &alpha, work2,
		  &Norb, occ, &Norb, &beta, work1, &Norb ); // work1 = work2 * OCC.T
	}
	
	// rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] = work1 + work1.T
	for ( int row = 0; row < Norb; row++ ){
	  for ( int col = 0; col < Norb; col++ ){
	    rdm_deriv[ row + Norb * ( col + Norb * deriv ) ] =
	      work1[ row + Norb * col ] + work1[ col + Norb * row ];
	  }
	}
      }
      
      free(work1);
      free(work2);
    }
    
    free(temp);
    free(eigvals);
    free(eigvecs);

  }
}

extern "C" {
  void rhf_response_T(const int Norb, const int Nterms, int * H1start,
		      int *H1row, int *H1col, double *H0, double Tempr,
		      double mu, double * rdm_deriv){
    
    const int size = Norb * Norb;
    
    double * eigvecs = (double *)malloc(sizeof(double)*size);
    double * eigvals = (double *)malloc(sizeof(double)*Norb);
    double * temp    = (double *)malloc(sizeof(double)*size);
    
    // eigvecs and eigvals contain the eigenvectors and eigenvalues of H0
    {
      int inc = 1;
      dcopy_( &size, H0, &inc, eigvecs, &inc );
      char jobz = 'V';
      char uplo = 'U';
      int info;
      int lwork = 3*Norb-1;
      double * work = (double *) malloc(sizeof(double)*lwork);
      dsyev_( &jobz, &uplo, &Norb, eigvecs, &Norb, eigvals, work, &lwork, &info );
      free(work);
    }
    
    // H0 contains the 1-RDM of the RHF calculation: H0 = 2 * Orb * 1/(1+exp(kb*T*Ei)) * Orb.T
    {
      // intermediate variable of diagonal matrix multiply orbitals
      double * temp0 = (double *)malloc(sizeof(double)*size);
      
      for ( int row = 0; row < Norb; row++){
	for ( int col = 0; col < Norb; col++){
	  temp0[row+Norb*col] = eigvecs[row+Norb*col]/(1+exp(Tempr*(eigvals[col]-mu)));
	}
      }
      
      char tran = 'T';
      char notr = 'N';
      double alpha = 2.0;
      double beta  = 0.0;
      dgemm_( &notr, &tran, &Norb, &Norb, &Norb, &alpha, eigvecs, &Norb, temp0,
      	      &Norb, &beta, H0, &Norb );
      
      free(temp0);
    }
        
    {
      // temp = (1 - np)*nq*(exp(Ep - Eq)-1)/(Ep - Eq)
      for ( int row = 0; row < Norb; row++ ){
	for ( int col = 0; col < Norb; col++){
	  double nrow = 1.0/(1.0 + exp(Tempr*(eigvals[row]-mu)));
	  double ncol = 1.0/(1.0 + exp(Tempr*(eigvals[col]-mu)));
	  
	  if ( abs ( eigvals[row] - eigvals[col] ) < 1E-10){
	    temp [ row + col*Norb ] = -1.0*(1.0-ncol)*nrow*Tempr;
	  }
	  else {
	    temp [ row + col*Norb ] = -1.0*(ncol-nrow)/(eigvals[row]-eigvals[col]);
	  }
	  // printf("%30.20e\n", temp[row+col*Norb]);
	}
      }
    }
    
# pragma omp parallel
    {
      double * work1 = (double *) malloc(sizeof(double)*size);
      double * work2 = (double *) malloc(sizeof(double)*size);
# pragma omp for schedule(static)
      for ( int deriv = 0; deriv < Nterms; deriv++){
	for ( int row = 0; row < Norb; row++){
	  for ( int col = 0; col < Norb; col++){
	    double value = 0.0;
	    for ( int elem = H1start[deriv]; elem < H1start[deriv+1]; elem++){
	      value += eigvecs[H1row[elem]+Norb*row]*eigvecs[H1col[elem]+Norb*col];
	    }
	    work1[row + Norb*col] = value*temp[row + Norb*col];
	  }
	}
	
	// work1
	{
	  char notr = 'N';
	  double alpha = 2.0;
	  double beta = 0.0;
	  dgemm_( &notr, &notr, &Norb, &Norb, &Norb, &alpha, eigvecs,
		  &Norb, work1, &Norb, &beta, work2, &Norb ); // work2 = eigvecs * work1
	  alpha = 1.0;
	  char tran = 'T';
	  dgemm_( &notr, &tran, &Norb, &Norb, &Norb, &alpha, work2,
		  &Norb, eigvecs, &Norb, &beta, work1, &Norb ); // work1 = work2 * OCC.T
	}
	
	for ( int row = 0; row < Norb; row++){
	  for ( int col = 0; col < Norb; col++){
	    rdm_deriv[row + Norb * (col+Norb*deriv)] =
	      work1[row + Norb*col];
	  }
	}
	
      }
      
      free(work1);
      free(work2);
    }
    
    free(temp);
    free(eigvals);
    free(eigvecs);
  }
  
}
