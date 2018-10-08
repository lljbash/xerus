// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2018 Benjamin Huber and Sebastian Wolf. 
// 
// Xerus is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
// 
// Xerus is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
// 
// You should have received a copy of the GNU Affero General Public License
// along with Xerus. If not, see <http://www.gnu.org/licenses/>.
//
// For further information on Xerus visit https://libXerus.org 
// or contact us at contact@libXerus.org.

/**
* @file
* @brief Implementation of the blas and lapack wrapper functions.
*/


#include <complex.h>
// fix for non standard-conform complex implementation
#undef I



#ifdef __has_include
	#if __has_include("arpack.hpp")
		#include "arpack.hpp"
	#elif __has_include("arpack/arpack.hpp")
		#include "arpack/arpack.hpp"
	#else
		#pragma error no arpack found
	#endif
#else
	#include "arpack/arpack.hpp"
#endif


#include <memory>
#include <xerus/misc/standard.h>
#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/check.h>

#include <xerus/misc/stringUtilities.h>
#include <xerus/basic.h>

#include <xerus/blasLapackWrapper.h>
#include <xerus/arpackWrapper.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>



namespace xerus {
	namespace arpackWrapper {
		

		
		

		
		/// Solves Ax = x*lambda for x and lambda
		void solve_ev(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps) {
			REQUIRE(_n <= static_cast<size_t>(std::numeric_limits<int>::max()), "Dimension to large for ARPACK");


			// iteration variables dsaupd
			int ido = 0;
			auto bmat_option = arpack::bmat::identity;
			auto ritz_option = arpack::which::smallest_algebraic;
			int nev = static_cast<int>(_k);
			int ncv = nev * 3; // TODO check for best value here
			std::unique_ptr<double[]> v(new double[_n * ncv]);
			std::unique_ptr<int[]> iparam(new int[11]);
			std::unique_ptr<int[]> ipntr(new int[11]);
			std::unique_ptr<double[]> workd(new double[3 * _n ]);

			int lworkl = ncv*(ncv + 8);
			std::unique_ptr<double[]> workl(new double[lworkl]);
			int info = 0;

			// ev extraction variables dseupd
			bool rvec = true;
			auto howmny_option = arpack::howmny::ritz_vectors;
			std::unique_ptr<int[]> select(new int[ncv]);
			double sigma = 0.0;

			// intialization of iparam, parameters for arpack
			iparam[0] = 1;
			iparam[2] = static_cast<int>(_maxiter);
			iparam[6] = 1;

			for (size_t i = 0; i < _maxiter; ++i){
				XERUS_LOG(info, "Iter = " << i);
				arpack::saupd( 							//NOTE for more details see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f
					ido,											// Reverse Communication flag
					bmat_option,							// Standard or Generalized EV problem
					static_cast<int>(_n), 		// Dimension of the EV problem
					ritz_option,							/* Specify which of the Ritz values of OP to compute.
																			'LA' - compute the NEV largest (algebraic) eigenvalues.
																			'SA' - compute the NEV smallest (algebraic) eigenvalues.
																			'LM' - compute the NEV largest (in magnitude) eigenvalues.
																			'SM' - compute the NEV smallest (in magnitude) eigenvalues.
																			'BE' - compute NEV eigenvalues, half from each end of the
																						 spectrum.  When NEV is odd, compute one more from the
																						 high end than from the low end.*/
					nev,											// Number of eigenvalues of OP to be computed. 0 < NEV < N.
					static_cast<double>(_eps), // Stopping Creterion
					_resid,										// Residual, if info = 0, random initialized if info  > 0 takes the value given
					ncv,											// Number of Lanczos vectors generated
					v.get(),												// n x ncv array (output) contains the lanczos basis vectors
					static_cast<int>(_n), 		// Leading dimension of V exactly as declared in the calling program.
					iparam.get(),										// method for selecting the implicit shifts.
					ipntr.get(),          					// Pointer to mark the starting locations in the WORKD and WORKL
					workd.get(),										// Distributed array to be used in the basic Arnoldi iteration for reverse communication.
					workl.get(),										// rivate (replicated) array on each PE or array allocated on the front end.
					lworkl,          					// LWORKL must be at least NCV**2 + 8*NCV .
					info											/* If INFO .EQ. 0, a randomly initial residual vector is used.
																			 If INFO .NE. 0, RESID contains the initial residual vector,
																			 possibly from a previous run.
																			 Error flag on output. */
				);
				XERUS_LOG(info, "saupd done, ido = " << ido);

				if (ido == -1 or ido == 1){
					XERUS_LOG(info, "ipntr = " << ipntr[0]);
					XERUS_LOG(info, "ipntr = " << ipntr[1]);
					XERUS_LOG(info, "A = " << _A[0] << " " << _A[1] << " " << _A[2] << " " << _A[3] << " " << _A[4] << " " << _A[5] << " " << _A[6] << " " << _A[7] << " " << _A[8] << " " << _A[9] << " " << _A[10] << " " << _A[11] << " " << _A[12] << " " << _A[13] << " " << _A[14] << " " << _A[15] << " ");
					auto x = workd.get() + ipntr[0] - 1;
					auto y = workd.get() + ipntr[1] - 1;
					XERUS_LOG(info, "x = " << x[0] << " " << x[1] << " " << x[2] << " " << x[3] );
					XERUS_LOG(info, "y = " << y[0] << " " << y[1] << " " << y[2] << " " << y[3] );

					blasWrapper::matrix_vector_product(workd.get() + ipntr[0] - 1,_n,1.0, _A, _n, false, workd.get() + ipntr[1] - 1);
					XERUS_LOG(info, "y = " << y[0] << " " << y[1] << " " << y[2] << " " << y[3] );

				}
				else {
					break;
				}
			}
			XERUS_LOG(info, "saupd done");

			arpack::seupd( 							// NOTE for more details see https://github.com/opencollab/arpack-ng/blob/master/SRC/dseupd.f
				rvec,											// Specifies whether Ritz vectors corresponding to the Ritz value approximations to the eigenproblem A*z = lambda*B*z are computed.
				howmny_option,						/* Specifies how many Ritz vectors are wanted and the form of Z
																		 the matrix of Ritz vectors. See remark 1 below.
																		 = 'A': compute NEV Ritz vectors;
																		 = 'S': compute some of the Ritz vectors, specified
															 	 	 	 by the logical array select. */
				select.get(),										// Logical array of dimension NCV.
				_ev,											// On exit, _ev contains the Ritz value approximations to the eigenvalues of A*z = lambda*B*z.
				_x, 											/* On exit, _x contains the B-orthonormal Ritz vectors of the
																		 eigensystem A*z = lambda*B*z corresponding to the Ritz
														 	 	 	 	 value approximations. */
				static_cast<int>(_n),		  // The leading dimension of the array Z.
				sigma,										// If IPARAM(7) = 3,4,5 represents the shift. Not referenced if IPARAM(7) = 1 or 2.
				// Same as above
				bmat_option,
				static_cast<int>(_n),
				ritz_option,
				nev,
				static_cast<double>(_eps),
				_resid,
				ncv,
				v.get(),
				static_cast<int>(_n),
				iparam.get(),
				ipntr.get(),
				workd.get(),
				workl.get(),
				lworkl,
				info
			);
			XERUS_LOG(info, "seupd done");

			XERUS_LOG(info, "ev = " << _ev[0]);
			XERUS_LOG(info, "iparam = " << iparam[0]);
			XERUS_LOG(info, "ipntr = " << ipntr[0]);
			return;
		}
	

		
	} // namespace arpackWrapper

} // namespace xerus

