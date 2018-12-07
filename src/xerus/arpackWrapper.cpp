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

#ifdef ARPACK_LIBRARIES

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
#include <xerus/index.h>
#include <xerus/indexedTensorMoveable.h>

#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/math.h>
#include <xerus/misc/internal.h>
#include "xerus/tensor.h"



namespace xerus {
	namespace arpackWrapper {
		
		/// Solves Ax = x*lambda for x and lambda for the _k smallest eigenvalues
		void solve_ev(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, arpack::which const _ritz_option, int _info) {
			REQUIRE(_n <= static_cast<size_t>(std::numeric_limits<int>::max()), "Dimension to large for ARPACK");
			REQUIRE(_info >= 0, "Info == 0, random; Info > 0, take residual; info is " << _info);
			REQUIRE(_k < _n, "For some reason the number of Eigenvalues must be smaller than the dimension. see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f and error code -3" );
			//REQUIRE(is_symmetric(_A, _n), "A must be symmetric");

			// iteration variables dsaupd
			int ido = 0;
			auto bmat_option = arpack::bmat::identity;
			int nev = static_cast<int>(_k);
			int ncv = nev * 3 >= static_cast<int>(_n) ? static_cast<int>(_n) : nev * 3; // TODO check for best value here
			std::unique_ptr<double[]> v(new double[_n * ncv]);
			std::unique_ptr<int[]> iparam(new int[11]);
			std::unique_ptr<int[]> ipntr(new int[11]);
			std::unique_ptr<double[]> workd(new double[3 * _n ]);

			int lworkl = ncv*(ncv + 8);
			std::unique_ptr<double[]> workl(new double[lworkl]);

			// ev extraction variables dseupd
			bool rvec = true;
			auto howmny_option = arpack::howmny::ritz_vectors;
			std::unique_ptr<int[]> select(new int[ncv]);
			double sigma = 0.0;
			int info1 = _info;

			// intialization of iparam, parameters for arpack
			iparam[0] = 1;
			iparam[2] = static_cast<int>(_maxiter);
			iparam[6] = 1;

			for (size_t i = 0; i < _maxiter; ++i){
				//LOG(info, "iter = " << i);
				arpack::saupd( 							//NOTE for more details see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f
					ido,											// Reverse Communication flag
					bmat_option,							// Standard or Generalized EV problem
					static_cast<int>(_n), 		// Dimension of the EV problem
					_ritz_option,							/* Specify which of the Ritz values of OP to compute.
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
					info1											/* If INFO .EQ. 0, a randomly initial residual vector is used.
																			 If INFO .NE. 0, RESID contains the initial residual vector,
																			 possibly from a previous run.
																			 Error flag on output. */
				);

				if (ido == -1 or ido == 1){
					blasWrapper::matrix_vector_product(workd.get() + ipntr[1] - 1,_n,1.0, _A, _n, false, workd.get() + ipntr[0] - 1);
				}
				else {
					break;
				}
			}
			REQUIRE(info1 >= 0, "ARPACK exited with error, see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f, error code is " << info1);
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
				_ritz_option,
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
				info1
			);

			return;
		}
	
		/// Solves Ax = x*lambda for x and lambda for the _k smallest eigenvalues
		void solve_ev_dmrg_special(double* const _x, const Tensor& _l, const Tensor& _A, const Tensor& _A1, const Tensor& _r, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, arpack::which const _ritz_option, int _info) {
			REQUIRE(_n <= static_cast<size_t>(std::numeric_limits<int>::max()), "Dimension to large for ARPACK");
			REQUIRE(_info >= 0, "Info == 0, random; Info > 0, take residual; info is " << _info);
			REQUIRE(_k < _n, "For some reason the number of Eigenvalues must be smaller than the dimension. see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f and error code -3" );
			//REQUIRE(is_symmetric(_A, _n), "A must be symmetric");

			// iteration variables dsaupd
			int ido = 0;
			auto bmat_option = arpack::bmat::identity;
			int nev = static_cast<int>(_k);
			int ncv = nev * 3 >= static_cast<int>(_n) ? static_cast<int>(_n) : nev * 3; // TODO check for best value here
			std::unique_ptr<double[]> v(new double[_n * ncv]);
			std::unique_ptr<int[]> iparam(new int[11]);
			std::unique_ptr<int[]> ipntr(new int[11]);
			std::unique_ptr<double[]> workd(new double[3 * _n ]);

			int lworkl = ncv*(ncv + 8);
			std::unique_ptr<double[]> workl(new double[lworkl]);

			// ev extraction variables dseupd
			bool rvec = true;
			auto howmny_option = arpack::howmny::ritz_vectors;
			std::unique_ptr<int[]> select(new int[ncv]);
			double sigma = 0.0;
			int info1 = _info;

			// intialization of iparam, parameters for arpack
			iparam[0] = 1;
			iparam[2] = static_cast<int>(_maxiter);
			iparam[6] = 1;

			for (size_t i = 0; i < _maxiter; ++i){
				//LOG(info, "iter = " << i);
				arpack::saupd( 							//NOTE for more details see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f
					ido,											// Reverse Communication flag
					bmat_option,							// Standard or Generalized EV problem
					static_cast<int>(_n), 		// Dimension of the EV problem
					_ritz_option,							/* Specify which of the Ritz values of OP to compute.
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
					info1											/* If INFO .EQ. 0, a randomly initial residual vector is used.
																			 If INFO .NE. 0, RESID contains the initial residual vector,
																			 possibly from a previous run.
																			 Error flag on output. */
				);

				if (ido == -1 or ido == 1){
					Tensor tmpX({_l.dimensions[2], _A.dimensions[2], _A1.dimensions[2], _r.dimensions[2]});
					Tensor tmpX1({_l.dimensions[0], _A.dimensions[1], _A1.dimensions[1], _r.dimensions[0]});
					auto tmpX_ptr = tmpX.override_dense_data();
					misc::copy(tmpX_ptr, workd.get() + ipntr[0] - 1, _n);
					Index i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3;
					tmpX1(i1,i2,i3,i4) = _l(i1, k1, j1)* (_A(k1, i2, j2, k2) * (_A1(k2,i3,j3,k3)* (_r(i4, k3, j4) * tmpX(j1,j2,j3,j4))));

					auto tmpX1_ptr = tmpX1.override_dense_data();
					misc::copy(workd.get() + ipntr[1] - 1,tmpX1_ptr, _n);
				}
				else {
					break;
				}
			}
			REQUIRE(info1 >= 0, "ARPACK exited with error, see https://github.com/opencollab/arpack-ng/blob/master/SRC/dsaupd.f, error code is " << info1);
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
				_ritz_option,
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
				info1
			);

			return;
		}
		
		void solve_ev_smallest(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info) {
			solve_ev (_x, _A, _ev, _k, _n, _resid, _maxiter, _eps, arpack::which::smallest_algebraic, _info);
		}

		void solve_ev_biggest(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info) {
			solve_ev (_x, _A, _ev, _k, _n, _resid, _maxiter, _eps, arpack::which::largest_algebraic, _info);
		}

		void solve_ev_smallest_dmrg_special(double* const _x, const Tensor& _l, const Tensor& _A, const Tensor& _A1, const Tensor& _r, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info) {
			solve_ev_dmrg_special (_x, _l, _A, _A1, _r, _ev, _k, _n, _resid, _maxiter, _eps, arpack::which::smallest_algebraic, _info);
		}

	} // namespace arpackWrapper

} // namespace xerus
#endif

