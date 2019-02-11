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
* @brief Header file for the arpack wrapper functions.
*/
#pragma once

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



#include "misc/standard.h"
#include <memory>


namespace xerus {
	class Tensor;
	/**
	* @brief In this namespace the minimal wrappers for the ARPACK functions are collected.
	* @details As an end user of xerus it should never be nessecary to call any of these functions, unless
	* a seriously low level implementation of a critical part of an algorithm is required.
	*/
	namespace arpackWrapper {
		///@brief: Solves Ax = lambda*x for x, this calls the Arpack Routine dsaupd
		void solve_ev(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, arpack::which const _ritz_option, int _info);
		///@brief: Solves Ax = lambda*x for x, for the smallest _k eigenvalues
		void solve_ev_smallest(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info);
		///@brief: Solves Ax = lambda*x for x, for the biggest _k eigenvalues
		void solve_ev_biggest(double* const _x, const double* const _A, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info);


		//TODO check if this can be simplified!!
		///@brief: Solves Ax = lambda*x for x, this calls the Arpack Routine dsaupd
		void solve_ev_dmrg_special(double* const _x, const Tensor& _l, const Tensor& _A, const Tensor& _A1, const Tensor& _r, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, arpack::which const _ritz_option, int _info);
		///@brief: Solves Ax = lambda*x for x, for the smallest _k eigenvalues
		void solve_ev_smallest_dmrg_special(double* const _x, const Tensor& _l, const Tensor& _A, const Tensor& _A1, const Tensor& _r, double* const _ev, const size_t _k, const size_t _n, double* const _resid, const size_t _maxiter, const double _eps, int _info);




	}
}
#endif




