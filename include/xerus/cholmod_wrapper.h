// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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
* @brief Header file for suitesparse wrapper functions.
*/

#pragma once

#include <memory>
#include <map>
#include <mutex>
#include <functional>

#include <suitesparse/cholmod.h>
#include <suitesparse/SuiteSparseQR.hpp>


namespace xerus {
	class Tensor;
	
	namespace internal {
		
		
		///@brief wrapper object for the cholmod_common struct to automatically call the constructor and destructor
		struct CholmodCommon final {
			struct RestrictedAccess final {
				cholmod_common* const c;
				std::mutex &lock;
				RestrictedAccess(cholmod_common* const _c, std::mutex &_lock);
				operator cholmod_common*() const;
				~RestrictedAccess();
			};
			
			std::unique_ptr<cholmod_common> c;
			std::mutex lock;
			CholmodCommon();
			~CholmodCommon();
			RestrictedAccess get();
			std::function<void(cholmod_sparse*)> get_deleter();
		};
		
		///@brief wrapper class for the cholmod sparse matrix objects
		class CholmodSparse final {
		public:
			using ptr_type = std::unique_ptr<cholmod_sparse, std::function<void(cholmod_sparse*)>>;
			ptr_type matrix;
			
			///@brief stores the given cholmod_sparse object in this wrapper. NOTE: the object must have been created by the same thread that uses this constructor!
			CholmodSparse(cholmod_sparse *_matrix);
			
			///@brief Creates a cholmod sparse matrix with given dimensions and number of entries.
			CholmodSparse(const size_t _m, const size_t _n, const size_t _N);
			
			///@brief Converts the given tensor to the cholmod sparse format using the given matrification.
			///@note _input is supposed to be a _m by _n matrix _before_ transposition.
			CholmodSparse(const std::map<size_t, double>& _input, const size_t _m, const size_t _n, const bool _transpose);
			
			///@brief Transforms a cholmod sparse matrix to sparse Tensor format.
			std::map<size_t, double> to_map(double _alpha=1.0) const;
			
			///@brief Calculates the Matrix Matrix product with another sparse matrix.
			CholmodSparse operator*(const CholmodSparse &_rhs) const;
			
			///@brief transposes the represented sparse matrix.
			void transpose();
			
			///@brief Calculates the Matrix Matrix product between two sparse matrices.
			static void matrix_matrix_product(std::map<size_t, double>& _C,
										const size_t _leftDim,
										const size_t _rightDim,
										const double _alpha,
										const std::map<size_t, double>& _A,
										const bool _transposeA,
										const size_t _midDim,
										const std::map<size_t, double>& _B,
										const bool _transposeB);
			
			///@brief solve operator / for sparse right hand sites
			static void solve_sparse_rhs(std::map<size_t, double>& _x,
								size_t _xDim,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const std::map<size_t, double>& _b,
								size_t _bDim
			);
			
			///@brief solve operator / for dense right hand sites
			static void solve_dense_rhs(double * _x,
								size_t _xDim,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const double* _b,
								size_t _bDim
			);
			
			/**
			 * @brief calculates the sparse QR decomposition (or qc if fullrank = false)
			 * @note A is assumed to be a _m by _n matrix _before_ transposition. Outputs are given in sparse format but usually not very sparse anymore...
			 * @returns (q, c, rank), rank = min(m,n) if fullrank was set to true
			 */
			static std::tuple<std::map<size_t, double>, std::map<size_t, double>, size_t> qc(
				const std::map<size_t, double> &_A,
				const bool _transposeA,
				size_t _m,
				size_t _n,
				bool _fullrank
			);
			
			/**
			 * @brief calculates the sparse CQ decomposition
			 * @note A is assumed to be a _m by _n matrix _before_ transposition. Outputs are given in sparse format but usually not very sparse anymore...
			 * @returns (c, q, rank), rank = min(m,n) if fullrank was set to true
			 */
			static std::tuple<std::map<size_t, double>, std::map<size_t, double>, size_t> cq(
				const std::map<size_t, double> &_A,
				const bool _transposeA,
				size_t _m,
				size_t _n,
				bool _fullrank
			);
		};
		
		
		extern thread_local CholmodCommon cholmodObject;
	}
}
