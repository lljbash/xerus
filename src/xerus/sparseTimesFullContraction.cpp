// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2019 Benjamin Huber and Sebastian Wolf. 
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
 * @brief Implementation of sparse matrix times dense matrix wrapper functions.
 */
#include <memory>

#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/check.h>
#include <xerus/misc/stringUtilities.h>
#include <xerus/sparseTimesFullContraction.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/internal.h>

namespace xerus {
	
	XERUS_force_inline void transpose(double* const __restrict _out, const double* const __restrict _in, const size_t _leftDim, const size_t _rightDim) {
		for(size_t i = 0; i < _leftDim; ++i) {
			for(size_t j = 0; j < _rightDim; ++j) {
				_out[j*_leftDim+i] = _in[i*_rightDim+j];
			}
		}
	}
	
	XERUS_force_inline std::unique_ptr<double[]> transpose(const double* const _A, const size_t _leftDim, const size_t _rightDim) {
		std::unique_ptr<double[]> AT(new double[_leftDim*_rightDim]);
		transpose(AT.get(), _A, _leftDim, _rightDim);
		return AT;
	}
	
	
	XERUS_force_inline void transpose(std::map<size_t, double>& __restrict _out, const std::map<size_t, double>& __restrict _in, const size_t _leftDim, const size_t _rightDim) {
		for(const auto& entry : _in) {
			const size_t i = entry.first/_rightDim;
			const size_t j = entry.first%_rightDim;
			_out.emplace(j*_leftDim + i, entry.second);
		}
	}
	
	XERUS_force_inline std::map<size_t, double> transpose(const std::map<size_t, double>& _A, const size_t _leftDim, const size_t _rightDim) {
		std::map<size_t, double> AT;
		transpose(AT, _A, _leftDim, _rightDim);
		return AT;
	}
	
	// - - - - - - - - - - - - - - - - - - - - - - - - - Mix to Full - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	
	void matrix_matrix_product( double* const _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const size_t _midDim,
								const double* const _B) {
		XERUS_PA_START;
		
		// Prepare output array
		misc::set_zero(_C, _leftDim*_rightDim);
		
		// Transposition of A only changes how i and j are calculated
		if(!_transposeA) {
			for(const auto& entry : _A) {
				const size_t i = entry.first/_midDim;
				const size_t j = entry.first%_midDim;
				misc::add_scaled(_C+i*_rightDim, _alpha*entry.second, _B+j*_rightDim, _rightDim);
			}
		} else {
			for(const auto& entry : _A) {
				const size_t i = entry.first%_leftDim;
				const size_t j = entry.first/_leftDim;
				misc::add_scaled(_C+i*_rightDim, _alpha*entry.second, _B+j*_rightDim, _rightDim);
			}
		}
		
		XERUS_PA_END("Mixed BLAS", "Matrix-Matrix-Multiplication ==> Full", misc::to_string(_leftDim)+"x"+misc::to_string(_midDim)+" * "+misc::to_string(_midDim)+"x"+misc::to_string(_rightDim));
	}
	
	void matrix_matrix_product( double* const _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const size_t _midDim,
								const double* const _B,
								const bool _transposeB) {
		if(_transposeB) {
			const std::unique_ptr<double[]> BT = transpose(_B, _rightDim, _midDim);
			matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, _A, _transposeA, _midDim, BT.get());
		} else {
			matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, _A, _transposeA, _midDim, _B);
		}
	}
	
	void matrix_matrix_product( double* const _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const double* const _A,
								const bool _transposeA,
								const size_t _midDim,
								const std::map<size_t, double>& _B,
								const bool _transposeB) {
		// It is significantly faster to calculate (B^T * A*T)^T
		const std::unique_ptr<double[]> CT(new double[_leftDim*_rightDim]);
		matrix_matrix_product(CT.get(), _rightDim, _leftDim, _alpha, _B, !_transposeB, _midDim, _A, !_transposeA);
		transpose(_C, CT.get(), _rightDim, _leftDim);
	}
	
	
	// - - - - - - - - - - - - - - - - - - - - - - - - - Mix to Sparse - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	
	void matrix_matrix_product( std::map<size_t, double>& _C,
								const size_t  /*_leftDim*/,
								const size_t _rightDim,
								const double _alpha,
								const std::map<size_t, double>& _A,
								const size_t _midDim,
								const double* const _B) {
		XERUS_PA_START;
		
		size_t currentRow = 0;
		std::unique_ptr<double[]> row(new double[_rightDim]);
		misc::set_zero(row.get(), _rightDim);
		
		for(const auto& entry : _A) {
			const size_t i = entry.first/_midDim;
			const size_t j = entry.first%_midDim;
			
			if(i == currentRow) {
				misc::add_scaled(row.get(), _alpha*entry.second, _B+j*_rightDim, _rightDim);
			} else {
				INTERNAL_CHECK(i > currentRow, "Internal Error");
				
				// Copy old row to _C
				for(size_t k = 0; k < _rightDim; ++k) {
					#pragma GCC diagnostic push
					#pragma GCC diagnostic ignored "-Wfloat-equal"
					if(row.get()[k] != 0) {
						_C.emplace(currentRow*_rightDim + k, row.get()[k]);
					}
					#pragma GCC diagnostic pop
				}
				
				// Start new row
				currentRow = i;
				misc::copy_scaled(row.get(), _alpha*entry.second, _B+j*_rightDim, _rightDim);
			}
		}
		
		// Copy the last row to _C
		for(size_t k = 0; k < _rightDim; ++k) {
			#pragma GCC diagnostic push
				#pragma GCC diagnostic ignored "-Wfloat-equal"
				if(row.get()[k] != 0) {
					_C.emplace(currentRow*_rightDim + k, row.get()[k]);
				}
			#pragma GCC diagnostic pop
		}
		
		XERUS_PA_END("Mixed BLAS", "Matrix-Matrix-Multiplication ==> Sparse", "?x"+misc::to_string(_midDim)+" * "+misc::to_string(_midDim)+"x"+misc::to_string(_rightDim));
	}
	
	void matrix_matrix_product( std::map<size_t, double>& _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const size_t _midDim,
								const double* const _B,
								const bool _transposeB) {
		if(_transposeA) {
			const std::map<size_t, double> AT = transpose(_A, _midDim, _leftDim);
			if(_transposeB) {
				std::unique_ptr<double[]> BT = transpose(_B, _rightDim, _midDim);
				matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, AT, _midDim, BT.get());
			} else {
				matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, AT, _midDim, _B);
			}
		} else {
			if(_transposeB) {
				std::unique_ptr<double[]> BT = transpose(_B, _rightDim, _midDim);
				matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, _A, _midDim, BT.get());
			} else {
				matrix_matrix_product(_C, _leftDim, _rightDim, _alpha, _A, _midDim, _B);
			}
		}
	}
	
	void matrix_matrix_product( std::map<size_t, double>& _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const double* const _A,
								const bool _transposeA,
								const size_t _midDim,
								const std::map<size_t, double>& _B,
								const bool _transposeB) {
		// It is significantly faster to calculate (B^T * A*T)^T (this is only benchmarked for mix -> Full yet...)
		std::map<size_t, double> CT;
		matrix_matrix_product(CT, _rightDim, _leftDim, _alpha, _B, !_transposeB, _midDim, _A, !_transposeA);
		transpose(_C, CT, _rightDim, _leftDim);
	}
} // namespace xerus
