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
* @brief Implementation of the suitesparse wrapper functions.
*/

#include <xerus/cholmod_wrapper.h>
#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/misc/performanceAnalysis.h>
#include <xerus/misc/basicArraySupport.h>

#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

namespace xerus { namespace internal {
	
	CholmodCommon::RestrictedAccess::RestrictedAccess(cholmod_common* const _c, std::mutex& _lock) 
		: c(_c), lock(_lock)
	{
		lock.lock();
	}
	
	CholmodCommon::RestrictedAccess::operator cholmod_common*() const {
		return c;
	}
	
	CholmodCommon::RestrictedAccess::~RestrictedAccess() {
		lock.unlock();
	}

	static void error_handler(int status, const char *file, int line, const char *message) {
		if (status < 0) {
			LOG(fatal, "CHOLMOD had a fatal error in " << file << ":" << line << " (status: " << status << ") msg: " << message);
		} else {
			LOG(cholmod_warning, "CHOLMOD warns in " << file << ":" << line << " (status: " << status << ") msg: " << message);
		}
	}
	
	CholmodCommon::CholmodCommon() : c(new cholmod_common()) {
		cholmod_l_start(c.get());
		REQUIRE(c->itype == CHOLMOD_LONG, "atm only cholmod compiled with itype = long is supported...");
		REQUIRE(c->dtype == CHOLMOD_DOUBLE, "atm only cholmod compiled with dtype = double is supported...");
		c->error_handler = &error_handler;
		c->print = 0;
		REQUIRE(c->status == 0, "unable to initialize CHOLMOD");
	}
	
	CholmodCommon::~CholmodCommon() {
		cholmod_l_finish(c.get());
	}

	CholmodCommon::RestrictedAccess CholmodCommon::get() {
		return RestrictedAccess(c.get(), lock);
	}

	std::function<void(cholmod_sparse*)> CholmodCommon::get_deleter() {
		return [&](cholmod_sparse* _toDelete) {
			cholmod_l_free_sparse(&_toDelete, this->get());
		};
	}
	
	thread_local CholmodCommon cholmodObject;

	
	
	CholmodSparse::CholmodSparse(cholmod_sparse* _matrix) : matrix(_matrix, cholmodObject.get_deleter()) { }

	
	CholmodSparse::CholmodSparse(const size_t _m, const size_t _n, const size_t _N) 
		 : matrix(cholmod_l_allocate_sparse(_m, _n, _N, 1, 1, 0, CHOLMOD_REAL, cholmodObject.get()), cholmodObject.get_deleter())
	{
		REQUIRE(matrix && cholmodObject.c->status == 0, "cholmod_allocate_sparse did not allocate anything... status: " << cholmodObject.c->status << " call: " << _m << " " << _n << " " << _N << " alloc: " << cholmodObject.c->malloc_count);
	}

	CholmodSparse::CholmodSparse(const std::map<size_t, double>& _input, const size_t _m, const size_t _n, const bool _transpose) 
		: CholmodSparse(_n, _m, _input.size())
	{
		size_t entryPos = 0;
		long* i = static_cast<long*>(matrix->i);
		long* p = static_cast<long*>(matrix->p);
		double* x = static_cast<double*>(matrix->x);
		i[0] = 0;
		
		// create compressed column storage of A^T aka compressed row storage of A
		
		long currRow = -1;
		
		for(const auto& entry : _input) {
			x[entryPos] = entry.second;
			i[entryPos] = static_cast<long>(entry.first%_n);
			while(currRow < static_cast<long>(entry.first/_n)) {
				p[++currRow] = long(entryPos);
			}
			entryPos++;
		}
		
		REQUIRE(currRow < long(_m) && entryPos == _input.size(), "cholmod error - invalid input? " << currRow << ", " << _m << " | " << entryPos << ", " <<  _input.size());
		
		while(currRow < static_cast<long>(_m)) {
			p[++currRow] = long(entryPos);
		}

		if(!_transpose) {
			// we didn't want A^T, so transpose the data to get compressed column storage of A
			transpose();
		}
	}

	std::map<size_t, double> CholmodSparse::to_map(double _alpha) const {
		std::map<size_t, double> result;
		long* mi = reinterpret_cast<long*>(matrix->i);
		long* p = reinterpret_cast<long*>(matrix->p);
		double* x = reinterpret_cast<double*>(matrix->x);
		
		for(size_t i = 0; i < matrix->ncol; ++i) {
			for(long j = p[i]; j < p[i+1]; ++j) {
				IF_CHECK( auto ret = ) result.emplace(size_t(mi[size_t(j)])*matrix->ncol+i, _alpha*x[j]);
				INTERNAL_CHECK(ret.second, "Internal Error");
			}
		}
		return result;
	}

	CholmodSparse CholmodSparse::operator*(const CholmodSparse& _rhs) const {
		return CholmodSparse(cholmod_l_ssmult(matrix.get(), _rhs.matrix.get(), 0, 1, 1, cholmodObject.get()));
	}

	void CholmodSparse::transpose() {
		ptr_type newM(cholmod_l_allocate_sparse(matrix->ncol, matrix->nrow, matrix->nzmax, 1, 1, 0, CHOLMOD_REAL, cholmodObject.get()), cholmodObject.get_deleter());
		cholmod_l_transpose_unsym(matrix.get(), 1, nullptr, nullptr, 0, newM.get(), cholmodObject.get());
		matrix = std::move(newM);
	}

	
	void CholmodSparse::matrix_matrix_product( std::map<size_t, double>& _C,
								const size_t _leftDim,
								const size_t _rightDim,
								const double _alpha,
								const std::map<size_t, double>& _A,
								const bool _transposeA,
								const size_t _midDim,
								const std::map<size_t, double>& _B,
								const bool _transposeB ) 
	{
// 		LOG(ssmult, _leftDim << " " << _midDim << " " << _rightDim << " " << _transposeA << " " << _transposeB);
		const CholmodSparse lhsCs(_A, _transposeA?_midDim:_leftDim, _transposeA?_leftDim:_midDim, _transposeA);
		const CholmodSparse rhsCs(_B, _transposeB?_rightDim:_midDim, _transposeB?_midDim:_rightDim, _transposeB);
		const CholmodSparse resultCs = lhsCs * rhsCs;
		_C = resultCs.to_map(_alpha);
	}
	
	void CholmodSparse::solve_sparse_rhs(std::map<size_t, double>& _x,
						  size_t _xDim,
					   const std::map<size_t, double>& _A,
					   const bool _transposeA,
					   const std::map<size_t, double>& _b,
					   size_t _bDim)
	{
		const CholmodSparse A(_A, _transposeA?_xDim:_bDim, _transposeA?_bDim:_xDim, _transposeA);
		const CholmodSparse b(_b, 1, _bDim, true); // avoids transpose call by giving transposed dimensions
		CholmodSparse x(SuiteSparseQR<double>(0, xerus::EPSILON, A.matrix.get(), b.matrix.get(), cholmodObject.get()));
		_x = x.to_map();
	}
	
	
	void CholmodSparse::solve_dense_rhs(double * _x,
								 size_t _xDim,
							  const std::map<size_t, double>& _A,
							  const bool _transposeA,
							  const double* _b,
							  size_t _bDim)
	{
		REQUIRE(_xDim < std::numeric_limits<long>::max() && _bDim < std::numeric_limits<long>::max() && _A.size() < std::numeric_limits<long>::max(),
			"sparse matrix given to qc decomposition too large for suitesparse"
		);
		const CholmodSparse A(_A, _transposeA?_xDim:_bDim, _transposeA?_bDim:_xDim, _transposeA);
		cholmod_dense b{
			_bDim, 1, _bDim, _bDim,
			static_cast<void*>(const_cast<double*>(_b)), nullptr, 
			CHOLMOD_REAL, CHOLMOD_DOUBLE
		};
		std::unique_ptr<cholmod_dense, std::function<void(cholmod_dense*)>> x(
			SuiteSparseQR<double>(A.matrix.get(), &b, cholmodObject.get()), 
			[](cholmod_dense* _toDelete) {
				cholmod_l_free_dense(&_toDelete, cholmodObject.get());
			}
		);
		INTERNAL_CHECK(x->z == nullptr, "IE");
		INTERNAL_CHECK(x->nrow == _xDim, "IE");
		misc::copy(_x, static_cast<double*>(x->x), _xDim);
	}
	
	
	std::tuple<std::map<size_t, double>, std::map<size_t, double>, size_t> CholmodSparse::qc(
				const std::map<size_t, double> &_A,
				const bool _transposeA,
				size_t _m,
				size_t _n,
				bool _fullrank)
	{
		REQUIRE(_m < std::numeric_limits<long>::max() && _n < std::numeric_limits<long>::max() && _A.size() < std::numeric_limits<long>::max(),
			"sparse matrix given to qc decomposition too large for suitesparse"
		);
		REQUIRE(_m>0 && _n>0, "invalid matrix of dimensions " << _m << 'x' << _n);
		CholmodSparse A(_A, _m, _n, _transposeA);
		cholmod_sparse *Q, *R;
		SuiteSparse_long *E;
		long rank = SuiteSparseQR<double>(0, xerus::EPSILON, _fullrank?long(std::min(_m,_n)):1, A.matrix.get(), &Q, &R, &E, cholmodObject.get());
		rank = std::max(rank, 1l);
		CholmodSparse Qs(Q);
		CholmodSparse Rs(R);
		INTERNAL_CHECK(E == nullptr, "IE: sparse QR returned a permutation despite fixed ordering?!");
		INTERNAL_CHECK(rank>=0, "SuiteSparseQR returned negative value " << rank);
		INTERNAL_CHECK((_fullrank?std::min(_m,_n):size_t(rank)) == Qs.matrix->ncol, "IE: strange rank deficiency after sparse qr " << (_fullrank?long(std::min(_m,_n)):rank) << " vs " << Qs.matrix->ncol);
		return std::make_tuple(Qs.to_map(), Rs.to_map(), _fullrank?std::min(_m,_n):size_t(rank));
	}
	
	std::tuple<std::map<size_t, double>, std::map<size_t, double>, size_t> CholmodSparse::cq(
				const std::map<size_t, double> &_A,
				const bool _transposeA,
				size_t _m,
				size_t _n,
				bool _fullrank)
	{
		REQUIRE(_m < std::numeric_limits<long>::max() && _n < std::numeric_limits<long>::max() && _A.size() < std::numeric_limits<long>::max(),
			"sparse matrix given to qc decomposition too large for suitesparse"
		);
		CholmodSparse A(_A, _n, _m, !_transposeA);
		cholmod_sparse *Q, *R;
		SuiteSparse_long *E;
		// decompose A^T = q^T*r^T
		long rank = SuiteSparseQR<double>(0, xerus::EPSILON, _fullrank?long(std::min(_m,_n)):1, A.matrix.get(), &Q, &R, &E, cholmodObject.get());
		rank = std::max(rank, 1l);
		CholmodSparse Qs(Q);
		CholmodSparse Rs(R);
		INTERNAL_CHECK(E == nullptr, "IE: sparse QR returned a permutation despite fixed ordering?!");
		INTERNAL_CHECK(rank>=0, "SuiteSparseQR returned negative value " << rank);
		INTERNAL_CHECK((_fullrank?std::min(_m,_n):size_t(rank)) == Qs.matrix->ncol, "IE: strange rank deficiency after sparse qr " << (_fullrank?long(std::min(_m,_n)):rank) << " vs " << Qs.matrix->ncol);
		//transpose q and r to get r*q=A
		Qs.transpose();
		Rs.transpose();
		return std::make_tuple(Rs.to_map(), Qs.to_map(), _fullrank?std::min(_m,_n):size_t(rank));
	}


} // namespace internal
} // namespace xerus
