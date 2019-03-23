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
 * @brief Implementation of the Tensor factorizations.
 */

#include <xerus/indexedTensor_tensor_factorisations.h>
#include <xerus/index.h>
#include <xerus/tensor.h>
 
#include <xerus/blasLapackWrapper.h>
#include <xerus/misc/basicArraySupport.h>
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/internal.h>

namespace xerus {
	
	std::unique_ptr<Tensor> prepare_split(size_t& _lhsSize, size_t& _rhsSize, size_t& _rank, size_t& _splitPos, std::vector<Index>& _lhsPreliminaryIndices, std::vector<Index>& _rhsPreliminaryIndices, internal::IndexedTensorReadOnly<Tensor>&& _base, internal::IndexedTensor<Tensor>&& _lhs, internal::IndexedTensor<Tensor>&& _rhs) {
		_base.assign_indices();
		
		// Calculate the future order of lhs and rhs.
		size_t lhsOrder = 1, rhsOrder = 1; // Start with 1 because there is a new dimension introduced in the split. 
		
		for (const Index& idx : _base.indices) {
			if (idx.open()) {
				if(misc::contains(_lhs.indices, idx)) {
					lhsOrder += idx.span; 
				} else {
					REQUIRE(misc::contains(_rhs.indices, idx), "Every open index of factorisation base must be contained in one of the targets");
					rhsOrder += idx.span;
				}
			}
		}
		
		_splitPos = lhsOrder-1;
		
		_lhs.assign_indices(lhsOrder);
		_rhs.assign_indices(rhsOrder);
		
		std::vector<Index> reorderedBaseIndices;
		reorderedBaseIndices.reserve(_base.indices.size());
		
		_lhsPreliminaryIndices.reserve(_lhs.indices.size());
		
		_rhsPreliminaryIndices.reserve(_rhs.indices.size());
		
		std::vector<size_t> reorderedBaseDimensions, lhsDims, rhsDims;
		reorderedBaseDimensions.reserve(_base.order());
		lhsDims.reserve(_lhs.indices.size());
		rhsDims.reserve(_rhs.indices.size());
		
		_lhsSize=1;
		_rhsSize=1;
		
		Index auxiliaryIndex;

		// Work through the indices of lhs
		IF_CHECK(bool foundCommon = false;)
		for (const auto &idx : _lhs.indices) {
			// Find index in A and get dimension offset
			size_t j, dimOffset = 0;
			for(j = 0; j < _base.indices.size() && idx != _base.indices[j]; ++j) {
				dimOffset += _base.indices[j].span;
			}
			
			if(j < _base.indices.size()) {
				_lhsPreliminaryIndices.push_back(_base.indices[j]);
				reorderedBaseIndices.push_back(_base.indices[j]);
				for(size_t k = 0; k < _base.indices[j].span; ++k) {
					reorderedBaseDimensions.push_back(_base.tensorObjectReadOnly->dimensions.at(dimOffset+k));
					lhsDims.push_back(_base.tensorObjectReadOnly->dimensions[dimOffset+k]);
					_lhsSize *= _base.tensorObjectReadOnly->dimensions[dimOffset+k];
				}
			} else {
				REQUIRE(!foundCommon, "Left part of factorization must have exactly one index that is not contained in base. Here it is more than one.");
				IF_CHECK(foundCommon = true;)
				auxiliaryIndex = idx;
			}
		}
		_lhsPreliminaryIndices.push_back(auxiliaryIndex);

		// Work through the indices of rhs
		IF_CHECK(foundCommon = false;)
		for (const auto &idx : _rhs.indices) {
			// Find index in A and get dimension offset
			size_t j, dimOffset = 0;
			for(j = 0; j < _base.indices.size() && idx != _base.indices[j]; ++j) {
				dimOffset += _base.indices[j].span;
			}
			
			if(j < _base.indices.size()) {
				_rhsPreliminaryIndices.push_back(_base.indices[j]);
				reorderedBaseIndices.push_back(_base.indices[j]);
				for(size_t k = 0; k < _base.indices[j].span; ++k) {
					reorderedBaseDimensions.push_back(_base.tensorObjectReadOnly->dimensions.at(dimOffset+k));
					rhsDims.push_back(_base.tensorObjectReadOnly->dimensions[dimOffset+k]);
					_rhsSize *= _base.tensorObjectReadOnly->dimensions[dimOffset+k];
				}
			} else {
				REQUIRE(!foundCommon, "Right part of factorization must have exactly one index that is not contained in base. Here it is more than one.");
				IF_CHECK(foundCommon = true;)
				auxiliaryIndex = idx;
			}
		}
		_rhsPreliminaryIndices.insert(_rhsPreliminaryIndices.begin(), auxiliaryIndex);
		
		internal::IndexedTensor<Tensor> reorderedBaseTensor(new Tensor(std::move(reorderedBaseDimensions), _base.tensorObjectReadOnly->representation, Tensor::Initialisation::None), std::move(reorderedBaseIndices), false);
		evaluate(std::move(reorderedBaseTensor), std::move(_base));
		reorderedBaseTensor.tensorObject->ensure_own_data();
		
		_rank = std::min(_lhsSize, _rhsSize);
		lhsDims.push_back(_rank);
		rhsDims.insert(rhsDims.begin(), _rank);
		
		_lhs.tensorObject->reset(std::move(lhsDims), Tensor::Initialisation::None);
		_lhs.tensorObject->ensure_own_data_no_copy();
		
		_rhs.tensorObject->reset(std::move(rhsDims), Tensor::Initialisation::None);
		_rhs.tensorObject->ensure_own_data_no_copy();
		
		return std::unique_ptr<Tensor>(reorderedBaseTensor.tensorObject);
	}
	
	void SVD::operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const {
		REQUIRE(_output.size() == 3, "SVD requires two output tensors, not " << _output.size());
		internal::IndexedTensorReadOnly<Tensor>& A = *input;
		internal::IndexedTensor<Tensor>& U = *_output[0];
		internal::IndexedTensor<Tensor>& S = *_output[1];
		internal::IndexedTensor<Tensor>& Vt = *_output[2];
		
		REQUIRE(epsilon < 1, "Epsilon must be smaller than one.");
		REQUIRE(maxRank > 0, "maxRank must be larger than zero.");
		
		size_t lhsSize, rhsSize, rank, splitPos;
		std::vector<Index> lhsPreliminaryIndices, rhsPreliminaryIndices;
		
		std::unique_ptr<Tensor> reorderedBaseTensor = prepare_split(lhsSize, rhsSize, rank, splitPos, lhsPreliminaryIndices, rhsPreliminaryIndices, std::move(A), std::move(U), std::move(Vt));
		
		calculate_svd(*U.tensorObject, *S.tensorObject, *Vt.tensorObject, *reorderedBaseTensor, splitPos, maxRank, epsilon);
		
		if(softThreshold > 0.0) {
			const size_t oldRank = S.tensorObjectReadOnly->dimensions[0];
			rank = oldRank;
			
			(*S.tensorObject)[0] = std::max((*S.tensorObject)[0]-softThreshold, preventZero ? EPSILON*(*S.tensorObject)[0] : 0.0);
			
			for(size_t i = 1; i < oldRank; ++i) {
				if((*S.tensorObject)[i+i*oldRank] < softThreshold) {
					rank = i;
					break;
				} else {
					(*S.tensorObject)[i+i*oldRank] -= softThreshold;
				}
			}
			
			if(rank != oldRank) {
				Tensor newS({rank, rank}, Tensor::Representation::Sparse);
				for(size_t i = 0; i < rank; ++i) {
					newS[i+i*rank] = (*S.tensorObject)[i+i*oldRank];
				}
				*S.tensorObject = newS;
				
				U.tensorObject->resize_mode(U.order()-1, rank);
				Vt.tensorObject->resize_mode(0, rank);
			}
		}
		
		// Post evaluate the results
		std::vector<Index> midPreliminaryIndices({lhsPreliminaryIndices.back(), rhsPreliminaryIndices.front()});
		U = (*U.tensorObjectReadOnly)(lhsPreliminaryIndices);
		S = (*S.tensorObjectReadOnly)(midPreliminaryIndices);
		Vt = (*Vt.tensorObjectReadOnly)(rhsPreliminaryIndices);
	}


	void QR::operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const {
		REQUIRE(_output.size() == 2, "QR factorisation requires two output tensors, not " << _output.size());
		internal::IndexedTensorReadOnly<Tensor>& A = *input;
		internal::IndexedTensor<Tensor>& Q = *_output[0];
		internal::IndexedTensor<Tensor>& R = *_output[1];
		
		size_t lhsSize, rhsSize, rank, splitPos;
		std::vector<Index> lhsPreliminaryIndices, rhsPreliminaryIndices;
		
		std::unique_ptr<Tensor> reorderedBaseTensor = prepare_split(lhsSize, rhsSize, rank, splitPos, lhsPreliminaryIndices, rhsPreliminaryIndices, std::move(A), std::move(Q), std::move(R));
		
		calculate_qr(*Q.tensorObject, *R.tensorObject, *reorderedBaseTensor, splitPos);
		
		// Post evaluate the results
		Q = (*Q.tensorObjectReadOnly)(lhsPreliminaryIndices);
		R = (*R.tensorObjectReadOnly)(rhsPreliminaryIndices);
	}

	void RQ::operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const {
		REQUIRE(_output.size() == 2, "RQ factorisation requires two output tensors, not " << _output.size());
		internal::IndexedTensorReadOnly<Tensor>& A = *input;
		internal::IndexedTensor<Tensor>& R = *_output[0];
		internal::IndexedTensor<Tensor>& Q = *_output[1];
		
		size_t lhsSize, rhsSize, rank, splitPos;
		std::vector<Index> lhsPreliminaryIndices, rhsPreliminaryIndices;
		
		std::unique_ptr<Tensor> reorderedBaseTensor = prepare_split(lhsSize, rhsSize, rank, splitPos, lhsPreliminaryIndices, rhsPreliminaryIndices, std::move(A), std::move(R), std::move(Q));
		
		calculate_rq(*R.tensorObject, *Q.tensorObject, *reorderedBaseTensor, splitPos);
		
		// Post evaluate the results
		R = (*R.tensorObjectReadOnly)(lhsPreliminaryIndices);
		Q = (*Q.tensorObjectReadOnly)(rhsPreliminaryIndices);
	}
	
	
	void QC::operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const {
		REQUIRE(_output.size() == 2, "QC factorisation requires two output tensors, not " << _output.size());
		internal::IndexedTensorReadOnly<Tensor>& A = *input;
		internal::IndexedTensor<Tensor>& Q = *_output[0];
		internal::IndexedTensor<Tensor>& C = *_output[1];
		
		size_t lhsSize, rhsSize, rank, splitPos;
		std::vector<Index> lhsPreliminaryIndices, rhsPreliminaryIndices;
		
		std::unique_ptr<Tensor> reorderedBaseTensor = prepare_split(lhsSize, rhsSize, rank, splitPos, lhsPreliminaryIndices, rhsPreliminaryIndices, std::move(A), std::move(Q), std::move(C));
		
		calculate_qc(*Q.tensorObject, *C.tensorObject, *reorderedBaseTensor, splitPos);
		
		// Post evaluate the results
		Q = (*Q.tensorObjectReadOnly)(lhsPreliminaryIndices);
		C = (*C.tensorObjectReadOnly)(rhsPreliminaryIndices);
	}
	
	void CQ::operator()(const std::vector<internal::IndexedTensor<Tensor>*>& _output) const {
		REQUIRE(_output.size() == 2, "CQ factorisation requires two output tensors, not " << _output.size());
		internal::IndexedTensorReadOnly<Tensor>& A = *input;
		internal::IndexedTensor<Tensor>& C = *_output[0];
		internal::IndexedTensor<Tensor>& Q = *_output[1];
		
		size_t lhsSize, rhsSize, rank, splitPos;
		std::vector<Index> lhsPreliminaryIndices, rhsPreliminaryIndices;
		
		std::unique_ptr<Tensor> reorderedBaseTensor = prepare_split(lhsSize, rhsSize, rank, splitPos, lhsPreliminaryIndices, rhsPreliminaryIndices, std::move(A), std::move(C), std::move(Q));
		
		calculate_cq(*C.tensorObject, *Q.tensorObject, *reorderedBaseTensor, splitPos);
		
		// Post evaluate the results
		C = (*C.tensorObjectReadOnly)(lhsPreliminaryIndices);
		Q = (*Q.tensorObjectReadOnly)(rhsPreliminaryIndices);
	}
} // namespace xerus
