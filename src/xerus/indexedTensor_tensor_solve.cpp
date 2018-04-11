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
* @brief Implementation of the indexed tensor / operator.
*/

#include <xerus/misc/internal.h>
#include <xerus/misc/containerSupport.h>

#include <xerus/index.h>
#include <xerus/indexedTensor.h>
#include <xerus/indexedTensorMoveable.h>
#include <xerus/tensor.h>

namespace xerus {
	internal::IndexedTensorMoveable<Tensor> operator/ (internal::IndexedTensorReadOnly<Tensor>&& _b, internal::IndexedTensorReadOnly<Tensor>&& _A) {
		_A.assign_indices();
		_b.assign_indices();
		
		size_t extraDims = 0;
		std::vector<Index> orderA;
		std::vector<Index> orderB;
		std::vector<Index> orderX;
		
		// If possible we don't want to reorder A, so first divide A into those shared with b and x. 
		for(const Index& idx : _A.indices) {
			if(misc::contains(_b.indices, idx)) {
				orderA.push_back(idx);
			} else {
				orderX.push_back(idx);
			}
		}
		
		// This far the dimensions of A and B coincide
		orderB = orderA;
		
		// Add the second part of indices in the order obtained for X
		orderA.insert(orderA.end(), orderX.begin(), orderX.end());
		
		// Now complete indices of b and x with those not shared with A ( in order of b as we don't want to reorder b if possible).
		for(const Index& idx : _b.indices) {
			if(!misc::contains(_A.indices, idx)) {
				orderB.push_back(idx);
				orderX.push_back(idx);
				for(size_t i = 0; i < idx.span; ++i) {
					extraDims++;
				}
			}
		}
		
		// If indices coincide no reordering occours (only shared data pointer is set).
		Tensor reorderedA, reorderedB;
		reorderedA(orderA) = std::move(_A);
		reorderedB(orderB) = std::move(_b);
		
		internal::IndexedTensorMoveable<Tensor> tmpX(new Tensor(), std::move(orderX));
		
		//solve_least_squares(*tmpX.tensorObject, reorderedA, reorderedB, extraDims);
		solve(*tmpX.tensorObject, reorderedA, reorderedB, extraDims);
		
		return tmpX;
	}
} // namespace xerus
