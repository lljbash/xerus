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
 * @brief Implementation of the IndexedTensorList.
 */

#include <xerus/indexedTensorList.h>
#include <xerus/indexedTensor_tensor_factorisations.h>

namespace xerus {
	namespace internal {
		IndexedTensorList::IndexedTensorList(IndexedTensorList&& _old) noexcept : tensors(std::move(_old.tensors)) { } 
		
		IndexedTensorList::IndexedTensorList(IndexedTensor<Tensor>&& _first, IndexedTensor<Tensor>&& _second) {
			tensors.emplace_back(&_first);
			tensors.emplace_back(&_second);
		}
		
		void IndexedTensorList::operator=(TensorFactorisation&& _factorisation) const {
			_factorisation(tensors);
		}
		
		/*- - - - - - - - - - - - - - - - - - - - - - - - - - External functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -*/
		
		IndexedTensorList operator,(IndexedTensor<Tensor>&& _first, IndexedTensor<Tensor>&& _second) {
			return IndexedTensorList(std::move(_first), std::move(_second));
		}

		IndexedTensorList operator,(IndexedTensorList&& _first, IndexedTensor<Tensor>&& _second) {
			_first.tensors.emplace_back(&_second); // Hope this is standardconform. maybe we have to move-construct a new object
			return std::move(_first);
		}
	} // namespace internal
} // namespace xerus
