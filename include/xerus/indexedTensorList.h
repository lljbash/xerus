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
 * @brief Header file defining lists of indexed tensors.
 */

#pragma once

#include <vector>

namespace xerus {
	// Necessary forward declaritons
	class Tensor;
	class TensorFactorisation;
	
	namespace internal {
		template<class tensor_type> class IndexedTensor;
		/**
		* @brief Internal representation of a tuple of indexed Tensor s.
		* @details This class appears inplicitly by using expressiong like (Q(i,r), R(r,j)) and is particulary used for a convinient syntax for Tensor factorisations.
		*/
		class IndexedTensorList {
		public:
			///@brief Collection of pointers to the Tensor objects referenced by the tuple.
			std::vector<IndexedTensor<Tensor>*> tensors;
			
			///@brief No default construction is intended.
			IndexedTensorList() = delete;
			
			///@brief No copy construction is intended.
			IndexedTensorList(const IndexedTensorList& _old) = delete;
			
			///@brief Move constructor.
			IndexedTensorList(IndexedTensorList&& _old) noexcept;
			
			/**
			* @brief constructor initializing an IndexedTensorList with two initial Tensor refrences.
			*/
			IndexedTensorList(IndexedTensor<Tensor>&& _first, IndexedTensor<Tensor>&& _second);
			
			/**
			* @brief Generic assignment operator that takes any TensorFactorisation object which is in then invoked to perform the assignment.
			*/
			void operator=(TensorFactorisation&& _factorisation) const;
		};

		/**
		* @brief Using the "," operator tuples of writeable indexed tensor can be created.
		* @param _first the first element of the tuple.
		* @param _second the second element of the tuple.
		* @returns an IndexedTensorList representing the desired 2-tuple.
		*/
		IndexedTensorList operator,(IndexedTensor<Tensor>&& _first, IndexedTensor<Tensor>&& _second);

		/**
		* @brief Using the "," operator tuples of writeable indexed tensor can be created.
		* @param _first an existing tuple of writeable indexed tensor.
		* @param _second a further writeable indexed tensor that shall be appended to the existing tuple.
		* @returns an IndexedTensorList representing the desired (n+1)-tuple.
		*/
		IndexedTensorList operator,(IndexedTensorList &&_first, IndexedTensor<Tensor>&& _second);
	}
}
