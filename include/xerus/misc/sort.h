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
* @brief Header file for some extended sorting functions.
*/

#include <vector>
#include <algorithm>

#include "check.h"

namespace xerus {
	namespace misc {
		template<class T, class Comparator>
		std::vector<size_t> create_sort_permutation(const std::vector<T>& _vec, Comparator _comp) {
			std::vector<size_t> permutation(_vec.size());
			std::iota(permutation.begin(), permutation.end(), 0);
			std::sort(permutation.begin(), permutation.end(), [&](const size_t _i, const size_t _j){ return _comp(_vec[_i], _vec[_j]); });
			return permutation;
		}
		
		template<class T>
		void apply_permutation( std::vector<T>& _vec, const std::vector<size_t>& _permutation) {
			XERUS_REQUIRE(_vec.size() == _permutation.size(), "Vector and permutation size must coincide.");
			std::vector<T> sorted_vec;
			sorted_vec.reserve(_permutation.size());
			for(size_t i = 0; i < _permutation.size(); ++i) {
				sorted_vec.emplace_back(std::move(_vec[_permutation[i]]));
			}
			_vec = std::move(sorted_vec);
		}
		
		template <class KeyType, class DataType, class Comparator>
		void simultaneous_sort( std::vector<KeyType>& _keyVector, std::vector<DataType>& _dataVector, Comparator _comp) {
			XERUS_REQUIRE(_keyVector.size() == _dataVector.size(), "Vector sizes must coincide.");
			const std::vector<size_t> permutation = create_sort_permutation(_keyVector, _comp);
			apply_permutation(_keyVector, permutation);
			apply_permutation(_dataVector, permutation);
		}
	}
}
