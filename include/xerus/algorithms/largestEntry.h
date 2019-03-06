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
 * @brief Header file for the TT largest entry algorithm.
 */

#pragma once
#include "../ttNetwork.h"

namespace xerus {
	/** 
	 * @brief Finds the position of the approximately largest entry.
	 * @details Finds an entry that is at least of size @a _accuracy * X_max in absolute value,
	 * where X_max is the largest entry of the tensor. The smaller @a _accuracy, the faster the algorithm will work.
	 * @param _T TTNetwork to check for largest entry
	 * @param _accuracy factor that determains the maximal deviation of the returned entry from the true largest entry.
	 * @param _lowerBound a lower bound for the largest entry, i.e. there must be an entry in the tensor which is at least of
	 * this size (in absolute value). The algorithm may fail completely if this is not fullfilled, but will work using its own 
	 * approximation if no value (i.e. 0.0) is given.
	 * @return the position of the entry found.
	 */
	template<bool isOperator>
	size_t find_largest_entry(const TTNetwork<isOperator> &_T, double _accuracy, value_t _lowerBound = 0.0);
}

