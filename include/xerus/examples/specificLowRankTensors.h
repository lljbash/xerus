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
 * @brief Header file for some examples of low-rank TT tensors.
 */

#pragma once
#include "../ttNetwork.h"

namespace xerus { namespace examples {
	
	/**
	 * @brief Constructs a rank _n+1 TTTensor of degree @a _degree and external dimensions @a _n that has entries >0, maximal where all indices coincide.
	 * @details Constructed as a sum of nearest-neighbor terms that each have entries as 1/(std::abs(i-j)+alpha)
	 */
	TTTensor peaking_diagonals(size_t _degree, size_t _n, value_t _alpha = 1.0);
	
}}

