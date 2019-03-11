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
 * @brief Header file for some example measure sets for tensor completion.
 */

#pragma once
#include "../ttNetwork.h"
#include "../measurments.h"

namespace xerus { namespace examples { namespace completion {
	
	/**
	 * @brief fills in the values of the @a _measurements as @$ 1/\sqrt{\sum i_\mu^2} @$
	 * @details the corresponding tensor is only approximately low-rank, cf. Hackbusch, 2012 "Tensor spaces and numerical tensor calculus" and http://www.mis.mpg.de/scicomp/EXP_SUM
	 * @note that index (0,0,0,0...) is singular for @a _additiveConst = 0
	 */
	value_t inverse_index_norm(const std::vector<size_t>& _position, const value_t _additiveConst = 1.0);
	
	
	
	/**
	 * @brief fills in the values of the @a _measurements as @$ (\alpha + \sum i_\mu/(i_{\mu+1} + \alpha - 1))^{-1} @$
	 * @details the corresponding tensor is only approximately low-rank, cf. Grasedyck, Kluge, Kraemer 2015
	 */
	value_t inverse_index_ratio(const std::vector<size_t>& _position, const value_t _additiveConst = 1.0);
}}}

