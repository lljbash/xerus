// Xerus - A General Purpose Tensor Library
// Copyright (C) 2014-2017 Benjamin Huber and Sebastian Wolf. 
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
* @brief Header file for the OptimizationAlgorithm class.
*/

#pragma once

#include "../performanceData.h"

namespace xerus {

	/**
	* @brief Base class for all xerus optimization algorithms, allowing a uniform set of settings
	*/
	class OptimizationAlgorithm {
	public:
		///@brief Maximal allowed number of iterations
		const size_t maxIterations; 
		
		///@brief The target relative residual at which the algorithm shall stop.
		const double targetRelativeResidual;
		
		///@brief Minimal relative decrease of the residual norm ( newRes/oldRes ) until either the ranks are increased (if allowed) or the algorithm stops.
		const double minimalResidualNormDecrease;
		
	protected:
		OptimizationAlgorithm(const size_t _maxIterations, const double _targetRelativeResidual, const double _minimalResidualNormDecrease);
	};
} // namespace xerus
