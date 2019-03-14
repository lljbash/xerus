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
 * @brief Header file for the ADF algorithm and its variants.
 */

#pragma once

#include "optimizationAlgorithm.h"
#include "../forwardDeclarations.h"

namespace xerus {
	
	/**
	 * @brief Wrapper class for all ADF variants.
	 * @details By creating a new object of this class and modifying the member variables, the behaviour of the solver can be modified.
	 * This algorithm is a modified implementation of the alternating directional fitting algrothim, first introduced by Grasedyck, Kluge and Kraemer (2015).
	 */
    class ADFVariant : public OptimizationAlgorithm {
	public:
		///@brief Fully defining constructor. alternatively ALSVariants can be created by copying a predefined variant and modifying it
        ADFVariant(const size_t _maxIteration, const double _targetRelativeResidual, const double _minimalResidualDecrease);
        
        /**
		* @brief Tries to reconstruct the (low rank) tensor _x from the given measurments. 
		* @param[in,out] _x On input: an initial guess of the solution, also defining the ranks. On output: The reconstruction found by the algorithm.
		* @param _measurments the available measurments, can be either a SinglePointMeasurementSet or RankOneMeasurementSet.
		* @param _perfData optinal performanceData object to be used.
		* @returns the residual @f$|P_\Omega(x-b)|_2@f$ of the final @a _x.
		*/
		template<class MeasurmentSet>
		double operator()(TTTensor& _x, const MeasurmentSet& _measurments, PerformanceData& _perfData = NoPerfData) const;
		
		
		/**
		* @brief Tries to reconstruct the (low rank) tensor _x from the given measurments. 
		* @param[in,out] _x On input: an initial guess of the solution, may be of smaller rank. On output: The reconstruction found by the algorithm.
		* @param _measurments the available measurments, can be either a SinglePointMeasurementSet or RankOneMeasurementSet.
		* @param _maxRanks the maximal ranks the algorithm may use to decrease the resdiual.
		* @param _perfData optinal performanceData object to be used.
		* @returns the residual @f$|P_\Omega(x-b)|_2@f$ of the final @a _x.
		*/
		template<class MeasurmentSet>
		double operator()(TTTensor& _x, const MeasurmentSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData = NoPerfData) const;
	};
	
	/// @brief Default variant of the ADF algorithm
    extern const ADFVariant ADF;
}
