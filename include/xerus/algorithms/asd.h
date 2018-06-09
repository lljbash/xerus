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
 * @brief Header file for the ADF algorithm and its variants.
 */

#pragma once

#include "optimizationAlgorithm.h"
#include "../ttNetwork.h"
#include "../performanceData.h"
#include "../measurments.h"

namespace xerus {
	
	class ASDVariant : public OptimizationAlgorithm {
	public:

		double minRankEps = 1e-4;
		
// 		double maxRankEps = 1e-1;
		
		double epsDecay = 1.1;
		
		double controlSetFraction = 0.1;
		
		double initialRankEps = 5e-3;
		
		/// Basic constructor
        ASDVariant(const size_t _maxIterations, const double _targetRelativeResidual, const double _minimalResidualNormDecrease)
                : OptimizationAlgorithm(0, _maxIterations, _targetRelativeResidual, _minimalResidualNormDecrease) { }
        
        /**
		* @brief Tries to reconstruct the (low rank) tensor _x from the given measurments. 
		* @param[in,out] _x On input: an initial guess of the solution, also defining the ranks. On output: The reconstruction found by the algorithm.
		* @param _measurments the available measurments, can be either a SinglePointMeasurementSet or RankOneMeasurementSet.
		* @param _perfData optinal performanceData object to be used.
		* @returns nothing
		*/
		void operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, PerformanceData& _perfData) const;
		
		/**
		* @brief Tries to reconstruct the (low rank) tensor _x from the given measurments. 
		* @param[in,out] _x On input: an initial guess of the solution, may be of smaller rank. On output: The reconstruction found by the algorithm.
		* @param _measurments the available measurments, can be either a SinglePointMeasurementSet or RankOneMeasurementSet.
		* @param _maxRanks the maximal ranks the algorithm may use to decrease the resdiual.
		* @param _perfData optinal performanceData object to be used.
		* @returns nothing
		*/
		void operator()(TTTensor& _x, const RankOneMeasurementSet& _measurments, const std::vector<size_t>& _maxRanks, PerformanceData& _perfData) const;
	};
	
	/// @brief Default variant of the ASD algorithm
    extern const ASDVariant TRASD;
}

