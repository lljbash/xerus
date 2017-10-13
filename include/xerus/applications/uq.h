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
 * @brief Header file for utilities for uncertainity quantification.
 */

#pragma once

#include "../ttNetwork.h"

namespace xerus {namespace uq {
	
	enum class PolynomBasis : bool { Hermite, Legendre };
    
    Tensor hermite_position(const double _v, const size_t _polyDegree);
	
	
	Tensor legendre_position(const double _v, const size_t _polyDegree);
	
    
	class UQMeasurementSet {
	public:
		std::vector<std::vector<double>> randomVectors;
		std::vector<Tensor> solutions;
		
		std::vector<std::vector<double>> initialRandomVectors;
		std::vector<Tensor> initialSolutions;
		
		UQMeasurementSet() = default;
		UQMeasurementSet(const UQMeasurementSet&  _other) = default;
		UQMeasurementSet(      UQMeasurementSet&& _other) = default;
		
		void add(const std::vector<double>& _rndvec, const Tensor& _solution);
		
		void add_initial(const std::vector<double>& _rndvec, const Tensor& _solution);
        
        void clear();
	};
	
    
	TTTensor initial_guess(const UQMeasurementSet& _measurments, const TTTensor& _guess);
	
    
	Tensor mc_average(const TTTensor& _x, const size_t _N);
	
	Tensor average(const TTTensor& _x);
}}


