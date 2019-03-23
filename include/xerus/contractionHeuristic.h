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
 * @brief Header file for the class managing the contraction heuristics.
 */

#pragma once

#include <vector>

#include "basic.h"

namespace xerus {
	class TensorNetwork;
	
	namespace internal {
		using ContractionHeuristic = void (*)(double &, std::vector<std::pair<size_t,size_t>> &, TensorNetwork);
		
		template<double (*scoreFct)(double, double, double, double, double)>
		void greedy_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network);
		
		double contraction_cost(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		
		double score_size(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		double score_mn(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		double score_speed(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		double score_r(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		double score_big_tensor(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		double score_littlestep(double _m, double _n, double _r, double _sparsity1, double _sparsity2);
		
		
		void greedy_best_of_three_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network);
		void exchange_heuristic(double &_bestCost, std::vector<std::pair<size_t,size_t>> &_contractions, TensorNetwork _network);
		
		extern const std::vector<ContractionHeuristic> contractionHeuristics;
	}
}
