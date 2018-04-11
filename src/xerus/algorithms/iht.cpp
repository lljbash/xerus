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
 * @brief Implementation of the IHT variants.
 */

#include <xerus.h>
#include "../../../include/xerus/misc/internal.h"

namespace xerus {
	double IHT(TTTensor& _x, const SinglePointMeasurementSet& _measurments, PerformanceData& _perfData) {
		const size_t numMeasurments = _measurments.size();
		const size_t degree = _x.degree();
		const size_t USER_MEASUREMENTS_PER_ITR = numMeasurments;
		const value_t ALPHA_CHG = 1.1;
		

		TTTensor largeX(_x);
		// 		SinglePointMeasurementSet currentValues(_measurments);
		std::vector<value_t> currentValues(numMeasurments);
		std::vector<size_t> measurementOrder(numMeasurments);
		std::iota(measurementOrder.begin(), measurementOrder.end(), 0);
		
		_perfData.start();
		
		Index i1, i2, i3, i4, i5;
		
// 		std::mt19937_64 rnd(rd());
// 		std::uniform_real_distribution<value_t> dist(0, 1);
		
		std::vector<size_t> twoR(_x.ranks());
		for (auto &r : twoR) {
			r *= 2;
		}
		
		double alpha = 1;
		
		value_t residual = 1;
		
		for(size_t iteration = 0; iteration < 1000000; ++iteration) {
// 			_x.measure(currentValues);
			
			for(size_t i = 0; i < numMeasurments; ++i) {
				currentValues[i] = _x[_measurments.positions[i]];
			}
			
// 			std::shuffle(measurementOrder.begin(), measurementOrder.end(), rnd);
// 			std::sort(measurementOrder.begin(), measurementOrder.end(), [&](size_t a, size_t b){
// 				return std::abs(_measurments.measuredValues[a] - currentValues[a]) > std::abs(_measurments.measuredValues[b] - currentValues[b]);
// 			});
			
			value_t bestResidual = residual*2;
			value_t newAlpha = alpha;
			
			for (value_t beta = 1/ALPHA_CHG; beta < ALPHA_CHG*1.5; beta *= ALPHA_CHG) {
				// Build the largeX
				for(size_t d = 0; d < degree; ++d) {
					Tensor& currComp = _x.component(d);
					Tensor newComp({d == 0 ? 1 : (currComp.dimensions[0]+USER_MEASUREMENTS_PER_ITR), currComp.dimensions[1], d == degree-1 ? 1 : (currComp.dimensions[2]+USER_MEASUREMENTS_PER_ITR)});
					
					// Copy dense part
					for(size_t r1 = 0; r1 < currComp.dimensions[0]; ++r1) {
						for(size_t n = 0; n < currComp.dimensions[1]; ++n) {
							for(size_t r2 = 0; r2 < currComp.dimensions[2]; ++r2) {
								newComp[{r1, n, r2}] = currComp[{r1, n, r2}];
							}
						}
					}
					
					// Copy sparse part
					if (d==0) {
						for(size_t i = 0; i < USER_MEASUREMENTS_PER_ITR; ++i) {
							newComp[{0, _measurments.positions[measurementOrder[i]][d], i+currComp.dimensions[2]}] = beta*alpha*(_measurments.measuredValues[measurementOrder[i]] - currentValues[measurementOrder[i]]);
						}
					} else if (d!=degree-1) {
						for(size_t i = 0; i < USER_MEASUREMENTS_PER_ITR; ++i) {
							newComp[{i + currComp.dimensions[0], _measurments.positions[measurementOrder[i]][d], i+currComp.dimensions[2]}] = 1.0;
						}
					} else {
						// d == degree-1
						for(size_t i = 0; i < USER_MEASUREMENTS_PER_ITR; ++i) {
							newComp[{i + currComp.dimensions[0], _measurments.positions[measurementOrder[i]][d], 0}] = 1.0;
						}
					}
					
					
					largeX.set_component(d, newComp);
				}
				// one ALS sweep to 2r
				TTTensor newX = _x;//TTTensor::random(_x.dimensions, twoR, dist);
				for (size_t o=0; o<1; ++o) {
					newX.move_core(0, true);
					// build stack from right to left
					std::vector<Tensor> stack;
					stack.emplace_back(Tensor::ones({1,1}));
					for (size_t i=degree-1; i>0; --i) {
						Tensor next;
						next(i1,i2) = newX.get_component(i)(i1,i5,i3) * largeX.get_component(i)(i2,i5,i4) * stack.back()(i3,i4);
						stack.emplace_back(next);
					}
					Tensor left = Tensor::ones({1,1});
					// sweep left to right
					for (size_t i=0; i<degree; ++i) {
						newX.component(i)(i1,i2,i3) = left(i1,i4) * largeX.get_component(i)(i4,i2,i5) * stack.back()(i3,i5);
						if (i+1<degree) {
							newX.move_core(i+1, true);
							left(i1,i2) = left(i3,i4) * newX.get_component(i)(i3,i5,i1) * largeX.get_component(i)(i4,i5,i2);
							stack.pop_back();
						}
					}
				}
				residual = 0;
				
				for(size_t i = 0; i < numMeasurments; ++i) {
					residual += misc::sqr(_measurments.measuredValues[i] - newX[_measurments.positions[i]]);
				}
				residual = std::sqrt(residual);
				
// 				LOG(best, beta*alpha << " " << residual);
				
	// 			newX.round(_x.ranks(), 0.0);
				if (residual <= bestResidual) {
					_x = newX;
					bestResidual = residual;
					newAlpha = alpha * beta;
				}
			}
			residual = bestResidual;
			alpha = newAlpha;
			LOG(newAlpha, alpha);
			
			_perfData.add(iteration, bestResidual, _x, 0);
		}
		
		return residual;
	}
} // namespace xerus
