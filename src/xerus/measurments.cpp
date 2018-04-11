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
 * @brief Implementation of the measurment classes class.
 */

#include <xerus/misc/check.h>
#include <xerus/measurments.h>
 
#include <xerus/misc/sort.h>
#include <xerus/misc/random.h>

#include <xerus/index.h>
#include <xerus/tensor.h> 
#include <xerus/tensorNetwork.h>
#include <xerus/ttNetwork.h>
#include <xerus/indexedTensor.h>
#include <xerus/misc/internal.h>
 

namespace xerus {
	// --------------------- SinglePointMeasurementSet -----------------
	
	SinglePointMeasurementSet SinglePointMeasurementSet::random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) {
		SinglePointMeasurementSet result;
		result.create_random_positions(_numMeasurements, _dimensions);
		return result;
	}
	
	
	SinglePointMeasurementSet SinglePointMeasurementSet::random(const size_t _numMeasurements, const Tensor& _solution) {
		SinglePointMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution);
		return result;
	}
	
	SinglePointMeasurementSet SinglePointMeasurementSet::random(const size_t _numMeasurements, const TTTensor& _solution) {
		SinglePointMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution);
		return result;
	}
		
	SinglePointMeasurementSet SinglePointMeasurementSet::random(const size_t _numMeasurements, const TensorNetwork& _solution) {
		SinglePointMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution);
		return result;
	}
	
	SinglePointMeasurementSet SinglePointMeasurementSet::random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<size_t>&)> _callback) {
		SinglePointMeasurementSet result;
		result.create_random_positions(_numMeasurements, _dimensions);
		result.measure(_callback);
		return result;
	}
	
	
	size_t SinglePointMeasurementSet::size() const {
		REQUIRE(positions.size() == measuredValues.size(), "Inconsitend SinglePointMeasurementSet encountered.");
		return positions.size();
	}
	
	
	size_t SinglePointMeasurementSet::degree() const {
		return positions.empty() ? 0 : positions[0].size();
	}
	
	
	void SinglePointMeasurementSet::add(std::vector<size_t> _position, const value_t _measuredValue) {
		REQUIRE(positions.empty() || _position.size() == positions.back().size(), "Given _position has incorrect degree " << _position.size() << ". Expected " << positions.back().size() << ".");
		positions.emplace_back(_position);
		measuredValues.emplace_back(_measuredValue);
	}
	
	
	value_t SinglePointMeasurementSet::frob_norm() const {
		const auto cSize = size();
		double norm = 0.0;
		for(size_t i = 0; i < cSize; ++i) {
			norm += misc::sqr(measuredValues[i]);
		}
		return std::sqrt(norm);
	}
	
	
	void SinglePointMeasurementSet::measure(const Tensor& _solution) {
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			measuredValues[i] = _solution[positions[i]];
		}
	}
	
	void SinglePointMeasurementSet::measure(const TensorNetwork& _solution) {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		std::vector<TensorNetwork> stack(degree()+1);
		stack[0] = _solution;
		stack[0].reduce_representation();
		
		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			size_t rebuildIndex = 0;
			
			if(j > 0) {
				// Find the maximal recyclable stack position
				for(; rebuildIndex < degree(); ++rebuildIndex) {
					if(positions[j-1][rebuildIndex] != positions[j][rebuildIndex]) {
						break;
					}
				}
			}
			
			// Rebuild stack
			for(size_t i = rebuildIndex; i < degree(); ++i) {
				stack[i+1] = stack[i];
				stack[i+1].fix_mode(0, positions[j][i]);
				stack[i+1].reduce_representation();
			}
			
			measuredValues[j] = stack.back()[0];
		}
	}
	
	
	void SinglePointMeasurementSet::measure(std::function<value_t(const std::vector<size_t>&)> _callback) {
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			measuredValues[i] = _callback(positions[i]);
		}
	}
	
	
	double SinglePointMeasurementSet::test(const Tensor& _solution) const {
		const auto cSize = size();
		double error = 0.0, norm = 0.0;
		for(size_t i = 0; i < cSize; ++i) {
			error += misc::sqr(measuredValues[i] - _solution[positions[i]]);
			norm += misc::sqr(measuredValues[i]);
		}
		return std::sqrt(error/norm);
	}
	
	
	double SinglePointMeasurementSet::test(const TensorNetwork& _solution) const {
		const auto cSize = size();
		double error = 0.0, norm = 0.0;
		
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		std::vector<TensorNetwork> stack(degree()+1);
		stack[0] = _solution;
		stack[0].reduce_representation();
		
		for(size_t j = 0; j < cSize; ++j) {
			size_t rebuildIndex = 0;
			
			if(j > 0) {
				// Find the maximal recyclable stack position
				for(; rebuildIndex < degree(); ++rebuildIndex) {
					if(positions[j-1][rebuildIndex] != positions[j][rebuildIndex]) {
						break;
					}
				}
			}
			
			// Rebuild stack
			for(size_t i = rebuildIndex; i < degree(); ++i) {
				stack[i+1] = stack[i];
				stack[i+1].fix_mode(0, positions[j][i]);
				stack[i+1].reduce_representation();
			}
			
			error += misc::sqr(measuredValues[j] - stack.back()[0]);
			norm += misc::sqr(measuredValues[j]);
		}
		return std::sqrt(error/norm);
	}
	
	
	double SinglePointMeasurementSet::test(std::function<value_t(const std::vector<size_t>&)> _callback) const {
		const auto cSize = size();
		double error = 0.0, norm = 0.0;
		for(size_t i = 0; i < cSize; ++i) {
			error += misc::sqr(measuredValues[i] - _callback(positions[i]));
			norm += misc::sqr(measuredValues[i]);
		}
		return std::sqrt(error/norm);
	}
	
	
	struct vec_compare {
		bool operator() (const std::vector<size_t>& _lhs, const std::vector<size_t>& _rhs) const {
			REQUIRE(_lhs.size() == _rhs.size(), "Inconsistent degrees in measurment positions."); 
			for (size_t i = 0; i < _lhs.size(); ++i) {
				if (_lhs[i] < _rhs[i]) { return true; }
				if (_lhs[i] > _rhs[i]) { return false; }
			}
			return false; // equality
		}
	};
	
	
	void SinglePointMeasurementSet::sort(const bool _positionsOnly) {
		const vec_compare comperator;

		if(_positionsOnly) {
			std::sort(positions.begin(), positions.end(), comperator);
		} else {
			REQUIRE(positions.size() == measuredValues.size(), "Inconsitend SinglePointMeasurementSet encountered.");
			misc::simultaneous_sort(positions, measuredValues, comperator);
		}
	}
	
	
	void SinglePointMeasurementSet::create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) {
		XERUS_REQUIRE(misc::product(_dimensions) >= _numMeasurements, "It's impossible to perform as many measurements as requested. " << _numMeasurements << " > " << _dimensions);
		
		// Create distributions
		std::vector<std::uniform_int_distribution<size_t>> indexDist;
		for (size_t i = 0; i < _dimensions.size(); ++i) {
			indexDist.emplace_back(0, _dimensions[i]-1);
		}
		
		std::set<std::vector<size_t>, vec_compare> measuredPositions;
		std::vector<size_t> multIdx(_dimensions.size());
		while (measuredPositions.size() < _numMeasurements) {
			for (size_t i = 0; i < _dimensions.size(); ++i) {
				multIdx[i] = indexDist[i](misc::randomEngine);
			}
			measuredPositions.insert(multIdx);
		}
		
		for(const auto& pos : measuredPositions) {
			positions.push_back(pos);
		}
		
		measuredValues.resize(_numMeasurements, 0.0);
	}
	
	
	
	
	
	
	
	
	// --------------------- RankOneMeasurementSet -----------------
	
	
	RankOneMeasurementSet::RankOneMeasurementSet(const SinglePointMeasurementSet&  _other, const std::vector<size_t>& _dimensions) {
		REQUIRE(_other.degree() == _dimensions.size(), "Inconsistent degrees.");
		std::vector<Tensor> zeroPosition; zeroPosition.reserve(_dimensions.size());
		for(size_t j = 0; j < _dimensions.size(); ++j) {
			zeroPosition.emplace_back(Tensor({_dimensions[j]}));
		}
			
		for(size_t i = 0; i < _other.size(); ++i) {
			add(zeroPosition, _other.measuredValues[i]);
			for(size_t j = 0; j < _other.degree(); ++j) {
				positions.back()[j][_other.positions[i][j]] = 1.0;
			}
		}
	}
	
	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _dimensions);
		return result;
	}
	
	
	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const Tensor& _solution) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution );
		return result;
	}
		
	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const TTTensor& _solution) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution );
		return result;
	}
	
	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const TensorNetwork& _solution) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution );
		return result;
	}
	
	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<Tensor>&)> _callback) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _dimensions);
		result.measure(_callback );
		return result;
	}
	
	
	size_t RankOneMeasurementSet::size() const {
		REQUIRE(positions.size() == measuredValues.size(), "Inconsitend RankOneMeasurementSet encountered.");
		return positions.size();
	}
	
	
	size_t RankOneMeasurementSet::degree() const {
		return positions.empty() ? 0 : positions[0].size();
	}
	
	void RankOneMeasurementSet::add(const std::vector<Tensor>& _position, const value_t _measuredValue) {
		IF_CHECK(
			INTERNAL_CHECK(positions.size() == measuredValues.size(), "Internal Error.");
			if(size() > 0) {
				for(size_t i = 0; i < degree(); ++i) {
					REQUIRE(positions.back()[i].dimensions == _position[i].dimensions, "Inconsitend dimensions obtained.");
				}
			}
			for (const Tensor& t : _position) {
				REQUIRE(t.degree() == 1, "Illegal measurement.");
			}
		);
		
		positions.emplace_back(_position);
		measuredValues.emplace_back(_measuredValue);
	}
	
	void RankOneMeasurementSet::sort(const bool _positionsOnly) {
		const auto comperator = [](const std::vector<Tensor>& _lhs, const std::vector<Tensor>& _rhs) {
			REQUIRE(_lhs.size() == _rhs.size(), "Inconsistent degrees in measurment positions.");
			for (size_t i = 0; i < _lhs.size(); ++i) {
				const auto res = internal::comp(_lhs[i], _rhs[i]);
				if(res == -1) { return true; }
				if(res == 1) { return false; }
			}
			return false; // equality
		};
			
		if(_positionsOnly) {
			std::sort(positions.begin(), positions.end(), comperator);
		} else {
			REQUIRE(positions.size() == measuredValues.size(), "Inconsitend SinglePointMeasurementSet encountered.");
			misc::simultaneous_sort(positions, measuredValues, comperator);
		}
	}
	
	void RankOneMeasurementSet::normalize() {
		for(size_t i = 0; i < size(); ++i) {
			for(size_t j = 0; j < degree(); ++j) {
				const auto norm = positions[i][j].frob_norm();
				positions[i][j] /= norm;
				positions[i][j].apply_factor();
				measuredValues[i] /= norm;
			}
		}
	}
	
	value_t RankOneMeasurementSet::frob_norm() const {
		const auto cSize = size();
		double norm = 0.0;
		for(size_t i = 0; i < cSize; ++i) {
			norm += misc::sqr(measuredValues[i]);
		}
		return std::sqrt(norm);
	}
	
	
	void RankOneMeasurementSet::measure(const Tensor& _solution) {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		std::vector<Tensor> stack(degree()+1);
		stack[0] = _solution;
		
		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			for(size_t i = 0; i < degree(); ++i) {
				contract(stack[i+1], positions[j][i], stack[i], 1);
			}
			
			REQUIRE(stack.back().degree() == 0, "IE");
			measuredValues[j] = stack.back()[0];
		}
	}
	
	void RankOneMeasurementSet::measure(const TTTensor& _solution) {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		std::vector<Tensor> stack(degree()+1);
		stack[0] = Tensor::ones({1});
		
		Tensor tmp;
		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			for(size_t i = 0; i < degree(); ++i) {
				contract(tmp, stack[i], _solution.get_component(i) , 1);
				contract(stack[i+1], positions[j][i], tmp, 1);
			}
			
			stack.back().reinterpret_dimensions({});
			REQUIRE(stack.back().degree() == 0, "IE");
			measuredValues[j] = stack.back()[0];
		}
	}
	
	void RankOneMeasurementSet::measure(const TensorNetwork& _solution) {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		std::vector<TensorNetwork> stack(degree()+1);
		stack[0] = _solution;
		stack[0].reduce_representation();
		
		const Index l, k;
		
		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			
			// Rebuild stack
			for(size_t i = 0; i < degree(); ++i) {
				stack[i+1](k&0) = positions[j][i](l) * stack[i](l, k&1);
				stack[i+1].reduce_representation();
			}
			
			REQUIRE(stack.back().degree() == 0, "IE");
			measuredValues[j] = stack.back()[0];
		}
	}
	
	
	void RankOneMeasurementSet::measure(std::function<value_t(const std::vector<Tensor>&)> _callback) {
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			measuredValues[i] = _callback(positions[i]);
		}
	}
	
	
	double RankOneMeasurementSet::test(const Tensor& _solution) const {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		const auto cSize = size();
		double error = 0.0, norm = 0.0;
		
		std::vector<Tensor> stack(degree()+1);
		stack[0] = _solution;
		
		for(size_t j = 0; j < cSize; ++j) {
			size_t rebuildIndex = 0;
			
			if(j > 0) {
				// Find the maximal recyclable stack position
				for(; rebuildIndex < degree(); ++rebuildIndex) {
					if(!approx_equal(positions[j-1][rebuildIndex], positions[j][rebuildIndex])) {
						break;
					}
				}
			}
			
			// Rebuild stack
			for(size_t i = rebuildIndex; i < degree(); ++i) {
				contract(stack[i+1], positions[j][i], false, stack[i], false, 1);
			}
			
			error += misc::sqr(measuredValues[j] - stack.back()[0]);
			norm += misc::sqr(measuredValues[j]);
		}
		
		return std::sqrt(error/norm);
	}
	
	
	double RankOneMeasurementSet::test(const TensorNetwork& _solution) const {
		REQUIRE(_solution.degree() == degree(), "Degrees of solution and measurements must match!");
		const auto cSize = size();
        const Index l, k;
        
        double error = 0.0, norm = 0.0;
        
        #pragma omp parallel reduction(+:error, norm)
        {
            std::vector<TensorNetwork> stack(degree()+1);
            stack[degree()] = _solution;
            stack[degree()].reduce_representation();
            
            bool init = true;
            #pragma omp for
            for(size_t j = 0; j < cSize; ++j) {
                size_t unchangedModes = 0;
                
                if(!init) {
                    // Find the maximal recyclable stack position
                    for(; unchangedModes < degree(); ++unchangedModes) {
                        if(!approx_equal(positions[j-1][degree()-1-unchangedModes], positions[j][degree()-1-unchangedModes])) { break; }
                    }
                } else { init = false; }
                
                // Rebuild stack
                for(long i = long(degree())-1-long(unchangedModes); i >= 0; --i) {
					stack[size_t(i)](k&0) = stack[size_t(i+1)](k&1, l) * positions[j][size_t(i)](l);
					stack[size_t(i)].reduce_representation();
                }
                
                error += misc::sqr(measuredValues[j] - stack.front()[0]);
                norm += misc::sqr(measuredValues[j]);
            }
		}
		
		return std::sqrt(error/norm);
	}
	
	
	double RankOneMeasurementSet::test(std::function<value_t(const std::vector<Tensor>&)> _callback) const {
		const auto cSize = size();
		double error = 0.0, norm = 0.0;
		for(size_t i = 0; i < cSize; ++i) {
			error += misc::sqr(measuredValues[i] - _callback(positions[i]));
			norm += misc::sqr(measuredValues[i]);
		}
		return std::sqrt(error/norm);
	}
	
	
	
	void RankOneMeasurementSet::create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) {
		using ::xerus::misc::operator<<;
		XERUS_REQUIRE(misc::product(_dimensions) >= _numMeasurements, "It's impossible to perform as many measurements as requested. " << _numMeasurements << " > " << _dimensions);
		
		std::vector<Tensor> randOnePosition(_dimensions.size());
		while (positions.size() < _numMeasurements) {
			for (size_t i = 0; i < _dimensions.size(); ++i) {
				randOnePosition[i] = Tensor::random({_dimensions[i]});
				randOnePosition[i] /= xerus::frob_norm(randOnePosition[i]);
				randOnePosition[i].apply_factor();
			}
			
			// NOTE Assuming our random generator works, no identical positions should occour.
			positions.push_back(randOnePosition);
		}
		
		measuredValues.resize(_numMeasurements, 0);
	}
	
	
	
	namespace internal {
		int comp(const Tensor& _a, const Tensor& _b) {
			REQUIRE(_a.dimensions == _b.dimensions, "Compared Tensors must have the same dimensions.");
			
			if(_a.is_dense() || _b.is_dense()) {
				for(size_t k = 0; k < _a.size; ++k) {
					if (_a.cat(k) < _b.cat(k)) { return 1; }
					if (_a.cat(k) > _b.cat(k)) { return -1; }
				}
				return 0;
			} 
			INTERNAL_CHECK(!_a.has_factor(), "IE");
			INTERNAL_CHECK(!_b.has_factor(), "IE");
			
			const std::map<size_t, double>& dataA = _a.get_unsanitized_sparse_data();
			const std::map<size_t, double>& dataB = _b.get_unsanitized_sparse_data();
			
			auto itrA = dataA.begin();
			auto itrB = dataB.begin();
			
			while(itrA != dataA.end() && itrB != dataB.end()) {
				if(itrA->first == itrB->first) {
					if(itrA->second < itrB->second) {
						return 1;
					} 
					if(itrA->second > itrB->second) {
						return -1;
					}
					++itrA; ++itrB;
				} else if(itrA->first < itrB->first) {
					if(itrA->second < 0.0) {
						return 1;
					} 
					if(itrA->second > 0.0) {
						return -1;
					}
					++itrA;
				} else { // itrA->first > itrB->first
					if(0.0 < itrB->second) {
						return 1;
					} 
					if(0.0 > itrB->second) {
						return -1;
					}
					++itrB;
				}
			}
			
			while(itrA != dataA.end()) {
				if(itrA->second < 0.0) {
					return 1;
				} 
				if(itrA->second > 0.0) {
					return -1;
				}
				++itrA;
			}
			
			while(itrB != dataB.end()) {
				if(0.0 < itrB->second) {
					return 1;
				} 
				if(0.0 > itrB->second) {
					return -1;
				}
				++itrB;
			}
			
			return 0;
		}
	} // namespace internal
	
} // namespace xerus
