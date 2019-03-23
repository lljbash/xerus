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
* @brief Implementation of the measurment classes class.
*/

#include <xerus/measurments.h>

#include <xerus/misc/check.h>
#include <xerus/misc/math.h>
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/sort.h>
#include <xerus/misc/random.h>


#include <xerus/index.h>
#include <xerus/tensor.h>
#include <xerus/tensorNetwork.h>
#include <xerus/ttNetwork.h>
#include <xerus/indexedTensor.h>
#include <xerus/misc/internal.h>


namespace xerus {
	double relative_weighted_l2_difference(const std::vector<value_t>& _reference, const std::vector<value_t>& _test, const std::vector<value_t>& _weights) {
		const size_t cSize = _reference.size();
		double error = 0.0, norm = 0.0;
		if(_weights.size() == 0) {
			for(size_t i = 0; i < cSize; ++i) {
				error += misc::sqr(_reference[i] - _test[i]);
				norm += misc::sqr(_reference[i]);
			}
		} else {
			for(size_t i = 0; i < cSize; ++i) {
				error += _weights[i]*misc::sqr(_reference[i] - _test[i]);
				norm += _weights[i]*misc::sqr(_reference[i]);
			}
		}
		return std::sqrt(error/norm);
	}
	
	
	// --------------------- MeasurementSet -----------------
	
	template<class PositionType>
	size_t MeasurementSet<PositionType>::size() const {
		REQUIRE(positions.size() == measuredValues.size(), "Inconsitend MeasurementSet encountered.");
		REQUIRE(weights.size() == 0  || weights.size() == measuredValues.size(), "Inconsitend MeasurementSet encountered.");
		return positions.size();
	}


	template<class PositionType>
	size_t MeasurementSet<PositionType>::order() const {
		XERUS_REQUIRE_TEST;
		return positions.empty() ? 0 : positions.back().size();
	}
	
	
	template<class PositionType>
	value_t MeasurementSet<PositionType>::norm_2() const {
		XERUS_REQUIRE_TEST;
		
		const auto cSize = size();
		double norm = 0.0;
		if(weights.size() == 0) {
			for(size_t i = 0; i < cSize; ++i) {
				norm += misc::sqr(measuredValues[i]);
			}
		} else {
			for(size_t i = 0; i < cSize; ++i) {
				norm += weights[i]*misc::sqr(measuredValues[i]);
			}
		}
		return std::sqrt(norm);
	}
	
	
	template<class PositionType>
	void MeasurementSet<PositionType>::clear() {
		positions.clear();
		measuredValues.clear();
		weights.clear();
	}
	
	
	template<class PositionType>
	void MeasurementSet<PositionType>::add(const std::vector<PositionType>& _position, const value_t _measuredValue) {
		XERUS_REQUIRE_TEST;
		check_position(_position);
		positions.emplace_back(_position);
		measuredValues.emplace_back(_measuredValue);
	}
	

	template<class PositionType>
	void MeasurementSet<PositionType>::add(const std::vector<PositionType>& _position, const value_t _measuredValue, const value_t _weight) {
		XERUS_REQUIRE_TEST;
		check_position(_position);
		positions.emplace_back(_position);
		measuredValues.emplace_back(_measuredValue);
		weights.emplace_back(_weight);
	}
	
	
	template<class PositionType>
	void MeasurementSet<PositionType>::sort() {
		const auto comperator = [](const std::vector<PositionType>& _lhs, const std::vector<PositionType>& _rhs) {
			REQUIRE(_lhs.size() == _rhs.size(), "Inconsistent orders in measurment positions.");
			for (size_t i = 0; i < _lhs.size(); ++i) {
				const auto res = internal::compare(_lhs[i], _rhs[i]);
				if(res == -1) { return true; }
				if(res == 1) { return false; }
			}
			return false; // equality
		};
		
		const std::vector<size_t> permutation = misc::create_sort_permutation(positions, comperator);
		
		misc::apply_permutation(positions, permutation);
		
		if(weights.size() != 0) {
			REQUIRE(positions.size() == weights.size(), "Inconsistent sizes for positions and weights");
			misc::apply_permutation(weights, permutation);
		}
		
		REQUIRE(positions.size() == measuredValues.size(), "Inconsistent sizes for positions and measuredValues");
		misc::apply_permutation(measuredValues, permutation);
		
	}

	
	template<class PositionType>
	void MeasurementSet<PositionType>::add_noise(const double _epsilon) {
		XERUS_REQUIRE_TEST;
		const auto cSize = size();
		const auto noiseTensor = Tensor::random({size()});
		const double norm = xerus::frob_norm(noiseTensor);

		for(size_t i = 0; i < cSize; ++i) {
			measuredValues[i] += (_epsilon/norm)*noiseTensor[i];
		}
	}

	
	template<class PositionType>
	void MeasurementSet<PositionType>::measure(const Tensor& _solution) {
		XERUS_REQUIRE_TEST;
		get_values(measuredValues, _solution);
	}
	
	
	template<class PositionType>
	void MeasurementSet<PositionType>::measure(const TensorNetwork& _solution) {
		XERUS_REQUIRE_TEST;
		get_values(measuredValues, _solution);
	}
	
	
	template<class PositionType>
	void MeasurementSet<PositionType>::measure(std::function<value_t(const std::vector<PositionType>&)> _callback) {
		XERUS_REQUIRE_TEST;
		get_values(measuredValues, _callback);
	}
	

	template<class PositionType>
	double MeasurementSet<PositionType>::test(const Tensor& _solution) const {
		XERUS_REQUIRE_TEST;
		std::vector<value_t> testValues(size());
		get_values(testValues, _solution);
		return relative_weighted_l2_difference(measuredValues, testValues, weights);
	}

	
	template<class PositionType>
	double MeasurementSet<PositionType>::test(const TensorNetwork& _solution) const {
		XERUS_REQUIRE_TEST;
		std::vector<value_t> testValues(size());
		get_values(testValues, _solution);
		return relative_weighted_l2_difference(measuredValues, testValues, weights);
	}

	
	template<class PositionType>
	double MeasurementSet<PositionType>::test(std::function<value_t(const std::vector<PositionType>&)> _callback) const {
		XERUS_REQUIRE_TEST;
		std::vector<value_t> testValues(size());
		get_values(testValues, _callback);
		return relative_weighted_l2_difference(measuredValues, testValues, weights);
	}
	
	// Explicit instantiation of the two template parameters that will be implemented in the xerus library
	template class MeasurementSet<size_t>;
	template class MeasurementSet<Tensor>;
	
	
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

	
	void SinglePointMeasurementSet::check_consitency() {
		IF_CHECK(
			REQUIRE(positions.size() == measuredValues.size(), "Inconsistent amount of positions/values.");
			REQUIRE(weights.size() == 0 || positions.size() == weights.size(), "Inconsistent amount of positions/weights.");
			const auto ord = order();
			for(size_t i = 0; i < positions.size(); ++i) {
				REQUIRE(positions[i].size() == ord, "Inconsistent orders.");
			}
		);
	}
	
	
	struct vec_compare final {
		bool operator() (const std::vector<size_t>& _lhs, const std::vector<size_t>& _rhs) const {
			REQUIRE(_lhs.size() == _rhs.size(), "Inconsistent orders in measurment positions.");
			for (size_t i = 0; i < _lhs.size(); ++i) {
				if (_lhs[i] < _rhs[i]) { return true; }
				if (_lhs[i] > _rhs[i]) { return false; }
			}
			return false; // equality
		}
	};
	
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
	
	
	void SinglePointMeasurementSet::check_position(const std::vector<size_t>& _position) {
		REQUIRE(positions.empty() || _position.size() == positions.back().size(), "Given _position has incorrect order " << _position.size() << ". Expected " << positions.back().size() << ".");
	}
	
	
	void SinglePointMeasurementSet::get_values(std::vector<value_t>& _values, const Tensor& _solution) const {
		XERUS_REQUIRE_TEST;
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			_values[i] = _solution[positions[i]];
		}
	}

	
	void SinglePointMeasurementSet::get_values(std::vector<value_t>& _values, const TensorNetwork& _solution) const {
		REQUIRE(_solution.order() == order(), "Degrees of solution and measurements must match!");
		std::vector<TensorNetwork> stack(order()+1);
		stack[0] = _solution;
		stack[0].reduce_representation();

		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			size_t rebuildIndex = 0;

			if(j > 0) {
				// Find the maximal recyclable stack position
				for(; rebuildIndex < order(); ++rebuildIndex) {
					if(positions[j-1][rebuildIndex] != positions[j][rebuildIndex]) {
						break;
					}
				}
			}

			// Rebuild stack
			for(size_t i = rebuildIndex; i < order(); ++i) {
				stack[i+1] = stack[i];
				stack[i+1].fix_mode(0, positions[j][i]);
				stack[i+1].reduce_representation();
			}

			_values[j] = stack.back()[0];
		}
	}
	

	void SinglePointMeasurementSet::get_values(std::vector<value_t>& _values, std::function<value_t(const std::vector<size_t>&)> _callback) const {
		XERUS_REQUIRE_TEST;
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			_values[i] = _callback(positions[i]);
		}
	}





	// --------------------- RankOneMeasurementSet -----------------

	RankOneMeasurementSet::RankOneMeasurementSet(const SinglePointMeasurementSet&  _other, const std::vector<size_t>& _dimensions) {
		REQUIRE(_other.order() == _dimensions.size(), "Inconsistent orders.");
		std::vector<Tensor> zeroPosition; zeroPosition.reserve(_dimensions.size());
		for(size_t j = 0; j < _dimensions.size(); ++j) {
			zeroPosition.emplace_back(Tensor({_dimensions[j]}));
		}

		for(size_t i = 0; i < _other.size(); ++i) {
			add(zeroPosition, _other.measuredValues[i]);
			for(size_t j = 0; j < _other.order(); ++j) {
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
		result.measure(_solution);
		return result;
	}


	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const TensorNetwork& _solution) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _solution.dimensions);
		result.measure(_solution);
		return result;
	}
	

	RankOneMeasurementSet RankOneMeasurementSet::random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<Tensor>&)> _callback) {
		RankOneMeasurementSet result;
		result.create_random_positions(_numMeasurements, _dimensions);
		result.measure(_callback);
		return result;
	}



	void RankOneMeasurementSet::normalize() {
		XERUS_REQUIRE_TEST;
		for(size_t i = 0; i < size(); ++i) {
			for(size_t j = 0; j < order(); ++j) {
				const auto norm = positions[i][j].frob_norm();
				positions[i][j] /= norm;
				positions[i][j].apply_factor();
				measuredValues[i] /= norm;
			}
		}
	}
	
	void RankOneMeasurementSet::check_consitency() {
		IF_CHECK(
			REQUIRE(positions.size() == measuredValues.size(), "Inconsistent amount of positions/values.");
			REQUIRE(weights.size() == 0 || positions.size() == weights.size(), "Inconsistent amount of positions/weights.");
			const auto ord = order();
			for(size_t i = 0; i < positions.size(); ++i) {
				REQUIRE(positions[i].size() == ord, "Inconsistent orders.");
				for(size_t k = 0; k < ord; ++k) {
					REQUIRE(positions.back()[k].dimensions == positions[i][k].dimensions, "Inconsistent dimensions obtained.");
				}
			}
			for(size_t k = 0; k < ord; ++k) {
				REQUIRE(positions.back()[k].order() == 1, "Illegal measurement.");
			}
		);
	}


	void RankOneMeasurementSet::create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) {
		XERUS_REQUIRE_TEST;

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

		measuredValues.resize(_numMeasurements, 0.0);
	}
	
	void RankOneMeasurementSet::check_position(const std::vector<Tensor>& _position) {
		IF_CHECK(
			INTERNAL_CHECK(positions.size() == measuredValues.size(), "Internal Error.");
			if(size() > 0) {
				for(size_t i = 0; i < order(); ++i) {
					REQUIRE(positions.back()[i].dimensions == _position[i].dimensions, "Inconsitend dimensions obtained.");
				}
			}
			for (const Tensor& t : _position) {
				REQUIRE(t.order() == 1, "Illegal measurement.");
			}
		);
	}
	
	
	void RankOneMeasurementSet::get_values(std::vector<value_t>& _values, const Tensor& _solution) const {
		REQUIRE(_solution.order() == order(), "Degrees of solution and measurements must match!");
		std::vector<Tensor> stack(order()+1);
		stack[0] = _solution;

		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {
			for(size_t i = 0; i < order(); ++i) {
				contract(stack[i+1], positions[j][i], stack[i], 1);
			}

			REQUIRE(stack.back().order() == 0, "IE");
			_values[j] = stack.back()[0];
		}
	}


//     void RankOneMeasurementSet::get_values(std::vector<value_t>& _values, const TTTensor& _solution) const {
//         REQUIRE(_solution.order() == order(), "Degrees of solution and measurements must match!");
//         std::vector<Tensor> stack(order()+1);
//         stack[0] = Tensor::ones({1});
// 
//         Tensor tmp;
//         const auto cSize = size();
//         for(size_t j = 0; j < cSize; ++j) {
//             for(size_t i = 0; i < order(); ++i) {
//                 contract(tmp, stack[i], _solution.get_component(i) , 1);
//                 contract(stack[i+1], positions[j][i], tmp, 1);
//             }
// 
//             stack.back().reinterpret_dimensions({});
//             REQUIRE(stack.back().order() == 0, "IE");
//             _values[j] = stack.back()[0];
//         }
//     }


	void RankOneMeasurementSet::get_values(std::vector<value_t>& _values, const TensorNetwork& _solution) const {
		REQUIRE(_solution.order() == order(), "Degrees of solution and measurements must match!");
		std::vector<TensorNetwork> stack(order()+1);
		stack[0] = _solution;
		stack[0].reduce_representation();

		const Index l, k;

		const auto cSize = size();
		for(size_t j = 0; j < cSize; ++j) {

			// Rebuild stack
			for(size_t i = 0; i < order(); ++i) {
				stack[i+1](k&0) = positions[j][i](l) * stack[i](l, k&1);
				stack[i+1].reduce_representation();
			}

			REQUIRE(stack.back().order() == 0, "IE");
			_values[j] = stack.back()[0];
		}
	}


	void RankOneMeasurementSet::get_values(std::vector<value_t>& _values, std::function<value_t(const std::vector<Tensor>&)> _callback) const {
		XERUS_REQUIRE_TEST;
		const auto cSize = size();
		for(size_t i = 0; i < cSize; ++i) {
			_values[i] = _callback(positions[i]);
		}
	}

	
	


	namespace internal {
		int compare(const size_t _a, const size_t _b) {
			if (_a < _b) { return 1; }
			if (_a > _b) { return -1; }
			return 0;
		}
		
		int compare(const Tensor& _a, const Tensor& _b) {
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
