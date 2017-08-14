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
 * @brief Header file for the different measurment classes.
 */

#pragma once

#include <set>
#include <random>
#include <functional>

#include "basic.h"

#include "misc/math.h"
#include "misc/containerSupport.h"

namespace xerus {
	class Tensor;
	class TensorNetwork;
	
	/** 
	* @brief Class used to represent a single point measurments.
	*/
    class SinglePointMeasurementSet {
	public:
		std::vector<std::vector<size_t>> positions;
		std::vector<value_t> measuredValues;
		
		SinglePointMeasurementSet() = default;
		SinglePointMeasurementSet(const SinglePointMeasurementSet&  _other) = default;
		SinglePointMeasurementSet(      SinglePointMeasurementSet&& _other) = default;
		
		SinglePointMeasurementSet& operator=(const SinglePointMeasurementSet&  _other) = default;
		SinglePointMeasurementSet& operator=(      SinglePointMeasurementSet&& _other) = default;
		
		static SinglePointMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions);
		
		static SinglePointMeasurementSet random(const size_t _numMeasurements, const Tensor& _solution);
		
		static SinglePointMeasurementSet random(const size_t _numMeasurements, const TensorNetwork& _solution);
		
		static SinglePointMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<size_t>&)> _callback);
		
		
		size_t size() const;
		
		size_t degree() const;
		
		value_t frob_norm() const;
		
		void add(std::vector<size_t> _position, const value_t _measuredValue);
		
		void sort(const bool _positionsOnly = false);
		
		
		void measure(const Tensor& _solution);
		
		void measure(const TensorNetwork& _solution);
		
		void measure(std::function<value_t(const std::vector<size_t>&)> _callback);
		
		
		double test(const Tensor& _solution) const;
		
		double test(const TensorNetwork& _solution) const;
		
		double test(std::function<value_t(const std::vector<size_t>&)> _callback) const;
		
		
	private:
		void create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions);
	};
	
	
	class RankOneMeasurementSet {
	public:
		std::vector<std::vector<Tensor>> positions;
		std::vector<value_t> measuredValues;
		
		RankOneMeasurementSet() = default;
		RankOneMeasurementSet(const RankOneMeasurementSet&  _other) = default;
		RankOneMeasurementSet(      RankOneMeasurementSet&& _other) = default;
		
		RankOneMeasurementSet(const SinglePointMeasurementSet&  _other, const std::vector<size_t> &_dimensions);

		RankOneMeasurementSet& operator=(const RankOneMeasurementSet&  _other) = default;
		RankOneMeasurementSet& operator=(      RankOneMeasurementSet&& _other) = default;
		
		static RankOneMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions);
		
		static RankOneMeasurementSet random(const size_t _numMeasurements, const Tensor& _solution);
		
		static RankOneMeasurementSet random(const size_t _numMeasurements, const TensorNetwork& _solution);
		
		static RankOneMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<Tensor>&)> _callback);
		
		
		size_t size() const;
		
		size_t degree() const;
		
		value_t frob_norm() const;
		
		void add(const std::vector<Tensor>& _position, const value_t _measuredValue);
		
		void sort(const bool _positionsOnly);
		
		void normalize();
		
		
		void measure(const Tensor& _solution);
		
		void measure(const TensorNetwork& _solution);
		
		void measure(std::function<value_t(const std::vector<Tensor>&)> _callback);
		
		
		double test(const Tensor& _solution) const;
		
		double test(const TensorNetwork& _solution) const;
		
		double test(std::function<value_t(const std::vector<Tensor>&)> _callback) const;
		
		
	private:
		void create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions);
	};
	
	namespace internal {
		int comp(const Tensor& _a, const Tensor& _b);
	}
}
