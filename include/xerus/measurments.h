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
* @brief Header file for the different measurment classes.
*/

#pragma once

#include <functional>

#include "basic.h"
#include "forwardDeclarations.h"



namespace xerus {
	template<class PositionType> class MeasurementSet {
	public:
		///@brief Vector containing the positions of the measurements.
		std::vector<std::vector<PositionType>> positions;
		
		///@brief Vector containing the measured values.
		std::vector<value_t> measuredValues;
		
		///@brief If empty no weights are considered, otherwise a vector containign the weights for the measurements.
		std::vector<value_t> weights;
		
		
		MeasurementSet() = default;
		MeasurementSet(const MeasurementSet&  _other) = default;
		MeasurementSet(      MeasurementSet&& _other) = default;

		MeasurementSet& operator=(const MeasurementSet&  _other) = default;
		MeasurementSet& operator=(      MeasurementSet&& _other) = default;
		
		virtual ~MeasurementSet() = default;
		
		
		///@brief Returns the number of measuremts.
		size_t size() const;

		///@brief Returns the order of the tensor that is measured.
		size_t order() const;
		
		///@brief Returns the 2-norm of the measurements, with weights considered if appropriate.
		value_t norm_2() const;
		
		///@brief Removes all measurements from the set.
		void clear();

		///@brief Add a measurment at @a _position with @a _value to the set.
		void add(const std::vector<PositionType>& _position, const value_t _measuredValue);
		
		///@brief Add a measurment at @a _position with @a _value and @a _weight to the set.
		void add(const std::vector<PositionType>& _position, const value_t _measuredValue, const value_t _weight);
		
		///@brief Sort the measurements.
		void sort();

		///@brief Add noise with relative 2-norm @a _epsilon to the measurements.
		void add_noise(const double _epsilon);
		
		///@brief Set the measuredValues equal to the ones measured from @a _solution. 
		void measure(const Tensor& _solution);

		///@brief Set the measuredValues equal to the ones measured from @a _solution.
		void measure(const TensorNetwork& _solution);

		///@brief Set the measuredValues equal to the ones given by @a _callback.
		void measure(std::function<value_t(const std::vector<PositionType>&)> _callback);
		

		///@brief Returns the relative 2-norm difference between the measuredValues and the ones measured from @a _solution.
		double test(const Tensor& _solution) const;

		///@brief Returns the relative 2-norm difference between the measuredValues and the ones measured from @a _solution.
		double test(const TensorNetwork& _solution) const;

		///@brief Returns the relative 2-norm difference between the measuredValues and the ones given by @a _callback.
		double test(std::function<value_t(const std::vector<PositionType>&)> _callback) const;
		
		///@brief Verifies that the MeasurementSet is in a consitent state.
		virtual void check_consitency() = 0;
		
	protected:
		virtual void create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) = 0;
		
		virtual void check_position(const std::vector<PositionType>& _position) = 0;
		
		virtual void get_values(std::vector<value_t>& _values, const Tensor& _solution) const = 0;

		virtual void get_values(std::vector<value_t>& _values, const TensorNetwork& _solution) const = 0;

		virtual void get_values(std::vector<value_t>& _values, std::function<value_t(const std::vector<PositionType>&)> _callback) const = 0;
		
	};
	
	/**
	* @brief Class used to represent a set of single point measurements.
	*/
	class SinglePointMeasurementSet : public MeasurementSet<size_t> {
	public:
		SinglePointMeasurementSet() = default;
		SinglePointMeasurementSet(const SinglePointMeasurementSet&  _other) = default;
		SinglePointMeasurementSet(      SinglePointMeasurementSet&& _other) = default;

		SinglePointMeasurementSet& operator=(const SinglePointMeasurementSet&  _other) = default;
		SinglePointMeasurementSet& operator=(      SinglePointMeasurementSet&& _other) = default;

		static SinglePointMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions);

		static SinglePointMeasurementSet random(const size_t _numMeasurements, const Tensor& _solution);

		static SinglePointMeasurementSet random(const size_t _numMeasurements, const TensorNetwork& _solution);

		static SinglePointMeasurementSet random(const size_t _numMeasurements, const std::vector<size_t>& _dimensions, std::function<value_t(const std::vector<size_t>&)> _callback);

		virtual void check_consitency() override;
		
	protected:
		virtual void create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) override;
		
		virtual void check_position(const std::vector<size_t>& _position) override;
		
		virtual void get_values(std::vector<value_t>& _values, const Tensor& _solution) const override;

		virtual void get_values(std::vector<value_t>& _values, const TensorNetwork& _solution) const override;

		virtual void get_values(std::vector<value_t>& _values, std::function<value_t(const std::vector<size_t>&)> _callback) const override;
	};

	/**
	* @brief Class used to represent a set of rank-one measurements.
	*/
	class RankOneMeasurementSet : public MeasurementSet<Tensor> {
	public:
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

		///@brief Ensures that each measurment vector has 2-norm of one.
		void normalize();
		
		virtual void check_consitency() override;
		
	protected:
		virtual void create_random_positions(const size_t _numMeasurements, const std::vector<size_t>& _dimensions) override;
		
		virtual void check_position(const std::vector<Tensor>& _position) override;
		
		virtual void get_values(std::vector<value_t>& _values, const Tensor& _solution) const override;

		virtual void get_values(std::vector<value_t>& _values, const TensorNetwork& _solution) const override;

		virtual void get_values(std::vector<value_t>& _values, std::function<value_t(const std::vector<Tensor>&)> _callback) const override;
	};

	namespace internal {
		int compare(const size_t _a, const size_t _b);
		int compare(const Tensor& _a, const Tensor& _b);
	}
}
