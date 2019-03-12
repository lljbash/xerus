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
 * @brief Header file for utilities for uncertainity quantification.
 */

#pragma once

#include "../ttNetwork.h"

namespace xerus {namespace uq {
	
	enum class PolynomBasis : bool { Hermite, Legendre };
    
	///@brief Returns a vector (as tensor), containing the evaluations at value @a _value of the first @a _basisSize polynomials of the specified basis.
	Tensor polynomial_basis_evaluation(const double _value, const PolynomBasis _polyBasis, const size_t _basisSize);
	
	
	///@brief Returns a vector (as tensor), containing the evaluations at value @a _value of the first @a _basisSize hermite polynomials.
    Tensor hermite_evaluation(const double _value, const size_t _basisSize);
	
	
	///@brief Returns a vector (as tensor), containing the evaluations at value @a _value of the first @a _basisSize legendre polynomials.
	Tensor legendre_evaluation(const double _value, const size_t _basisSize);
	

	///@brief Returns the first three stochastical moments of the solution @a _x. TODO Currently broken for m2 and m3.
	std::tuple<Tensor, Tensor, Tensor> det_moments(const TTTensor& _x, const PolynomBasis _polyBasis);


    ///@brief Approximates the first three stochastical moments of the solution @a _x using a monte carlo simulation with @a _N samples.
	std::tuple<Tensor, Tensor, Tensor> mc_moments(const TTTensor& _x, const PolynomBasis _polyBasis, const size_t _N);


	///@brief Calculates mean, variance and skewness using the first three stochastical moments.
	std::tuple<Tensor, Tensor, Tensor> mean_var_skew(const std::tuple<Tensor, Tensor, Tensor>& _moments);
	
	
	///@brief Evaluates the solution for the specified parameters
	Tensor evaluate(const TTTensor& _x, const std::vector<double>& _parameters, const PolynomBasis _basisType);
	
	///@brief Calculates the mean of all given samples.
	Tensor sample_mean(const std::vector<Tensor>& _samples);


	class UQMeasurementSet {
	public:
		std::vector<std::vector<double>> parameterVectors;
		std::vector<Tensor> solutions;
		
		UQMeasurementSet() = default;
		UQMeasurementSet(const UQMeasurementSet&  _other) = default;
		UQMeasurementSet(      UQMeasurementSet&& _other) = default;
		
		void add(const std::vector<double>& _paramVec, const Tensor& _solution);
		
		size_t size() const;
		
		void clear();
	};

	
	/**
	 * @brief: Calculates an inital guess for the UQ solution using the mean an linear distortions.
	 * @param _mean An approximation for the mean (expectation value) of the solution, e.g. by calculating the mean of all samples (see sample_mean).
	 * @param _parameterVectors The parameterVectors for measurements corresponding to linear distortions, i.e. only one parameter should be non-zero (and small) for each measurement.
	 * @param _solutions The solutions corresponding to the @a _parameterVectors
	 * @param _polyBasis The PolynomBasis to use for the UQ formulation.
	 * @param _dimensions The dimensions of the solutions tensor.
	 */
	TTTensor initial_guess(const Tensor& _mean, const std::vector<std::vector<double>>& _parameterVectors, const std::vector<Tensor>& _solutions, const PolynomBasis _polyBasis, const std::vector<size_t>& _dimensions);
}}


