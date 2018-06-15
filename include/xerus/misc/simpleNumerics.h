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
 * @brief Header file for several elementary numerical methods: polynomials, romberg integration and limit extractors.
 */

#pragma once

#include "standard.h"

#include <vector>
#include <limits>
#include <functional>

#include <boost/math/tools/polynomial.hpp>

namespace xerus { namespace misc {
	
	/// @brief Performs a Romberg Integration (richardson extrapolation of regular riemannian sum) + adaptive refinement
	double integrate(const std::function<double(double)>& _f, double _a, double _b, double _eps = std::numeric_limits<double>::epsilon(), uint _minIter = 4, uint _maxIter = 6, uint _branchFactor = 7, uint _maxRecursion = 10, bool _relativeError = true);

	double integrate_segmented(const std::function<double(double)> &_f, double _a, double _b, double _segmentation, double _eps = 1e-8, uint _minIter = 4, uint _maxIter = 6, uint _branchFactor = 8, uint _maxRecursion = 10);
	
	double find_root_bisection(const std::function<double(double)> &_f, double _min, double _max, double _epsilon = 1e-14);
	
	template<class T>
	__attribute__((const)) T difference(T _a, T _b) {
		if (_a > _b) {
			return (_a-_b);
		} else {
			return (_b-_a);
		}
	}

	/// @brief Class to represent a polynomial by its vector of coefficients.
	struct Polynomial {
		std::vector<double> coefficients;
		
		Polynomial();
		
		Polynomial(std::vector<double> _coeff);
		
		size_t terms() const;
		
		Polynomial& operator+=(const Polynomial &_rhs);
		
		Polynomial& operator-=(const Polynomial &_rhs);
		
		Polynomial& operator*=(double _rhs);
		
		Polynomial& operator/=(double _rhs);
		
		Polynomial operator*(const Polynomial &_rhs) const;
		
		Polynomial operator*(double _rhs) const;
		
		double operator()(double x) const;
		
		double scalar_product(const Polynomial &_rhs, const std::function<double (double)> &_weight, double _minX, double _maxX) const;
		
		double norm(const std::function<double (double)> &_weight, double _minX, double _maxX) const;
		
		/// orthogonalizes this polynomial with respect to the provided (@note orthogonal!) basis
		Polynomial &orthogonolize(const std::vector<Polynomial> &_orthoBase, const std::function<double (double)> &_weight, double _minX, double _maxX);
		
		/// returns @a _N pairwise orthogonal polynomials w.r.t. a scalar product defined by the @a _weight
		static std::vector<Polynomial> build_orthogonal_base(size_t _N, const std::function<double (double)> &_weight, double _minX, double _maxX);
	};


	/// @brief Classes that can extract an estimate of the limit of a sequence.
	template<class ft_type>
	class LimitExtractor {
	public:
		virtual void push_back(ft_type _val) = 0;
		virtual ft_type best_estimate() const = 0;
		virtual ft_type error_approximate() const = 0;
		virtual void reset() = 0;
		
		virtual ~LimitExtractor() {};
	};

	/** 
	 * @brief Limit extraction using the shanks transformation aka Aitken process.
	 * @details Can be derived by assuming the sequence to go as @f$ x_n = x_inf + alpha * q^n @f$ for large @f$ n @f$.
	 */
	template<class ft_type>
	class ShanksTransformation : public LimitExtractor<ft_type> {
	public:
		static constexpr ft_type epsilon = std::numeric_limits<ft_type>::epsilon();
	public:
		std::vector<ft_type> values;
		
		static ft_type shanks(ft_type x1, ft_type x2, ft_type x3);
		
		void push_back(ft_type _val) override;
		
		ft_type best_estimate() const override;
		
		ft_type error_approximate() const override;
		
		void reset() override;
		
		virtual ~ShanksTransformation() {};
	};


	//TODO the following is crap. implement Levin-t, Levin-u Levin-v
	/** 
	 * @brief Limit extraction using the richardson extrapolation.
	 * @details Can be derived by assuming that @f$ x_inf - x_n = alpha * n^(-1)@f$ .
	 */
	template<class ft_type>
	class RichardsonExtrapolation : public LimitExtractor<ft_type> {
	public:
		static constexpr ft_type epsilon = std::numeric_limits<ft_type>::epsilon();
	public:
		std::vector<ft_type> values;
		
		static ft_type richard(size_t n, ft_type x1, ft_type x2);
		
		void push_back(ft_type _val) override;
		
		ft_type best_estimate() const override;
		
		ft_type error_approximate() const override;
		
		void reset() override;
		
		virtual ~RichardsonExtrapolation() {};
	};

} } // namespaces
