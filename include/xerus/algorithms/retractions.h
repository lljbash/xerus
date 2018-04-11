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
 * @brief Header file for several retractions of type I and II as well as simple vector transport.
 */

#pragma once

#include <vector>
#include "../tensor.h"

namespace xerus {

	///@brief class to compactly represent tangent vectors of the manifold of constant TT-rank
	class TTTangentVector {
	public:
		TTTensor baseL;
		
		///@note components will not be changed. use a vector transport to update them accordingly instead of calling this function
		void set_base(const TTTensor &_newBase);
		
		std::vector<Tensor> components;
		TTTangentVector() {};
		///@brief creates a tangent vector by projecting @a _direction onto the tangent plane located at @a _base
		TTTangentVector(const TTTensor &_base, const TTTensor &_direction);
		TTTangentVector &operator+=(const TTTangentVector &_rhs);
		TTTangentVector &operator-=(const TTTangentVector &_rhs);
		TTTangentVector &operator*=(value_t _alpha);
		TTTangentVector operator*(value_t _alpha) const;
		value_t scalar_product(const TTTangentVector &_other) const;
		value_t frob_norm() const;
	private:
		TTTensor change_direction_incomplete() const;
	public:
		explicit operator TTTensor() const;
		TTTensor added_to_base() const;
	};
	
	TTTangentVector operator*(value_t _alpha, const TTTangentVector &_rhs);
	
	/// retraction that performs a HOSVD to project back onto the Manifold
	struct HOSVDRetraction {
		bool roundByVector;
		size_t rank;
		std::vector<size_t> rankVector;
		void operator()(TTTensor &_U, const TTTensor &_change) const;
		void operator()(TTTensor &_U, const TTTangentVector &_change) const;
		HOSVDRetraction(size_t _rank) : roundByVector(false), rank(_rank) {}
		HOSVDRetraction(const std::vector<size_t> &_rank) : roundByVector(true), rank(~0ul), rankVector(_rank) {}
	};
	
	using TTRetractionI = std::function<void(TTTensor &, const TTTangentVector &)>;
	using TTRetractionII = std::function<void(TTTensor &, const TTTensor &)>;
	using TTVectorTransport = std::function<void(const TTTensor &, TTTangentVector &)>;
	
	/// retraction that performs a HOSVD to project back onto the Manifold
	void HOSVDRetractionI(TTTensor &_U, const TTTangentVector &_change);
	void HOSVDRetractionII(TTTensor &_U, const TTTensor &_change);
	
	/// retraction that performs an ALS half-sweep to project back onto the Manifold. Automatically retains the ranks of @a _U
	void ALSRetractionII(TTTensor &_U, const TTTensor &_change);
	
	/// retraction that performs an ALS half-sweep to project back onto the Manifold. Automatically retains the ranks of @a _U
	void ALSRetractionI(TTTensor &_U, const TTTangentVector &_change);
	
	/// retraction that performs componentwise addition of @f$ U_i @f$ and @f$ W_i @f$ where @f$ W_i @f$ is the i-th component of the riemannian tangential vector representation
	void SubmanifoldRetractionII(TTTensor &_U, const TTTensor &_change);
	
	/// retraction that performs componentwise addition of @f$ U_i @f$ and @f$ W_i @f$ where @f$ W_i @f$ is the i-th component of the riemannian tangential vector representation
	void SubmanifoldRetractionI(TTTensor &_U, const TTTangentVector &_change);
	
	
	/// simple vector transport by projecting onto the new tangent plane
	void ProjectiveVectorTransport(const TTTensor &_newBase, TTTangentVector &_tangentVector);
	
}

