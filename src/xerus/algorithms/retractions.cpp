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
 * @brief Implementation of the retractions declared in retractions.h.
 */

#include <xerus.h>
#include <xerus/misc/internal.h>

namespace xerus {

	void HOSVDRetraction::operator()(TTTensor &_U, const TTTensor &_change) const {
		Index i;
		_U(i&0) = _U(i&0) + _change(i&0);
		if (roundByVector) {
			_U.round(rankVector);
		} else {
			_U.round(rank);
		}
	}
	
	void HOSVDRetraction::operator()(TTTensor &_U, const TTTangentVector &_change) const {
		Index i;
		_U = _change.added_to_base();
		if (roundByVector) {
			_U.round(rankVector);
		} else {
			_U.round(rank);
		}
	}
	
	void HOSVDRetractionII(TTTensor &_U, const TTTensor &_change) {
		std::vector<size_t> oldRank = _U.ranks();
		_U = _U + _change;
		_U.round(oldRank);
	}
	
	void HOSVDRetractionI(TTTensor &_U, const TTTangentVector &_change) {
		std::vector<size_t> oldRank = _U.ranks();
		_U = _change.added_to_base();
		_U.round(oldRank);
	}
	
	void ALSRetractionII(TTTensor &_U, const TTTensor &_change) {
		static const ALSVariant roundingALS(1, 2, ALSVariant::lapack_solver, false);
		TTTensor target = _U + _change;
		roundingALS(_U, target);
// 		_U.move_core(0);
	}
	
	void ALSRetractionI(TTTensor &_U, const TTTangentVector &_change) {
		static const ALSVariant roundingALS(1, 2, ALSVariant::lapack_solver, false);
		TTTensor target = _change.added_to_base();
		roundingALS(_U, target);
// 		_U.move_core(0);
	}
	
	void TTTangentVector::set_base(const TTTensor &_newBase) {
		REQUIRE(_newBase.dimensions == baseL.dimensions, "");
		baseL = _newBase;
		baseL.move_core(0, true);
	}
		
	TTTangentVector::TTTangentVector(const TTTensor& _base, const TTTensor& _direction) {
		REQUIRE(_base.canonicalized && _base.corePosition == 0, "projection onto tangent plane is only implemented for core position 0 at the moment");
		REQUIRE(_base.dimensions == _direction.dimensions, "");
		
		baseL = _base;
		baseL.move_core(0, true);
		
		Index i1,i2,j1,j2,r,s, s2;
		std::vector<Tensor> leftStackUV;
		std::vector<Tensor> leftStackUU;
		Tensor tmp({1,1}, [](){return 1.0;});
		leftStackUV.push_back(tmp);
		leftStackUU.push_back(tmp);
		for (size_t i=0; i<baseL.order()-1; ++i) {
			Tensor newLeft;
			newLeft(j1,j2) = leftStackUV.back()(i1,i2) * baseL.get_component(i)(i1,r,j1) * _direction.get_component(i)(i2,r,j2);
			leftStackUV.emplace_back(std::move(newLeft));
			newLeft(j1,j2) = leftStackUU.back()(i1,i2) * baseL.get_component(i)(i1,r,j1) * baseL.get_component(i)(i2,r,j2);
			leftStackUU.emplace_back(std::move(newLeft));
		}
		Tensor right(tmp);
		Tensor UTV;
		std::vector<Tensor> tmpComponents;
		for (size_t i=baseL.order(); i>0; --i) {
			const size_t currIdx = i-1;
			std::unique_ptr<Tensor> newComponent(new Tensor);
			const Tensor &UComp = baseL.get_component(currIdx);
			Tensor V;
			Tensor uuInv = pseudo_inverse(leftStackUU.back(), 1);
			V(i1,r,j1) =  uuInv(i1,s)* leftStackUV.back()(s,i2) * _direction.get_component(currIdx)(i2,r,j2) * right(j1,j2);
// 			if (i!=baseL.order()) {
// 				V(i1,r,j1) = V(i1,r,j1) + UComp(i1,r,s)*UTV(s,j1);
// 			}
			if (currIdx!=0) {
				UTV(i1,i2) = V(i1,r,j1) * UComp(i2,r,j1);
				V(i1,r,j1) = V(i1,r,j1) - UTV(i1,s) * UComp(s,r,j1);
			}
			tmpComponents.emplace_back(std::move(V));
			if (currIdx != 0) {
				right(j1,j2) = UComp(j1,r,i1) * _direction.get_component(currIdx)(j2,r,i2) * right(i1,i2);
			}
			leftStackUV.pop_back();
			leftStackUU.pop_back();
		}
		while (!tmpComponents.empty()) {
			components.emplace_back(std::move(tmpComponents.back()));
			tmpComponents.pop_back();
		}
	}
	
	TTTangentVector& TTTangentVector::operator+=(const TTTangentVector& _rhs) {
		REQUIRE(components.size() == _rhs.components.size(), "");
		for (size_t i=0; i<components.size(); ++i) {
			components[i] += _rhs.components[i];
		}
		return *this;
	}
	
	TTTangentVector& TTTangentVector::operator-=(const TTTangentVector& _rhs) {
		REQUIRE(components.size() == _rhs.components.size(), "");
		for (size_t i=0; i<components.size(); ++i) {
			components[i] -= _rhs.components[i];
		}
		return *this;
	}
	
	TTTangentVector& TTTangentVector::operator*=(value_t _alpha) {
		for (auto &c : components) {
			c *= _alpha;
		}
		return *this;
	}
	
	TTTangentVector TTTangentVector::operator*(value_t _alpha) const {
		TTTangentVector result(*this);
		result *= _alpha;
		return result;
	}
	
	TTTangentVector operator*(value_t _alpha, const TTTangentVector &_rhs) {
		TTTangentVector result(_rhs);
		result *= _alpha;
		return result;
	}
	
	value_t TTTangentVector::scalar_product(const TTTangentVector& _other) const {
		REQUIRE(components.size() == _other.components.size(), "");
		Index i1,i2,r,j1,j2;
		value_t result = 0;
		Tensor left({1,1}, [](){return 1.0;});
		for (size_t i=0; i<components.size(); ++i) {
			result += value_t(left(i1,i2)*components[i](i1,r,j1)*_other.components[i](i2,r,j1));
			if (i < components.size()-1) {
				left(j1,j2) = left(i1,i2) * baseL.get_component(i)(i1,r,j1) * baseL.get_component(i)(i2,r,j2);
			}
		}
		return result;
	}
	
	value_t TTTangentVector::frob_norm() const {
		return std::sqrt(scalar_product(*this));
	}

	
	TTTensor TTTangentVector::change_direction_incomplete() const {
		TTTensor result(baseL.order());
		Index i1,i2,n,r1,r2;
		for (size_t i=1; i<components.size(); ++i) {
			const Tensor &baseComponent = baseL.get_component(i);
			REQUIRE(baseComponent.dimensions == components[i].dimensions, "illegal base tensor for this tangent vector (ranks or external dimension do not coincide)");
			if (i < components.size()-1) {
				Tensor newComponent;
				newComponent(i1,r1,n,i2,r2) = Tensor::dirac({2,2},{0,0})(i1,i2) * baseComponent(r1,n,r2)
											+Tensor::dirac({2,2},{0,1})(i1,i2) * components[i](r1,n,r2)
											+Tensor::dirac({2,2},{1,1})(i1,i2) * baseComponent(r1,n,r2);
				newComponent.reinterpret_dimensions({components[i].dimensions[0]*2, components[i].dimensions[1], components[i].dimensions[2]*2});
				result.set_component(i, newComponent);
			} else {
				Tensor newComponent;
				newComponent(i1,r1,n,r2) = Tensor::dirac({2},0)(i1) * components[i](r1,n,r2)
											+Tensor::dirac({2},1)(i1) * baseComponent(r1,n,r2);
				newComponent.reinterpret_dimensions({components[i].dimensions[0]*2, components[i].dimensions[1], 1});
				result.set_component(i, newComponent);
			}
		}
		return result;
	}
	
	TTTangentVector::operator TTTensor() const {
		if (components.size() == 1) {
			TTTensor result(1);
			result.set_component(0, components[0]);
			return result;
		}
		TTTensor result = change_direction_incomplete();
		Index i1,i2,n,r1,r2;
		Tensor newComponent;
		newComponent(r1,n,i2,r2) = Tensor::dirac({2},0)(i2) * baseL.get_component(0)(r1,n,r2)
									+Tensor::dirac({2},1)(i2) * components[0](r1,n,r2);
		newComponent.reinterpret_dimensions({1, components[0].dimensions[1], components[0].dimensions[2]*2});
		result.set_component(0, newComponent);
		result.move_core(0);
		return result;
// 		TTTensor result(baseL);
// 		result.set_component(0, components[0]);
// 		for (size_t i=1; i<components.size(); ++i) {
// 			TTTensor tmp(baseL);
// 			tmp.set_component(i, components[i]);
// 			result += tmp;
// 		}
// 		result.move_core(0);
// 		return result;
	}
	
	TTTensor TTTangentVector::added_to_base() const {
		if (components.size() == 1) {
			TTTensor result(1);
			result.set_component(0, components[0]);
			result = result + baseL;
			return result;
		}
		TTTensor result = change_direction_incomplete();
		Tensor newComponent;
		Index i1,i2,n,r1,r2;
		newComponent(r1,n,i2,r2) = Tensor::dirac({2},0)(i2) * baseL.get_component(0)(r1,n,r2)
									+Tensor::dirac({2},1)(i2) * (baseL.get_component(0)(r1,n,r2) + components[0](r1,n,r2));
		newComponent.reinterpret_dimensions({1, components[0].dimensions[1], components[0].dimensions[2]*2});
		result.set_component(0, newComponent);
		result.move_core(0);
		return result;
// 		TTTensor result(baseL);
// 		for (size_t i=0; i<components.size(); ++i) {
// 			TTTensor tmp(baseL);
// 			tmp.set_component(i, components[i]);
// 			result += tmp;
// 		}
// 		result.move_core(0);
// 		return result;
	}
	
	
	
	void SubmanifoldRetractionII(TTTensor &_U, const TTTensor &_change) {
		TTTangentVector W(_U, _change);
		SubmanifoldRetractionI(_U, W);
	}
	
	void SubmanifoldRetractionI(TTTensor &_U, const TTTangentVector &_change) {
		static const Index i1,j1,r;
		for (size_t i=0; i<_U.order(); ++i) { XERUS_REQUIRE_TEST;
			std::unique_ptr<Tensor> newComponent(new Tensor);
			(*newComponent)(i1,r,j1) = _U.get_component(i)(i1,r,j1) + _change.components[i](i1,r,j1);
			_U.set_component(i, std::move(*newComponent));
		}
		_U.move_core(0, true);
	}
	
	
	
	
	// TODO do this without creating the change_direction tensor?
	void ProjectiveVectorTransport(const TTTensor &_newBase, TTTangentVector &_tangentVector) {
		REQUIRE(_newBase.canonicalized && _newBase.corePosition == 0, "Tangent vectors only implemented for core position 0 atm");
		
		_tangentVector = TTTangentVector(_newBase, TTTensor(_tangentVector));
	}
} // namespace xerus
