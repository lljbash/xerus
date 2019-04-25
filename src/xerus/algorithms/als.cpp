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
* @brief Implementation of the ALS variants.
*/

#include <xerus/misc/math.h>

#include <xerus/algorithms/als.h>
#include <xerus/basic.h>
#include <xerus/index.h>
#include <xerus/indexedTensorList.h>
#include <xerus/tensorNetwork.h>
#include <xerus/misc/internal.h>

#include <xerus/indexedTensorMoveable.h>
#include <xerus/indexedTensor_tensor_factorisations.h>

namespace xerus {

	// -------------------------------------------------------------------------------------------------------------------------
	//                                       local solvers
	// -------------------------------------------------------------------------------------------------------------------------
	
	void ALSVariant::lapack_solver(const TensorNetwork& _A, std::vector<Tensor>& _x, const TensorNetwork& _b, const ALSAlgorithmicData& _data) {
		Tensor A(_A);
		Tensor b(_b);
		Tensor x;
		
		xerus::solve(x, A, b);
		
		Index i,j,k,l;
		if (_data.direction == Increasing) {
			for (size_t p = 0; p+1 < _data.ALS.sites; ++p) {
				Tensor U, S;
// 				calculate_svd(U, S, x, x, 2, _data.targetRank[_data.currIndex+p], EPSILON); TODO
				(U(i^2,j), S(j,k), x(k,l&1)) = SVD(x(i^2,l&2), _data.targetRank[_data.currIndex+p]);
				_x[p] = std::move(U);
				x(j,l&1) = S(j,k) * x(k,l&1);
			}
			_x.back() = std::move(x);
		} else {
			// direction: decreasing index
			for (size_t p = _data.ALS.sites-1; p>0; --p) {
				Tensor S, Vt;
// 				calculate_svd(x, S, Vt, x, x.order()-1, _data.targetRank[_data.currIndex+p-1], EPSILON); TODO
				(x(i&1,j), S(j,k), Vt(k,l&1)) = SVD(x(i&2,l^2), _data.targetRank[_data.currIndex+p-1]);
				_x[p] = std::move(Vt);
				x(i&1,k) = x(i&1,j) * S(j,k);
			}
			_x[0] = std::move(x);
		}
	}
	
	void ALSVariant::ASD_solver(const TensorNetwork &_A, std::vector<Tensor> &_x, const TensorNetwork &_b, const ALSAlgorithmicData &_data) {
		// performs a single gradient step, so
		// x = x + alpha * P( A^t (b - Ax) )    or for SPD: x = x + alpha * P( b - Ax )
		// where the projection P is already part of the stacks 
		// and alpha is the exact stepsize for quadratic functionals
		REQUIRE(_data.ALS.sites == 1, "ASD only defined for single site alternation at the moment");
		Tensor grad;
		Index i,j,k;
		grad(i&0) = _b(i&0) - _A(i/2,j/2) * _x[0](j&0);
		value_t alpha;
		if (_data.ALS.assumeSPD) {
			// stepsize alpha = <y,y>/<y,Ay>
			alpha = misc::sqr(frob_norm(grad)) / value_t(grad(i&0) * _A(i/2,j/2) * grad(j&0));
		} else {
			grad(i&0) = _A(j/2,i/2) * grad(j&0);
			// stepsize alpha = <y,y>/<Ay,Ay> = 1/|Ay/|y||^2
			alpha = 1 / misc::sqr(frob_norm(_A(i/2,j/2) * (grad/frob_norm(grad))(j&0)));
		}
		_x[0] += alpha * grad;
	}
	
	// -------------------------------------------------------------------------------------------------------------------------
	//                                       helper functions
	// -------------------------------------------------------------------------------------------------------------------------
	
	/**
	 * @brief Finds the range of notes that need to be optimized and orthogonalizes @a _x properly
	 * @details finds full-rank nodes (these can wlog be set to identity and need not be optimized)
	 * requires canonicalizeAtTheEnd and corePosAtTheEnd to be set
	 * sets optimizedRange
	 * modifies x
	 */
	void ALSVariant::ALSAlgorithmicData::prepare_x_for_als() {
		const size_t d = x.order();
		Index r1,r2,n1,cr1;
		
		size_t firstOptimizedIndex = 0;
		size_t dimensionProd = 1;
		while (firstOptimizedIndex + 1 < d) {
			const size_t localDim = x.dimensions[firstOptimizedIndex];
			size_t newDimensionProd = dimensionProd * localDim;
			if (x.rank(firstOptimizedIndex) < newDimensionProd) {
				break;
			}
			
			Tensor& curComponent = x.component(firstOptimizedIndex);
			curComponent.reinterpret_dimensions({curComponent.dimensions[0]*curComponent.dimensions[1], curComponent.dimensions[2]});
			curComponent(r1,n1,r2) = curComponent(r1,cr1) * x.get_component(firstOptimizedIndex+1)(cr1,n1,r2);
			x.set_component(firstOptimizedIndex+1, std::move(curComponent));
			
			//TODO sparse
			x.set_component(firstOptimizedIndex, Tensor(
				{dimensionProd, localDim, newDimensionProd},
				[&](const std::vector<size_t> &_idx){
					if (_idx[0]*localDim + _idx[1] == _idx[2]) {
						return 1.0;
					} 
					return 0.0;
				})
			);
			
			x.require_correct_format();
			
			firstOptimizedIndex += 1;
			dimensionProd = newDimensionProd;
		}
		
		size_t firstNotOptimizedIndex = d;
		dimensionProd = 1;
		while (firstNotOptimizedIndex > firstOptimizedIndex+ALS.sites) {
			const size_t localDim = x.dimensions[firstNotOptimizedIndex-1];
			size_t newDimensionProd = dimensionProd * localDim;
			if (x.rank(firstNotOptimizedIndex-2) < newDimensionProd) {
				break;
			}
			
			Tensor& curComponent = x.component(firstNotOptimizedIndex-1);
			curComponent.reinterpret_dimensions({curComponent.dimensions[0], curComponent.dimensions[1] * curComponent.dimensions[2]});
			curComponent(r1,n1,r2) = x.get_component(firstNotOptimizedIndex-2)(r1,n1,cr1) * curComponent(cr1,r2);
			x.set_component(firstNotOptimizedIndex-2, std::move(curComponent));
			
			//TODO sparse
			x.set_component(firstNotOptimizedIndex-1, Tensor(
				{newDimensionProd, localDim, dimensionProd},
				[&](const std::vector<size_t> &_idx){
					if (_idx[0] == _idx[1]*dimensionProd + _idx[2]) {
						return 1.0;
					} 
					return 0.0;
				})
			);

			x.require_correct_format();
			
			firstNotOptimizedIndex -= 1;
			dimensionProd = newDimensionProd;
		}
		
		if (canonicalizeAtTheEnd && corePosAtTheEnd < firstOptimizedIndex) {
			x.assume_core_position(firstOptimizedIndex);
		} else {
			if (canonicalizeAtTheEnd && corePosAtTheEnd >= firstNotOptimizedIndex) {
				x.assume_core_position(firstNotOptimizedIndex-1);
			}
			
			x.move_core(firstOptimizedIndex, true);
		}
		
		optimizedRange = std::pair<size_t, size_t>(firstOptimizedIndex, firstNotOptimizedIndex);
	}

	TensorNetwork ALSVariant::ALSAlgorithmicData::localOperatorSlice(size_t _pos) {
		//TODO optimization: create these networks without indices
		Index cr1, cr2, cr3, cr4, r1, r2, r3, r4, n1, n2, n3;
		TensorNetwork res;
		INTERNAL_CHECK(A, "ie");
		if (ALS.assumeSPD) {
			res(r1,r2,r3, cr1,cr2,cr3) = x.get_component(_pos)(r1, n1, cr1) 
										* A->get_component(_pos)(r2, n1, n2, cr2) 
										* x.get_component(_pos)(r3, n2, cr3);
		} else {
			res(r1,r2,r3,r4, cr1,cr2,cr3, cr4) = x.get_component(_pos)(r1, n1, cr1) 
										* A->get_component(_pos)(r2, n2, n1, cr2) 
										* A->get_component(_pos)(r3, n2, n3, cr3) 
										* x.get_component(_pos)(r4, n3, cr4);
		}
		return res;
	}
	
	TensorNetwork ALSVariant::ALSAlgorithmicData::localRhsSlice(size_t _pos) {
		//TODO optimization: create these networks without indices
		Index cr1, cr2, cr3, r1, r2, r3, n1, n2;
		TensorNetwork res;
		if (ALS.assumeSPD || (A == nullptr)) {
			res(r1,r2, cr1,cr2) = b.get_component(_pos)(r1, n1, cr1) 
									* x.get_component(_pos)(r2, n1, cr2);
		} else {
			res(r1,r2,r3, cr1,cr2,cr3) = b.get_component(_pos)(r1, n1, cr1) 
												* A->get_component(_pos)(r2, n1, n2, cr2) 
												* x.get_component(_pos)(r3, n2, cr3);
		}
		return res;
	}
	
	void ALSVariant::ALSAlgorithmicData::prepare_stacks() {
		const size_t d = x.order();
		Index r1,r2;
		
		Tensor tmpA;
		Tensor tmpB;
		if (ALS.assumeSPD || (A == nullptr)) {
			tmpA = Tensor::ones({1,1,1});
			tmpB = Tensor::ones({1,1});
		} else {
			tmpA = Tensor::ones({1,1,1,1});
			tmpB = Tensor::ones({1,1,1});
		}
		
		
		localOperatorCache.left.emplace_back(tmpA);
		localOperatorCache.right.emplace_back(tmpA);
		rhsCache.left.emplace_back(tmpB);
		rhsCache.right.emplace_back(tmpB);
		
		for (size_t i = d-1; i > optimizedRange.first + ALS.sites - 1; --i) {
			if (A != nullptr) {
				tmpA(r1&0) = localOperatorCache.right.back()(r2&0) * localOperatorSlice(i)(r1/2, r2/2);
				localOperatorCache.right.emplace_back(tmpA);
			}
			tmpB(r1&0) = rhsCache.right.back()(r2&0) * localRhsSlice(i)(r1/2, r2/2);
			rhsCache.right.emplace_back(tmpB);
		}
		for (size_t i = 0; i < optimizedRange.first; ++i) {
			if (A != nullptr) {
				tmpA(r2&0) = localOperatorCache.left.back()(r1&0) * localOperatorSlice(i)(r1/2, r2/2);
				localOperatorCache.left.emplace_back((tmpA));
			}
			tmpB(r2&0) = rhsCache.left.back()(r1&0) * localRhsSlice(i)(r1/2, r2/2);
			rhsCache.left.emplace_back(tmpB);
		}
	}
	
	void ALSVariant::ALSAlgorithmicData::choose_energy_functional() {
		if (A != nullptr) {
			if (ALS.assumeSPD) {
				residual_f = [&](){
					Index n1, n2;
					return frob_norm((*A)(n1/2,n2/2)*x(n2&0) - b(n1&0))/normB;
				};
				if (ALS.useResidualForEndCriterion) {
					energy_f = residual_f;
				} else {
					energy_f = [&](){
						Index r1,r2;
						Tensor res;
						// 0.5*<x,Ax> - <x,b>
						TensorNetwork xAx = localOperatorCache.left.back();
						TensorNetwork bx = rhsCache.left.back();
						for (size_t i=0; i<ALS.sites; ++i) {
							xAx(r2&0) = xAx(r1&0) * localOperatorSlice(currIndex+i)(r1/2, r2/2);
							bx(r2&0) = bx(r1&0) * localRhsSlice(currIndex+i)(r1/2, r2/2);
						}
						res() = 0.5*xAx(r1&0) * localOperatorCache.right.back()(r1&0)
								- bx(r1&0) * rhsCache.right.back()(r1&0);
						return res.frob_norm();
					};
				}
			} else {
				// not Symmetric pos def
				residual_f = [&](){
					Index r1,r2;
					Tensor res;
					// <Ax,Ax> - 2 * <Ax,b> + <b,b>
					TensorNetwork xAtAx = localOperatorCache.left.back();
					TensorNetwork bAx = rhsCache.left.back();
					for (size_t i=0; i<ALS.sites; ++i) {
						xAtAx(r2&0) = xAtAx(r1&0) * localOperatorSlice(currIndex+i)(r1/2, r2/2);
						bAx(r2&0) = bAx(r1&0) * localRhsSlice(currIndex+i)(r1/2, r2/2);
					}
					res() = xAtAx(r1&0) * localOperatorCache.right.back()(r1&0)
							- 2 * bAx(r1&0) * rhsCache.right.back()(r1&0);
					return std::sqrt(res[0] + misc::sqr(normB))/normB;
				};
				energy_f = residual_f;
			}
		} else {
			// no operator A given
			residual_f = [&](){
				return frob_norm(x - b);
			};
			if (ALS.useResidualForEndCriterion) {
				energy_f = residual_f;
			} else {
				energy_f = [&](){
					Index r1,r2;
					Tensor res;
					// 0.5*<x,Ax> - <x,b> = 0.5*|x_i|^2 - <x,b>
					TensorNetwork bx = rhsCache.left.back();
					for (size_t i=0; i<ALS.sites; ++i) {
						bx(r2&0) = bx(r1&0) * localRhsSlice(currIndex+i)(r1/2, r2/2);
					}
					res() = 0.5*x.get_component(currIndex)(r1&0) * x.get_component(currIndex)(r1&0) 
							- bx(r1&0) * rhsCache.right.back()(r1&0);
					return res[0];
				};
			}
		}
	}
	
	ALSVariant::ALSAlgorithmicData::ALSAlgorithmicData(const ALSVariant &_ALS, const TTOperator *_A, TTTensor &_x, const TTTensor &_b) 
		: ALS(_ALS), A(_A), x(_x), b(_b)
		, targetRank(_x.ranks())
		, normB(frob_norm(_b))
		, canonicalizeAtTheEnd(_x.canonicalized)
		, corePosAtTheEnd(_x.corePosition)
		, lastEnergy2(1e102)
		, lastEnergy(1e101)
		, energy(1e100)
		, halfSweepCount(0)
		, direction(Increasing)
	{
		prepare_x_for_als();
		prepare_stacks();
		currIndex = optimizedRange.first;
		choose_energy_functional();
	}

	void ALSVariant::ALSAlgorithmicData::move_to_next_index() {
		Index r1,r2;
		Tensor tmpA, tmpB;
		if (direction == Increasing) {
			INTERNAL_CHECK(currIndex+ALS.sites < optimizedRange.second, "ie " << currIndex << " " << ALS.sites << " " << optimizedRange.first << " " << optimizedRange.second);
			// Move core to next position (assumed to be done by the solver if sites > 1)
			if (ALS.sites == 1) {
				x.move_core(currIndex+1, true);
			}
			
			// Move one site to the right
			if (A != nullptr) {
				localOperatorCache.right.pop_back();
				tmpA(r2&0) = localOperatorCache.left.back()(r1&0) * localOperatorSlice(currIndex)(r1/2, r2/2);
				localOperatorCache.left.emplace_back(std::move(tmpA));
			}
			
			rhsCache.right.pop_back();
			tmpB(r2&0) = rhsCache.left.back()(r1&0) * localRhsSlice(currIndex)(r1/2, r2/2);
			rhsCache.left.emplace_back(std::move(tmpB));
			currIndex++;
		} else {
			INTERNAL_CHECK(currIndex > optimizedRange.first, "ie");
			// Move core to next position (assumed to be done by the solver if sites > 1)
			if (ALS.sites == 1) {
				x.move_core(currIndex-1, true);
			}
			
			// move one site to the left
			if (A != nullptr) {
				localOperatorCache.left.pop_back();
				tmpA(r1&0) = localOperatorCache.right.back()(r2&0) * localOperatorSlice(currIndex)(r1/2, r2/2);
				localOperatorCache.right.emplace_back(std::move(tmpA));
			}
			
			rhsCache.left.pop_back();
			tmpB(r1&0) = rhsCache.right.back()(r2&0) * localRhsSlice(currIndex)(r1/2, r2/2);
			rhsCache.right.emplace_back(std::move(tmpB));
			currIndex--;
		}
	}
	
	
	TensorNetwork ALSVariant::construct_local_operator(ALSVariant::ALSAlgorithmicData& _data) const {
		INTERNAL_CHECK(_data.A, "IE");
		Index cr1, cr2, cr3, cr4, r1, r2, r3, r4, n1, n2, n3, n4, x;
		TensorNetwork ATilde = _data.localOperatorCache.left.back();
		if (assumeSPD) {
			for (size_t p=0; p<sites; ++p) {
				ATilde(n1^(p+1), n2, r2, n3^(p+1), n4) = ATilde(n1^(p+1), r1, n3^(p+1)) * _data.A->get_component(_data.currIndex+p)(r1, n2, n4, r2);
			}
			ATilde(n1^(sites+1), n2, n3^(sites+1), n4) = ATilde(n1^(sites+1), r1, n3^(sites+1)) * _data.localOperatorCache.right.back()(n2, r1, n4);
		} else {
			for (size_t p=0;  p<sites; ++p) {
				ATilde(n1^(p+1),n2, r3,r4, n3^(p+1),n4) = ATilde(n1^(p+1), r1,r2, n3^(p+1))
						* _data.A->get_component(_data.currIndex+p)(r1, x, n2, r3)
						* _data.A->get_component(_data.currIndex+p)(r2, x, n4, r4);
			}
			ATilde(n1^(sites+1), n2, n3^(sites+1), n4) = ATilde(n1^(sites+1), r1^2, n3^(sites+1)) * _data.localOperatorCache.right.back()(n2, r1^2, n4);
		}
		return ATilde;
	}
	
	
	TensorNetwork ALSVariant::construct_local_RHS(ALSVariant::ALSAlgorithmicData& _data) const {
		Index cr1, cr2, cr3, cr4, r1, r2, r3, r4, n1, n2, n3, n4, x;
		TensorNetwork BTilde;
		if (assumeSPD || (_data.A == nullptr)) {
			BTilde(n1,r1) = _data.rhsCache.left.back()(r1,n1);
			for (size_t p=0; p<sites; ++p) {
				BTilde(n1^(p+1), n2, cr1) = BTilde(n1^(p+1), r1) * _data.b.get_component(_data.currIndex+p)(r1, n2, cr1);
			}
			BTilde(n1^(sites+1),n2) = BTilde(n1^(sites+1), r1) * _data.rhsCache.right.back()(r1,n2);
		} else {
			BTilde(n1,r1^2) = _data.rhsCache.left.back()(r1^2,n1);
			for (size_t p=0; p<sites; ++p) {
				BTilde(n1^(p+1), n3, cr1, cr2) = BTilde(n1^(p+1), r1, r2) 
					* _data.b.get_component(_data.currIndex+p)(r1, n2, cr1)
					* _data.A->get_component(_data.currIndex+p)(r2, n2, n3, cr2);
			}
			BTilde(n1^(sites+1),n2) = BTilde(n1^(sites+1), r1^2) * _data.rhsCache.right.back()(r1^2,n2);
		}
		return BTilde;
	}


	bool ALSVariant::check_for_end_of_sweep(ALSAlgorithmicData& _data, size_t _numHalfSweeps, value_t _convergenceEpsilon, PerformanceData &_perfData) const {
		if ((_data.direction == Decreasing && _data.currIndex==_data.optimizedRange.first) 
			|| (_data.direction == Increasing && _data.currIndex==_data.optimizedRange.second-sites)) 
		{
			LOG(ALS, "Sweep Done");
			_data.halfSweepCount += 1;
			
			_data.lastEnergy2 = _data.lastEnergy;
			_data.lastEnergy = _data.energy;
			_data.energy = _data.energy_f();
			
			if (_perfData) {
				size_t flags = _data.direction == Increasing ? FLAG_FINISHED_HALFSWEEP : FLAG_FINISHED_FULLSWEEP;
				if (!useResidualForEndCriterion) {
					_perfData.stop_timer();
					value_t residual = _data.residual_f();
					_perfData.continue_timer();
					_perfData.add(residual, _data.x, flags);
				} else {
					_perfData.add(_data.energy, _data.x, flags);
				}
			}
			
			// Conditions for loop termination
			if (_data.halfSweepCount == _numHalfSweeps 
				|| std::abs(_data.lastEnergy-_data.energy) < _convergenceEpsilon 
				|| std::abs(_data.lastEnergy2-_data.energy) < _convergenceEpsilon 
				|| (_data.optimizedRange.second - _data.optimizedRange.first<=sites)) 
			{
				// we are done! yay
				LOG(ALS, "ALS done, " << _data.energy << " " << _data.lastEnergy << " " << std::abs(_data.lastEnergy2-_data.energy) << " " << std::abs(_data.lastEnergy-_data.energy) << " < " << _convergenceEpsilon);
				if (_data.canonicalizeAtTheEnd && preserveCorePosition) {
					_data.x.move_core(_data.corePosAtTheEnd, true);
				}
				return true;
			}
				
			// change walk direction
			_data.direction = _data.direction == Increasing ? Decreasing : Increasing;
		} else {
			// we are not done with the sweep, just calculate data for the perfdata
			if (_perfData) {
				_perfData.stop_timer();
				value_t residual = _data.residual_f();
				_perfData.continue_timer();
				_perfData.add(residual, _data.x, 0);
			}
		}
		return false;
	}
	
	
	// -------------------------------------------------------------------------------------------------------------------------
	//                                   the actual algorithm
	// -------------------------------------------------------------------------------------------------------------------------
	
	
	double ALSVariant::solve(const TTOperator *_Ap, TTTensor &_x, const TTTensor &_b, size_t _numHalfSweeps, value_t _convergenceEpsilon, PerformanceData &_perfData) const {
		LOG(ALS, "ALS("<< sites <<") called");
		
		IF_CHECK(
			REQUIRE(&_x != &_b, "Cannot use the same TTTensor as _x and _b. Make a copy of _x and use it as _b?");
			REQUIRE(_x.order() > 0, "The ALS implementation requires x to have order larger then 0.");
			REQUIRE(_x.dimensions == _b.dimensions, "The ALS implementation requires x and b to have the same dimensions."); //TODO why? 
			_x.require_correct_format();
			_b.require_correct_format();
			
			if (_Ap != nullptr) {
				_Ap->require_correct_format();
				REQUIRE(_Ap->dimensions.size() == _b.dimensions.size()*2, "Inconsistend order of A and x/b");
				for (size_t i=0; i<_x.dimensions.size(); ++i) {
					REQUIRE(_Ap->dimensions[i] == _x.dimensions[i], "Inconsistend local dimensions between of A and x/b");
					REQUIRE(_Ap->dimensions[i+_Ap->order()/2] == _x.dimensions[i], "Inconsistend local dimensions between of A and x/b");
				}
			}
		);
		
		if (_Ap != nullptr) {
			_perfData << "ALS for ||A*x - b||^2, x.dimensions: " << _x.dimensions << '\n'
					<< "A.ranks: " << _Ap->ranks() << '\n'
					<< "x.ranks: " << _x.ranks() << '\n'
					<< "b.ranks: " << _b.ranks() << '\n'
					<< "maximum number of half sweeps: " << _numHalfSweeps << '\n'
					<< "convergence epsilon: " << _convergenceEpsilon << '\n';
		} else {
			_perfData << "ALS for ||x - b||^2, x.dimensions: " << _x.dimensions << '\n'
					<< "x.ranks: " << _x.ranks() << '\n'
					<< "b.ranks: " << _b.ranks() << '\n'
					<< "maximum number of half sweeps: " << _numHalfSweeps << '\n'
					<< "convergence epsilon: " << _convergenceEpsilon << '\n';
		}
		_perfData.start();
		
		ALSAlgorithmicData data(*this, _Ap, _x, _b);
		
		data.energy = data.energy_f();
		
		if (_perfData) {
			_perfData.stop_timer();
			value_t residual = data.residual_f();
			_perfData.continue_timer();
			_perfData.add(residual, _x, FLAG_FINISHED_FULLSWEEP);
		}
		
		while (true) {
			LOG(ALS, "Starting to optimize index " << data.currIndex);
			
			// update current component tensor
			if (_Ap != nullptr) {
				std::vector<Tensor> tmpX;
				for (size_t p=0; p<sites; ++p) {
					tmpX.emplace_back(_x.get_component(data.currIndex+p));
				}
				localSolver(construct_local_operator(data), tmpX, construct_local_RHS(data), data);
				for (size_t p=0; p<sites; ++p) {
					_x.set_component(data.currIndex+p, std::move(tmpX[p]));
				}
			} else {
				//TODO?
				REQUIRE(sites==1, "approximation dmrg not implemented yet");
				_x.component(data.currIndex) = Tensor(construct_local_RHS(data));
			}
			
			if(check_for_end_of_sweep(data, _numHalfSweeps, _convergenceEpsilon, _perfData)) {
				return data.energy; // TODO residual?
			}
			
			data.move_to_next_index();
		}
	}
	
	
	const ALSVariant ALS(1, 0, ALSVariant::lapack_solver, false);
	const ALSVariant ALS_SPD(1, 0, ALSVariant::lapack_solver, true);
	
	const ALSVariant DMRG(2, 0, ALSVariant::lapack_solver, false);
	const ALSVariant DMRG_SPD(2, 0, ALSVariant::lapack_solver, true);
	
	const ALSVariant ASD(1, 0, ALSVariant::ASD_solver, false);
	const ALSVariant ASD_SPD(1, 0, ALSVariant::ASD_solver, true);
} // namespace xerus
