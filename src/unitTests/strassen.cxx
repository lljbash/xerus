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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;
/*
UNIT_TEST(Strassen, TTRanks,
// 	std::mt19937_64 rnd;
	std::normal_distribution<value_t> dist (0.0, 1.0);
	Index i1,i2,i3,i4,i5,i6,i7,i8;
	
	for (size_t n=2; n<20; ++n) {
		Tensor T({n,n,n,n,n,n});
		for (size_t i=0; i<n; ++i) {
			for (size_t j=0; j<n; ++j) {
				for (size_t k=0; k<n; ++k) {
					T[{i,j,j,k,i,k}] = 1;
				}
			}
		}
		Tensor A = Tensor::random({n,n}, dist);
		Tensor B = Tensor::random({n,n}, dist);
		Tensor C1(2);
		Tensor C2(2);
		C1(i1,i3) = A(i1,i2) * B(i2,i3);
		C2(i5,i6) = T(i1,i2,i3,i4,i5,i6) * A(i1,i2) * B(i3,i4);
		TEST(approx_equal(C1,C2,1e-12));
		
		TTTensor ttT(T);
		ttT.round(1e-12);
		LOG(unit_test, n << " " << ttT.ranks());
		
	}
)


UNIT_TEST(Strassen, CP,
// 	std::mt19937_64 rnd;
	rnd.seed(rd());
	std::uniform_real_distribution<value_t> dist (0.0, 1.0);
	Index i1,i2,i3,i4,i5,i6,i7,i8;
	
	auto cp_approx = [&](Tensor &_A, size_t _r, std::vector<Tensor*> &decomp)->value_t {
		value_t res = _A.frob_norm();
		value_t minres = res;
		Index i,j,k,r1,r2,r3;
		
		
		while (res > 1e-4) {
			TTTensor ttDiff(_A.order());
// 			bool set = false;
			while (decomp.size() >= _r) {
// 				ttDiff = TTTensor(*decomp.front()); //()
// 				ttDiff.round(1);					//()
// 				set = true;
				auto itr = decomp.begin();
				itr += size_t(dist(rnd)*double(decomp.size()));
				delete *itr;
				decomp.erase(itr);
			}
			Tensor diff(_A);
			for (Tensor *c : decomp) {
				diff(i&0) = diff(i&0) - (*c)(i&0);
			}
			
			
			while (dist(rnd)>0.92 && decomp.size()>0) {
				auto itr = decomp.begin();
				itr += size_t(dist(rnd)*double(decomp.size()));
				delete *itr;
				decomp.erase(itr);
			}
// 			if (!set) {
				// do this every time for other variant
				ttDiff = TTTensor(diff);
				for (size_t r=std::max(ttDiff.ranks()); r>0; --r) {
					ttDiff.round(r);
				}
// 			}
			
			if (_A.order() == 3) {
				Tensor &tn0 = *std::static_pointer_cast<Tensor>(ttDiff.nodes[0].tensorObject);
				Tensor &tn1 = *std::static_pointer_cast<Tensor>(ttDiff.nodes[1].tensorObject);
				Tensor &tn2 = *std::static_pointer_cast<Tensor>(ttDiff.nodes[2].tensorObject);
				value_t newMicroRes = frob_norm(diff(i&0) - Tensor(ttDiff)(i&0));
				value_t microItrRes = newMicroRes *2;
// 				LOG(test, "\t\t\t\t" << newMicroRes);
				while (std::abs(1-microItrRes/newMicroRes) > 1e-4) {
					microItrRes = newMicroRes;
					ttDiff.canonicalize_left();
					tn0(i,r1) = diff(i,j,k) * tn1(r1,j,r2) * tn2(r2,k);
					tn0 /= tn0.frob_norm();
					tn1(r1,j,r2) = diff(i,j,k) * tn0(i,r1) * tn2(r2,k);
					tn1 /= tn1.frob_norm();
					tn2(r2,k) = diff(i,j,k) * tn0(i,r1) * tn1(r1,j,r2);
					tn2 /= tn2.frob_norm();
					tn1(r1,j,r2) = diff(i,j,k) * tn0(i,r1) * tn2(r2,k);
					tn1 /= tn1.frob_norm();
					tn0(i,r1) = diff(i,j,k) * tn1(r1,j,r2) * tn2(r2,k);
					newMicroRes = frob_norm(diff(i&0) - Tensor(ttDiff)(i&0));
// 					LOG(test, "\t\t\t\t" << newMicroRes);
					std::cout<< '.';
				}
				std::cout << std::endl;
			} else {
				LOG(wtf, "wtf");
			}
			
			decomp.push_back(new Tensor(ttDiff));
			diff(i&0) = diff(i&0) - (*decomp.back())(i&0);
			value_t newres = diff.frob_norm();
// 			if (newres > res+0.1) break;
			res = newres;
			minres = std::min(minres, res);
			LOG(cp_approx, decomp.size() << " " << res << "\t" << minres);
		}
		return res;
	};
	
	size_t toBeat = 49;
	for (size_t n=4; n<=16; n*=2, toBeat*=7) {
		Tensor T({n*n,n*n,n*n});
		std::vector<Tensor*> decomp;
		for (size_t i=0; i<n; ++i) {
			for (size_t j=0; j<n; ++j) {
				for (size_t k=0; k<n; ++k) {
					decomp.push_back(Tensor::random({n*n,n*n,n*n}, dist));
					T[{i*n+j,j*n+k,i*n+k}] = 1;
// 					(*decomp.back())[{i*n+j,j*n+k,i*n+k}] = 1;
				}
			}
		}
		
		while (true) {
			LOG(unit_test, n << " " << cp_approx(T,toBeat,decomp));
			toBeat-=1;
		}
		
	}
)
*/
