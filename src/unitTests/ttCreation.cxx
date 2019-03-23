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
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static misc::UnitTest tt_crea("TT", "TTTensor_Creation", [](){
    Index i,j,k;
    
    Tensor A1 = Tensor::random({2});
    TTTensor TTA1(A1, 1e-14);
    Tensor B1;
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    Tensor A2 = Tensor::random({2,2});
    TTTensor TTA2(A2, 1e-14);
    Tensor B2;
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    Tensor A3 = Tensor::random({2,7});
    TTTensor TTA3(A3, 1e-14);
    Tensor B3;
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    Tensor A4 = Tensor::random({2,2,2,2,2,2,2,2});
    TTTensor TTA4(A4, 1e-14);
    Tensor B4;
    B4(j,i^7) = TTA4(j,i^7);
    TEST(approx_equal(B4,A4, 1e-14));
    
    Tensor A5 = Tensor::random({7,5,3,1,4,2,8,1});
    TTTensor TTA5(A5, 1e-14);
    Tensor B5;
    B5(j,i^7) = TTA5(j,i&1);
    TEST(approx_equal(B5,A5, 1e-14));
});


static misc::UnitTest tt_opcrea("TT", "TTOperator_Creation", [](){
    Index i,j,k;
    
    Tensor A1 = Tensor::random({2,2});
    TTOperator TTA1(A1, 1e-14);
    Tensor B1;
    B1(i^2) = TTA1(i^2);
    TEST(approx_equal(B1,A1, 1e-14));
    
    Tensor A2 = Tensor::random({2,7});
    TTOperator TTA2(A2, 1e-14);
    Tensor B2;
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    Tensor A3 = Tensor::random({2,7,3,1});
    TTOperator TTA3(A3, 1e-14);
    Tensor B3;
    B3(j,i^3) = TTA3(j,i^3);
    TEST(approx_equal(B3,A3, 1e-14));
    
    Tensor A4 = Tensor::random({2,2,2,2,2,2,2,2});
    TTOperator TTA4(A4, 1e-14);
    Tensor B4;
    B4(j,i^7) = TTA4(j,i&1);
    TEST(approx_equal(B4, A4, 1e-14));
    
    Tensor A5 = Tensor::random({7,5,6,3,1,4,2,1}); 
    TTOperator TTA5(A5, 1e-14);
    Tensor B5;
    B5(j,i^7) = TTA5(j,i^7);
    TEST(approx_equal(B5,A5, 1e-14));
});


static misc::UnitTest tt_crea_eps("TT", "creation_with_epsilon", [](){
	const value_t EPS = 0.03;
	Tensor A = Tensor::random({5,5,5,5});
	TTTensor ttA(A, EPS); 
	TTTensor ttB(A, 0);
	ttB.round(EPS);
	
	size_t numDecrease = 5-ttA.rank(0) + 25 - ttA.rank(1) + 5 - ttA.rank(2);
	Index i;
	
	TEST(frob_norm(A(i&0)-Tensor(ttA)(i&0))/frob_norm(A) < static_cast<double>(numDecrease)*EPS);
	TEST(ttA.ranks()[1] < 25);
	TEST(frob_norm(A(i&0)-Tensor(ttB)(i&0))/frob_norm(A) < static_cast<double>(numDecrease)*EPS);
	TEST(ttB.ranks()[1] < 25);
});

static misc::UnitTest tt_crea_full5("TT", "creation_from_fullTensor_5x5x5x5", [](){
	Tensor A = Tensor::random({5,5,5,5});
	TTTensor ttA(A);
	Tensor B(ttA);
	
	Index i;
	
	TEST(approx_equal(A,B, 1e-14));
	TEST(frob_norm(A(i&0)-B(i&0)) < 1e-13*5*5*5*5);
});


static misc::UnitTest tt_namedconstr("TT", "named_constructors", [](){
	std::vector<size_t> dimensions(10, 2);
	std::vector<size_t> operatorDimensions(20,2);
	std::vector<size_t> ranks(9, 4);
	ranks[4] = 1;
	
	TTTensor X = TTTensor::random(dimensions, ranks);
	std::vector<size_t> found_ranks = X.ranks();
	X.move_core(X.order()-1);
	X.move_core(0);
	MTEST(X.ranks() == found_ranks, X.ranks() << " vs " << found_ranks);
// 	{
// 		value_t mean = 0;
// 		Tensor tmp(X);
// 		for (size_t i=0; i<tmp.size; ++i) {
// 			mean += tmp[i];
// 		}
// 		mean /= (double)tmp.size;
// 		MTEST(std::abs(mean) < 0.5, "X mean " << mean);
// 	}
	
	TTOperator Xop = TTOperator::random(operatorDimensions, ranks);
	found_ranks = Xop.ranks();
	Xop.move_core(X.order()-1);
	Xop.move_core(0);
	MTEST(Xop.ranks() == found_ranks, Xop.ranks() << " vs " << found_ranks);
// 	{
// 		value_t mean = 0;
// 		Tensor tmp(Xop);
// 		for (size_t i=0; i<tmp.size; ++i) {
// 			mean += tmp[i];
// 		}
// 		mean /= (double)tmp.size;
// 		MTEST(std::abs(mean) < 0.5, "Xop mean " << mean);
// 	}
	
	TTOperator id = TTOperator::identity(operatorDimensions);
	MTEST(id.ranks() == std::vector<size_t>(9,1), id.ranks());
	
	Index i,j;
	TTTensor X2;
	X2(j/1) = id(j/2, i/2) * X(i/1);
	MTEST(frob_norm(X2-X) < 1e-14*1024, frob_norm(X2-X));
	
	TTTensor ttones = TTTensor::ones(dimensions);
	MTEST(ttones.ranks() == std::vector<size_t>(9,1), ttones.ranks());
	
	Tensor fones(ttones);
	MTEST(frob_norm(fones - Tensor::ones(dimensions)) < 1e-14*1024, "tt " << frob_norm(fones - Tensor::ones(dimensions)) );
	
	TTOperator ttopones = TTOperator::ones(dimensions);
	MTEST(ttopones.ranks() == std::vector<size_t>(4,1), ttopones.ranks());
	
	fones = Tensor(ttopones);
	MTEST(frob_norm(fones - Tensor::ones(dimensions)) < 1e-14*1024, "op " << frob_norm(fones - Tensor::ones(dimensions)) );
});


static misc::UnitTest tt_dyadic("TT", "dyadic_product", [](){
	TTOperator o1 = TTOperator::random({10,10}, {});
	TTOperator o2 = TTOperator::random({10,10}, {});
	TTOperator o3 = TTOperator::random({10,10}, {});
	TTTensor s1 = TTTensor::random({10},{});
	TTTensor s2 = TTTensor::random({10},{});
	TTTensor s3 = TTTensor::random({10},{});
	
	TTOperator O = dyadic_product<true>({o1,o2,o3});
	TTTensor S = dyadic_product<false>({s1,s2,s3});
	
	TEST(O.canonicalized);
	MTEST(O.corePosition == 0, O.corePosition);
	TEST(S.canonicalized);
	MTEST(S.corePosition == 0, S.corePosition);
	
	Index i,j;
	value_t r1 = frob_norm(o1(i/2,j/2) * s1(j/1));
	value_t r2 = frob_norm(o2(i/2,j/2) * s2(j/1));
	value_t r3 = frob_norm(o3(i/2,j/2) * s3(j/1));
	
	value_t R = frob_norm(O(i/2,j/2) * S(j/1));
	TEST(std::abs(R - r1*r2*r3) < 1e-12);
	
	S = dyadic_product(S,TTTensor::ones({10})) + dyadic_product(TTTensor::ones({10}), S);
	TEST(S.canonicalized);
	MTEST(S.corePosition == 0, S.corePosition);
	
	Tensor e0({10}, [&](const std::vector<size_t> &_idx){
		return (_idx[0]==0?1.0:0.0);
	});
	Tensor e1({10}, [&](const std::vector<size_t> &_idx){
		return (_idx[0]==1?1.0:0.0);
	});
	S *= 1/std::sqrt(2);
	S = dyadic_product(S,TTTensor(e0)) + dyadic_product(TTTensor(e1), S);
	TEST(S.canonicalized);
	MTEST(S.corePosition == 0, S.corePosition);
});

