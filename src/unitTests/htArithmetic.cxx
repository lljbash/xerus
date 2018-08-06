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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static misc::UnitTest ht_consitent("HT", "consistent networks", [](){
	auto ones = xerus::HTTensor::ones({3,4,5});
	auto rt = xerus::HTNetwork<false>::random({2,3,4,5,6,7},3);
	auto rtop = xerus::HTNetwork<true>::random({2,3,4,5,6,7},3);
	auto rt2 = xerus::Tensor::random({2,3,4,5,7,8});

	xerus::HTOperator rtop2(rt2,0.5);


	rt.require_valid_network();
	ones.require_valid_network();
	rtop.require_valid_network();
	rtop2.require_valid_network();
});


static misc::UnitTest ht_sum("HT", "sum", [](){
	//Random numbers
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::normal_distribution<value_t> dist (0.0, 1.0);
	std::uniform_int_distribution<size_t> intDist (1, 10);
	
	Index i;
	
	std::vector<size_t> dimensions;
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	dimensions.push_back(intDist(rnd));
	Tensor A = Tensor::random(dimensions, dist);
	Tensor B = Tensor::random(dimensions, dist);
	Tensor C;
	HTTensor ttA(A);
	HTTensor ttB(B);
	HTTensor ttC;
	HTOperator toA(A);
	HTOperator toB(B);
	HTOperator toC;
	
//	C(i&0) = A(i&0) + B(i&0);
//	ttC(i&0) = ttA(i&0) + ttB(i&0);
//	toC(i&0) = toA(i&0) + toB(i&0);
//	MTEST(frob_norm(Tensor(ttC)(i&0) - C(i&0)) < 3.1*1e-13, frob_norm(Tensor(ttC)(i&0) - C(i&0)));
//	MTEST(frob_norm(Tensor(toC)(i&0) - C(i&0)) < 3.1*1e-13, frob_norm(Tensor(toC) - C));
});

//static misc::UnitTest tt_diff("TT", "difference", [](){
//	Tensor A = Tensor::random({10,10,10,10});
//	Tensor B = Tensor::random({10,10,10,10});
//	Tensor C;
//	TTTensor ttA(A);
//	TTTensor ttB(B);
//	TTTensor ttC(4);
//
//	Index i;
//	C(i&0) = A(i&0) - B(i&0);
//	ttC(i&0) = ttA(i&0) - ttB(i&0);
//
//	double fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	MTEST(fnorm < 1e-12, fnorm);
//});
//
//
//static misc::UnitTest tt_real_diff("TT", "real_difference", [](){
//	TTTensor ttA = TTTensor::random({10,10,10,10,10}, {4,4,4,4});
//	TTTensor ttB = TTTensor::random({10,10,10,10,10}, {4,4,4,4});
//	TTTensor ttC(5);
//
//	Index i;
//	ttC(i&0) = ttA(i&0) - ttA(i&0);
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "1 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = ttB(i&0) - ttB(i&0);
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "2 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (ttA(i&0) + ttB(i&0)) - (ttA(i&0) + ttB(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "3 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (ttA(i&0) + ttB(i&0));
//	TEST(ttC.ranks() == std::vector<size_t>({8,8,8,8}));
//	ttC(i&0) = (ttB(i&0) + ttA(i&0));
//	TEST(ttC.ranks() == std::vector<size_t>({8,8,8,8}));
//	ttC(i&0) = (ttA(i&0) + ttB(i&0)) - (ttB(i&0) + ttA(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "4 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (73*ttA(i&0) + ttB(i&0)) - (ttB(i&0) + 73*ttA(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 1e-9, "5 " << frob_norm(ttC(i&0)));
//
//	ttA = TTTensor::random({10,10,10,10,10}, {2,5,7,2});
//	ttC(i&0) = ttA(i&0) - ttA(i&0);
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "6 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = ttB(i&0) - ttB(i&0);
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "7 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (ttA(i&0) + ttB(i&0)) - (ttA(i&0) + ttB(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "8 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (ttA(i&0) + ttB(i&0)) - (ttB(i&0) + ttA(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 1e-11, "9 " << frob_norm(ttC(i&0)));
//
//	ttC(i&0) = (73*ttA(i&0) + ttB(i&0)) - (ttB(i&0) + 73*ttA(i&0));
//	MTEST(frob_norm(ttC(i&0)) < 5e-10, "10 " << frob_norm(ttC(i&0)));
//});
//
//static misc::UnitTest tt_diff_stacks("TT", "difference_of_TTStacks", [](){
//	TTOperator ttO = TTOperator::random({10,10,10,10,10,10,10,10,10,10}, {4,4,4,4});
//	TTTensor ttA = TTTensor::random({10,10,10,10,10}, {4,4,4,4});
//	TTTensor ttB = TTTensor::random({10,10,10,10,10}, {4,4,4,4});
//	TTTensor ttC;
//
//	Index i,j,k;
//	ttC(i&0) = ttO(i/2, j/2)*ttA(j&0) - ttO(i/2, j/2)*ttA(j&0);
//	LOG(unit_tests, "Frob norm 1 " << frob_norm(ttC(i&0)));
//	TEST(frob_norm(ttC(i&0)) < 1e-7);
//
//	ttC(i&0) = ttO(i/2, j/2)*ttB(j&0) - ttO(i/2, j/2)*ttB(j&0);
//	LOG(unit_tests, "Frob norm 2 " << frob_norm(ttC(i&0)));
//	TEST(frob_norm(ttC(i&0)) < 1e-7);
//});
//
//static misc::UnitTest tt_stack_norm("TT", "ttStacks_frob_norm", [](){
//	const Index i, j, k;
//
//	TTOperator ttO1 = TTOperator::identity({10,10,10,10,10,10,10,10,10,10});
//	TTOperator ttO2 = TTOperator::identity({10,10,10,10,10,10,10,10,10,10});
//
//	MTEST(misc::approx_equal(frob_norm(ttO1(i&0)*ttO2(i&0)), double(misc::pow(10, 5))), frob_norm(ttO1(i&0)*ttO2(i&0)) << " vs " << misc::pow(10, 5));
//
//	TEST(misc::approx_equal(frob_norm(ttO1(i/2, j/2)*ttO2(j/2, k/2)), std::sqrt(misc::pow(10, 5))));
//});
//
//static misc::UnitTest tt_spec_sumdiff("TT", "special_sum_diff", [](){
//	Tensor A({10,10,10,10}); // NOTE that this is the 0 tensor
//	Tensor B = Tensor::random({10,10,10,10});
//	Tensor C;
//	TTTensor ttA(A);
//	TTTensor ttB(B);
//	TTTensor ttC(4);
//
//
//	Index i;
//
//	C(i&0) = A(i&0) + B(i&0);
//	ttC(i&0) = ttA(i&0) + ttB(i&0);
//	TEST(frob_norm(Tensor(ttC)(i&0) - C(i&0)) < 5*1e-13);
//	TEST(frob_norm(Tensor(ttC)(i&0) - Tensor(ttB)(i&0)) < 3.1*1e-13);
//
//	C(i&0) = B(i&0) + A(i&0);
//	ttC(i&0) = ttB(i&0) + ttA(i&0);
//	TEST(frob_norm(Tensor(ttC)(i&0) - C(i&0)) < 5*1e-13);
//	TEST(frob_norm(Tensor(ttC)(i&0) - Tensor(ttB)(i&0)) < 3.1*1e-13);
//
//	C(i&0) = A(i&0) - B(i&0);
//	ttC(i&0) = ttA(i&0) - ttB(i&0);
//	MTEST(frob_norm(Tensor(ttC)(i&0) - C(i&0)) < 5*1e-13, frob_norm(Tensor(ttC)(i&0) - C(i&0)));
//	MTEST(frob_norm(Tensor(ttC)(i&0) + Tensor(ttB)(i&0)) < 3.1*1e-13, frob_norm(Tensor(ttC)(i&0) + Tensor(ttB)(i&0)));
//
//	C(i&0) = B(i&0) - A(i&0);
//	ttC(i&0) = ttB(i&0) - ttA(i&0);
//	MTEST(frob_norm(Tensor(ttC)(i&0) - C(i&0)) < 5*1e-13, frob_norm(Tensor(ttC)(i&0) - C(i&0)));
//	MTEST(frob_norm(Tensor(ttC)(i&0) - Tensor(ttB)(i&0)) < 3.1*1e-13, frob_norm(Tensor(ttC)(i&0) - Tensor(ttB)(i&0)));
//
//	Tensor X({10});
//	Tensor Y = Tensor::random({10});
//	Tensor Z;
//	TTTensor ttX(X);
//	TTTensor ttY(Y);
//	TTTensor ttZ(1);
//
//	Z(i&0) = X(i&0) + Y(i&0);
//	ttZ(i&0) = ttX(i&0) + ttY(i&0);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Z(i&0)) < 3.1*1e-13);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Tensor(ttY)(i&0)) < 3.1*1e-13);
//
//	Z(i&0) = Y(i&0) + X(i&0);
//	ttZ(i&0) = ttY(i&0) + ttX(i&0);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Z(i&0)) < 3.1*1e-13);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Tensor(ttY)(i&0)) < 3.1*1e-13);
//
//	Z(i&0) = X(i&0) - Y(i&0);
//	ttZ(i&0) = ttX(i&0) - ttY(i&0);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Z(i&0)) < 3.1*1e-13);
//	TEST(frob_norm(Tensor(ttZ)(i&0) + Tensor(ttY)(i&0)) < 3.1*1e-13);
//
//	Z(i&0) = Y(i&0) - X(i&0);
//	ttZ(i&0) = ttY(i&0) - ttX(i&0);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Z(i&0)) < 3.1*1e-13);
//	TEST(frob_norm(Tensor(ttZ)(i&0) - Tensor(ttY)(i&0)) < 3.1*1e-13);
//});
//
//static misc::UnitTest tt_prod("TT", "product", [](){
//	Index i,j,k,l;
//
//	TTOperator ttA = TTOperator::random({10,10,10,10}, 1);
//	TTOperator ttB = TTOperator::random({10,10,10,10}, 1);
//	TTTensor ttD = TTTensor::random({10,10}, 2);
//	Tensor A(ttA);
//	Tensor B(ttB);
//	Tensor D(ttD);
//
//	Tensor C;
//	TTTensor ttCt;
//	TTOperator ttC;
//
//	C(i^2) = A(i^2,j^2) * D(j^2);
//	ttCt(i^2) = ttA(i^2,j^2) * ttD(j^2);
//	double fnorm = frob_norm(Tensor(ttCt)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	C(i^2,k^2) = A(i^2,j^2) * B(j^2,k^2);
//	ttC(i^2,k^2) = ttA(i^2,j^2) * ttB(j^2,k^2);
//	TEST(ttC.nodes.size() == 4);
//	fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	C(i/2,k/2) = A(j/2,i/2) * B(j/2,k/2);
//	ttC(i^2,k/2) = ttA(j^2,i/2) * ttB(j^2,k/2);
//	fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	ttC(i^2,k/2) = ttB(j/2,k/2) * ttA(j^2,i^2);
//	fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	C(i^2,k^2) = A(i^2,j^2) * B(k^2,j^2);
//	ttC(i^2,k^2) = ttA(i^2,j^2) * ttB(k^2,j^2);
//	fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	C(i^2,k^2) = A(j^2,i^2) * B(k^2,j^2);
//	ttC(i^2,k^2) = ttA(j^2,i^2) * ttB(k^2,j^2);
//	fnorm = frob_norm(Tensor(ttC)(i&0) - C(i&0));
//	LOG(unit_tests, "frob_norm " << fnorm);
//	TEST(fnorm < 10*10*10*10*1e-15);
//
//	FAILTEST(ttC(i^2,k^2) = ttA(j^2,i) * ttB(k^2,j^2));
//	FAILTEST(ttC(i^2,k^2) = ttA(j^2,i^2) * ttB(k^2,k^2));
//});
//
//static misc::UnitTest tt_id("TT", "identities", [](){
//	Tensor I = Tensor({2,2,2,2},[](const std::vector<size_t>& _idx)->value_t{
//		if ((_idx[0] == _idx[2] && _idx[1] == _idx[3]) || (_idx[0] == 0 && _idx[1]==1 &&_idx[2] ==0 && _idx[3]==0)) {
//			return 1;
//		} else {
//			return 0;
//		}
//	}
//	);
//	TTOperator ttI(I);
//	TTOperator ttC(4);
//	Tensor C;
//	Index i,j,k;
//
//	ttC(i^2,k^2) = ttI(i^2,j^2) * ttI(j^2,k^2);
//	C(i^2,k^2) = I(i^2,j^2) * I(j^2,k^2);
//	LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//	TEST(approx_equal(C, Tensor(ttC), 1e-15));
//
//	ttC(k^2,i^2) = ttI(i^2,j^2) * ttI(j^2,k^2);
//	C(k^2,i^2) = I(i^2,j^2) * I(j^2,k^2);
//	LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//	TEST(approx_equal(C, Tensor(ttC), 1e-15));
//
//	ttC(i^2,k^2) = ttI(i^2,j^2) * ttI(k^2,j^2);
//	C(i^2,k^2) = I(i^2,j^2) * I(k^2,j^2);
//	LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//	TEST(approx_equal(C, Tensor(ttC), 1e-15));
//
//	ttC(k^2,i^2) = ttI(i^2,j^2) * ttI(k^2,j^2);
//	C(k^2,i^2) = I(i^2,j^2) * I(k^2,j^2);
//	LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//	TEST(approx_equal(C, Tensor(ttC), 1e-15));
//});
//
//static misc::UnitTest tt_trans("TT", "transpose", [](){
//	Tensor A = Tensor::random({10,10,10,10});
//	Tensor B;
//	TTOperator ttA(A);
//	Index i,j;
//
//	B(i^2,j^2) = A(j^2,i^2);
//	ttA.transpose();
//	LOG(unit_test, frob_norm(B(i&0) - Tensor(ttA)(i&0)));
//	TEST(approx_equal(B, Tensor(ttA), 1e-14));
//});
//
//static misc::UnitTest tt_axb("TT", "ax_b", [](){
//	TTTensor X = TTTensor::random({10,10,10}, {2,2});
//	TTTensor B = TTTensor::random({10,10,10}, {2,2});
//
//	Tensor I({10,10,10,10,10,10}, [](const std::vector<size_t> &_idx) {
//		if (_idx[0]==_idx[3] && _idx[1] == _idx[4] && _idx[2] == _idx[5]) {
//			return 1.0;
//		} else {
//			return 0.0;
//		}
//	});
//
//	TTOperator A(I);
//	TTTensor T;
//	TTTensor S;
//
//	Index i,j,k;
//
//
//	T(i^3) = A(i^3, j^3) * X(j^3);
//	T(i^3) = T(i^3) - B(i^3);
//	S(i^3) = A(i^3, j^3) * X(j^3) - B(i^3);
//	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
//	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
//
//	Tensor fA(A);
//	Tensor fX(X);
//	Tensor fT;
//	fT(i^3) = fA(i^3, j^3) * fX(j^3);
//	TEST(frob_norm(fT(i^3) - fX(i^3))<1e-7);
//
//	T(i^3) = A(i^3, j^3) * X(j^3);
//	TEST(frob_norm(Tensor(T) - fT) < 1e-7);
//	TEST(frob_norm(Tensor(T)(i^3) - fT(i^3)) < 1e-7);
//
//	T(i^3) = A(i^3, j^3) * X(j^3);
//	TEST(frob_norm(A(i^3, j^3) * X(j^3) - T(i^3)) < 1e-7);
//
//	TEST(frob_norm(T(i^3) - X(i^3))<1e-7);
//	T(i^3) = T(i^3) - X(i^3);
//	S(i^3) = A(i^3, j^3) * X(j^3) - X(i^3);
//	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
//	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
//	TEST(frob_norm(S(i^3)) < 1e-7);
//
//	T(i^3) = A(j^3, i^3) * X(j^3);
//	TEST(frob_norm(T(i^3) - X(i^3))<1e-7);
//	T(i^3) = T(i^3) - X(i^3);
//	S(i^3) = A(j^3, i^3) * X(j^3) - X(i^3);
//	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
//	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
//	TEST(frob_norm(S(i^3)) < 1e-7);
//
//	T(i^3) = A(j^3, i^3) * B(j^3);
//	T(i^3) = T(i^3) - B(i^3);
//	S(i^3) = A(j^3, i^3) * B(j^3) - B(i^3);
//	LOG(unit_test, frob_norm(T(i^3)-S(i^3)));
//	TEST(frob_norm(T(i^3)-S(i^3)) < 1e-7);
//	TEST(frob_norm(S(i^3)) < 1e-7);
//});
//
//static misc::UnitTest tt_opt("TT", "operator_times_tensor", [](){
//	Tensor A = Tensor::random({10,10,10,10});
//	Tensor B = Tensor::random({10,10,10,10});
//	Tensor C = Tensor::random({10,10});
//	Tensor D;
//	Tensor Do;
//	TTOperator ttA(A);
//	ttA.round(2); A = Tensor(ttA);
//	TTOperator ttB(B);
//	ttB.round(2); B = Tensor(ttB);
//	TTTensor ttC(C);
//	TTTensor ttD(C);
//	TTOperator ttDo(A);
//
//	Index i,j,k,l,m;
//	ttD(i^2) = ttA(i^2,j^2) * ttB(j^2,k^2) * ttC(k^2);
//	D(i^2) = A(i^2,j^2) * B(j^2,k^2) * C(k^2);
//	MTEST(approx_equal(D, Tensor(ttD), 3e-13), "1 " << frob_norm(D-Tensor(ttD)) << " / " << frob_norm(D));
//
//	ttD(i^2) = ttA(i^2,j^2) * ttB(k^2,j^2) * ttC(k^2);
//	D(i^2) = A(i^2,j^2) * B(k^2,j^2) * C(k^2);
//	MTEST(approx_equal(D, Tensor(ttD), 2e-13), "2 " << frob_norm(D-Tensor(ttD)) << " / " << frob_norm(D));
//
//	ttDo(i^2,k^2) = ttA(i^2,j^2) * ttB(j^2,k^2);
//	Do(i^2,k^2) = A(i^2,j^2) * B(j^2,k^2);
//	MTEST(approx_equal(Do, Tensor(ttDo), 2e-13), "3 " << frob_norm(Do-Tensor(ttDo)) << " / " << frob_norm(Do));
//
//	ttDo(i^2,k^2) = ttA(i^2,j^2) * ttA(j^2,k^2);
//	Do(i^2,k^2) = A(i^2,j^2) * A(j^2,k^2);
//	MTEST(approx_equal(Do, Tensor(ttDo), 2e-13), "4 " << frob_norm(Do-Tensor(ttDo)) << " / " << frob_norm(Do));
//
//	ttDo(i^2,l^2) = ttA(i^2,j^2) * ttB(j^2,k^2) * ttA(l^2,k^2);
//	Do(i^2,l^2) = A(i^2,j^2) * B(j^2,k^2) * A(l^2,k^2);
//	MTEST(approx_equal(Do, Tensor(ttDo), 2e-13), "5 " << frob_norm(Do-Tensor(ttDo)) << " / " << frob_norm(Do));
//
//	ttDo(i^2,l^2) = ttA(i^2,j^2) * ttB(j^2,k^2) * ttB(l^2,k^2);
//	Do(i^2,l^2) = A(i^2,j^2) * B(j^2,k^2) * B(l^2,k^2);
//	MTEST(approx_equal(Do, Tensor(ttDo), 2e-15), "6 " << frob_norm(Do-Tensor(ttDo)) << " / " << frob_norm(Do));
//
//	ttDo(i^2,m^2) = ttA(i^2,j^2) * ttB(j^2,k^2) * ttB(l^2,k^2) * ttA(l^2,m^2);
//	Do(i^2,m^2) = A(i^2,j^2) * B(j^2,k^2) * B(l^2,k^2) * A(l^2,m^2);
//	MTEST(approx_equal(Do, Tensor(ttDo), 2e-13), "7 " << frob_norm(Do-Tensor(ttDo)) << " / " << frob_norm(Do));
//});
//
//static misc::UnitTest tt_fullcontr("TT", "full_contraction", [](){
//	Tensor A = Tensor::random({10,10,10,10});
//	Tensor B = Tensor::random({10,10,10,10});
//	TTTensor ttA(A);
//	TTTensor ttB(B);
//	TTOperator toA(A);
//	TTOperator toB(B);
//
//	Index i;
//	TEST(misc::approx_equal(frob_norm(A(i&0)), frob_norm(ttA(i&0)), 3e-13));
//	TEST(misc::approx_equal(frob_norm(A(i&0)), frob_norm(toA(i&0)), 2e-13));
//	TEST(misc::approx_equal(frob_norm(B(i&0)), frob_norm(ttB(i&0)), 2e-13));
//	TEST(misc::approx_equal(frob_norm(B(i&0)), frob_norm(toB(i&0)), 1e-13));
//	TEST(misc::approx_equal(frob_norm(A(i&0)-B(i&0)), frob_norm(ttA(i&0)-ttB(i&0)), 1e-12));
//	TEST(misc::approx_equal(frob_norm(A(i&0)-B(i&0)), frob_norm(toA(i&0)-toB(i&0)), 1e-12));
//	Tensor C;
//	C() = A(i/1)*B(i&0);
//
//	TTTensor ttC(0);
//	ttC() = ttA(i&0)*ttB(i&0);
//	TEST(misc::approx_equal(C[0], ttC[0], 1e-12));
//
//	TTOperator toC(0);
//	toC() = ttA(i&0)*ttB(i&0);
//	TEST(misc::approx_equal(C[{}], toC[{}], 1e-12));
//});
//
//static misc::UnitTest tt_disjoint("TT", "disjoint_product", [](){
//	//Random numbers
//	std::mt19937_64 &rnd = xerus::misc::randomEngine;
//	std::uniform_int_distribution<size_t> dimDist(1, 5);
//
//	std::vector<size_t> dimsA;
//	std::vector<size_t> dimsB;
//
//	const size_t D = 5;
//	Index i,j;
//
//	for(size_t d = 0; d <= D; ++d) {
//		Tensor A = Tensor::random(dimsA);
//		Tensor B = Tensor::random(dimsB);
//		Tensor C;
//		TTTensor ttA(A);
//		TTTensor ttB(B);
//		TTTensor ttC;
//
//
//		ttC = dyadic_product(ttA, ttB);
//		C(i/2,j/2) = A(i&0)*B(j&0);
//
//		LOG(unit_test, frob_norm(C(i&0) - Tensor(ttC)(i&0)));
//		TEST(approx_equal(C, Tensor(ttC), 1e-13));
//
//		dimsA.push_back(dimDist(rnd));
//		dimsB.push_back(dimDist(rnd));
//	}
//});
