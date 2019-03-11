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


#include <xerus.h>

#include "../../include/xerus/test/test.h"
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static const size_t MAX_RAM_REQUIREMENT = 1*1024*1024*1024;

static misc::UnitTest cons_sum_diff("Consistency", "sum_and_difference", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	for(size_t d = 0; d <= 6; ++d) {
		if (xerus::misc::product(dimsA) * 15 > MAX_RAM_REQUIREMENT) { break; }
		
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.2);
		TTOperator ttB(B, 0.2);
		TTOperator ttoC;
		TTTensor ttX(X, 0.2);
		TTTensor ttY(Y, 0.2);
		TTTensor tttC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;
		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(sA, ttA));
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, ttB));
		TEST(approx_equal(sB, ttB));
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, ttX));
		TEST(approx_equal(sX, ttX));
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, ttY));
		TEST(approx_equal(sY, ttY));
		
		
		C = X+X;
		sC = sX+sX;
		tttC = ttX+ttX;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		C = X+Y;
		sC = sX+sY;
		tttC = ttX+ttY;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		C = X-Y;
		sC = sX-sY;
		tttC = ttX-ttY;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		C = X+Y+X;
		sC = sX+sY+sX;
		tttC = ttX+ttY+ttX;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		
		C = 3.7*X+Y+X-3*Y;
		sC = 3.7*sX+sY+sX-3*sY;
		tttC = 3.7*ttX+ttY+ttX-3*ttY;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		
		C = A+A;
		sC = sA+sA;
		ttoC = ttA+ttA;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		C = A+B;
		sC = sA+sB;
		ttoC = ttA+ttB;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		
		C = A-B;
		sC = sA-sB;
		ttoC = ttA-ttB;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		
		C = 3.7*A+B-1.2*A;
		sC = 3.7*sA+sB-1.2*sA;
		ttoC = 3.7*ttA+ttB-1.2*ttA;
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims1;
		dimsA = dims1 | dims1;
		dimsB = dims1 | dims1;
	}
});


static misc::UnitTest cons_fixI("Consistency", "fixed_indices", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(2, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	// Add two initial dimension
	dims1.push_back(dimDist(rnd));
	dims1.push_back(dimDist(rnd));
	dims2.push_back(dimDist(rnd));
	dims2.push_back(dimDist(rnd));
	dimsX = dims1;
	dimsY = dims2;
	dimsA = dims1 | dims1;
	dimsB = dims2 | dims1;
	
	for(size_t d = 2; d <= 6; ++d) {
		if ((xerus::misc::product(dimsA)+xerus::misc::product(dimsB)) * 7 > MAX_RAM_REQUIREMENT) {
			break;
		}
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.75); 
		TTOperator ttB(B, 0.75); 
		TTTensor ttX(X, 0.6);
		TTTensor ttY(Y, 0.6);
		TTTensor ttC;
		
		TensorNetwork tnA(ttA);
		TensorNetwork tnB(ttB);
		TensorNetwork tnX(ttX);
		TensorNetwork tnY(ttY);
		TensorNetwork tnC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;
		
		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, tnA, 1e-14));
		TEST(approx_equal(A, ttA, 1e-14));
		TEST(approx_equal(sA, tnA, 1e-14));
		TEST(approx_equal(sA, ttA, 1e-14));
		
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, tnB, 1e-14));
		TEST(approx_equal(B, ttB, 1e-14));
		TEST(approx_equal(sB, tnB, 1e-14));
		TEST(approx_equal(sB, ttB, 1e-14));
		
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, tnX, 1e-14));
		TEST(approx_equal(X, ttX, 1e-14));
		TEST(approx_equal(sX, tnX, 1e-14));
		TEST(approx_equal(sX, ttX, 1e-14));
		
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, tnY, 1e-14));
		TEST(approx_equal(Y, ttY, 1e-14));
		TEST(approx_equal(sY, tnY, 1e-14));
		TEST(approx_equal(sY, ttY, 1e-14));
		
		
		
		C(i&0) = X(0, i&2, 1) + X(0, i&2, 0);
		sC(i&0) = sX(0, i&2, 1) + sX(0, i&2, 0);
		ttC(i&0) = ttX(0, i&2, 1)+ttX(0, i&2, 0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(i&0) = X(0, i&2, 1) * X(0, i&2, 0);
		sC(i&0) = sX(0, i&2, 1) * sX(0, i&2, 0);
		ttC(i&0) = ttX(0, i&2, 1) * ttX(0, i&2, 0);
		tnC(i&0) = tnX(0, i&2, 1) * tnX(0, i&2, 0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(k&0) = B(k/2, 1, j^(d-2), 0) * A(0, j^(d-2), 0, i/2) * X(i&0);
		sC(k&0) = sB(k/2, 1, j^(d-2), 0) * sA(0, j^(d-2), 0, i/2) * sX(i&0);
		ttC(k&0) = ttB(k/2, 1, j^(d-2), 0) * ttA(0, j^(d-2), 0, i/2) * ttX(i&0);
		tnC(k&0) = tnB(k/2, 1, j^(d-2), 0) * tnA(0, j^(d-2), 0, i/2) * tnX(i&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dims2.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims2;
		dimsA = dims1 | dims1;
		dimsB = dims2 | dims1;
	}
});


static misc::UnitTest cons_op_x_t("Consistency", "operator_times_tensor", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	for(size_t d = 0; d <= 6; ++d) {
		if ((xerus::misc::product(dimsA)+xerus::misc::product(dimsB)) * 7 > MAX_RAM_REQUIREMENT) {
			break;
		}
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.75); 
		TTOperator ttB(B, 0.75); 
		TTTensor ttX(X, 0.6);
		TTTensor ttY(Y, 0.6);
		TTTensor ttC;
		
		TensorNetwork tnA(ttA);
		TensorNetwork tnB(ttB);
		TensorNetwork tnX(ttX);
		TensorNetwork tnY(ttY);
		TensorNetwork tnC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;
		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, tnA, 1e-14));
		TEST(approx_equal(A, ttA, 1e-14));
		TEST(approx_equal(sA, tnA, 1e-14));
		TEST(approx_equal(sA, ttA, 1e-14));
		
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, tnB, 1e-14));
		TEST(approx_equal(B, ttB, 1e-14));
		TEST(approx_equal(sB, tnB, 1e-14));
		TEST(approx_equal(sB, ttB, 1e-14));
		
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, tnX, 1e-14));
		TEST(approx_equal(X, ttX, 1e-14));
		TEST(approx_equal(sX, tnX, 1e-14));
		TEST(approx_equal(sX, ttX, 1e-14));
		
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, tnY, 1e-14));
		TEST(approx_equal(Y, ttY, 1e-14));
		TEST(approx_equal(sY, tnY, 1e-14));
		TEST(approx_equal(sY, ttY, 1e-14));
		
		
		
		C(i&0) = A(i/2,j/2)*X(j&0);
		sC(i&0) = sA(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttA(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnA(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(i&0) = B(i/2,j/2)*X(j&0);
		sC(i&0) = sB(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttB(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnB(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(j&0) = B(i/2,j/2)*Y(i&0);
		sC(j&0) = sB(i/2,j/2)*sY(i&0);
		ttC(j&0) = ttB(i/2,j/2)*ttY(i&0);
		tnC(j&0) = tnB(i/2,j/2)*tnY(i&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(k&0) = B(k/2,i/2)*A(i/2,j/2)*X(j&0);
		sC(k&0) = sB(k/2,i/2)*sA(i/2,j/2)*sX(j&0);
		ttC(k&0) = ttB(k/2,i/2)*ttA(i/2,j/2)*ttX(j&0);
		tnC(k&0) = tnB(k/2,i/2)*tnA(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC, 1e-13));
		TEST(approx_equal(C, tnC, 1e-13));
		TEST(approx_equal(C, ttC, 1e-13));
		TEST(approx_equal(sC, tnC, 1e-13));
		TEST(approx_equal(sC, ttC, 1e-13));
		
		
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dims2.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims2;
		dimsA = dims1 | dims1;
		dimsB = dims2 | dims1;
	}
});

static misc::UnitTest cons_fix_mode("Consistency", "fix_mode", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	for(size_t d = 1; d <= 6; ++d) {
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dims2.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims2;
		dimsA = dims1 | dims1;
		dimsB = dims2 | dims1;
		
		if ((xerus::misc::product(dimsA)+xerus::misc::product(dimsB)) * 7 > MAX_RAM_REQUIREMENT) {
			break;
		}
		
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.75); 
		TTOperator ttB(B, 0.75); 
		TTTensor ttX(X, 0.6);
		TTTensor ttY(Y, 0.6);
		TTTensor ttC;
		
		TensorNetwork tnA(ttA);
		TensorNetwork tnB(ttB);
		TensorNetwork tnX(ttX);
		TensorNetwork tnY(ttY);
		TensorNetwork tnC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;
		
		
		std::uniform_int_distribution<size_t> slateSelect(0, d-1);
		const size_t slate = slateSelect(rnd);
		std::uniform_int_distribution<size_t> posSelect(0, dimsX[slate]-1);
		const size_t positionX = posSelect(rnd);
		std::uniform_int_distribution<size_t> posSelectY(0, dimsY[slate]-1);
		const size_t positionY = posSelectY(rnd);
		
		A.fix_mode(slate+d, positionX);
		A.fix_mode(slate, positionX);
		B.fix_mode(slate+d, positionX);
		B.fix_mode(slate, positionY);
		X.fix_mode(slate, positionX);
		Y.fix_mode(slate, positionY);
		
		sA.fix_mode(slate+d, positionX);
		sA.fix_mode(slate, positionX);
		sB.fix_mode(slate+d, positionX);
		sB.fix_mode(slate, positionY);
		sX.fix_mode(slate, positionX);
		sY.fix_mode(slate, positionY);
		
		ttA = TTOperator(A); // Can't use fix_mode
		ttB = TTOperator(B); // Can't use fix_mode
		ttX.fix_mode(slate, positionX);
		ttY.fix_mode(slate, positionY);
		
		ttX.require_correct_format();
		ttY.require_correct_format();
		
		tnA.fix_mode(slate+d, positionX);
		tnA.fix_mode(slate, positionX);
		tnB.fix_mode(slate+d, positionX);
		tnB.fix_mode(slate, positionY);
		tnX.fix_mode(slate, positionX);
		tnY.fix_mode(slate, positionY);
		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, tnA, 1e-14));
		TEST(approx_equal(A, ttA, 1e-14));
		TEST(approx_equal(sA, tnA, 1e-14));
		TEST(approx_equal(sA, ttA, 1e-14));
		
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, tnB, 1e-14));
		TEST(approx_equal(B, ttB, 1e-14));
		TEST(approx_equal(sB, tnB, 1e-14));
		TEST(approx_equal(sB, ttB, 1e-14));
		
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, tnX, 1e-14));
		TEST(approx_equal(X, ttX, 1e-14));
		TEST(approx_equal(sX, tnX, 1e-14));
		TEST(approx_equal(sX, ttX, 1e-14));
		
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, tnY, 1e-14));
		TEST(approx_equal(Y, ttY, 1e-14));
		TEST(approx_equal(sY, tnY, 1e-14));
		TEST(approx_equal(sY, ttY, 1e-14));
		
		
		
		C(i&0) = A(i/2,j/2)*X(j&0);
		sC(i&0) = sA(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttA(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnA(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC, 1e-14));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(i&0) = B(i/2,j/2)*X(j&0);
		sC(i&0) = sB(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttB(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnB(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(j&0) = B(i/2,j/2)*Y(i&0);
		sC(j&0) = sB(i/2,j/2)*sY(i&0);
		ttC(j&0) = ttB(i/2,j/2)*ttY(i&0);
		tnC(j&0) = tnB(i/2,j/2)*tnY(i&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(k&0) = B(k/2,i/2)*A(i/2,j/2)*X(j&0);
		sC(k&0) = sB(k/2,i/2)*sA(i/2,j/2)*sX(j&0);
		ttC(k&0) = ttB(k/2,i/2)*ttA(i/2,j/2)*ttX(j&0);
		tnC(k&0) = tnB(k/2,i/2)*tnA(i/2,j/2)*tnX(j&0);
		
		MTEST(approx_equal(C, sC, 1e-14), frob_norm(C-sC));
		MTEST(approx_equal(C, tnC, 1e-14), frob_norm(C-Tensor(tnC)));
		MTEST(approx_equal(C, ttC, 1e-14), frob_norm(C-Tensor(ttC)));
		MTEST(approx_equal(sC, tnC, 1e-14), frob_norm(sC- Tensor(tnC)));
		MTEST(approx_equal(sC, ttC, 1e-14), frob_norm(sC-Tensor(ttC)));
		
	}
});


static misc::UnitTest cons_resize_dim("Consistency", "resize_mode", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	for(size_t d = 1; d <= 6; ++d) {
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dims2.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims2;
		dimsA = dims1 | dims1;
		dimsB = dims2 | dims1;
		
		if ((xerus::misc::product(dimsA)+xerus::misc::product(dimsB)) * 7 > MAX_RAM_REQUIREMENT) {
			break;
		}
		
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.75); 
		TTOperator ttB(B, 0.75); 
		TTTensor ttX(X, 0.6);
		TTTensor ttY(Y, 0.6);
		TTTensor ttC;
		
		TensorNetwork tnA(ttA);
		TensorNetwork tnB(ttB);
		TensorNetwork tnX(ttX);
		TensorNetwork tnY(ttY);
		TensorNetwork tnC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;
		
		std::uniform_int_distribution<size_t> dimensionSelect(0, d-1);
		const size_t dimension = dimensionSelect(rnd);
		
		const size_t newDim1 = dimDist(rnd);
		const size_t newDim2 = dimDist(rnd);
		std::uniform_int_distribution<size_t> posSelect1(dimsX[dimension] - std::min(newDim1, dimsX[dimension]), dimsX[dimension]);
		std::uniform_int_distribution<size_t> posSelect2(dimsY[dimension] - std::min(newDim2, dimsY[dimension]), dimsY[dimension]);
		const size_t position1 = posSelect1(rnd);
		const size_t position2 = posSelect2(rnd);
		
		A.resize_mode(dimension+d, newDim1, position1);
		A.resize_mode(dimension, newDim1, position1);
		B.resize_mode(dimension+d, newDim1, position1);
		B.resize_mode(dimension, newDim2, position2);
		X.resize_mode(dimension, newDim1, position1);
		Y.resize_mode(dimension, newDim2, position2);
		
		sA.resize_mode(dimension+d, newDim1, position1);
		sA.resize_mode(dimension, newDim1, position1);
		sB.resize_mode(dimension+d, newDim1, position1);
		sB.resize_mode(dimension, newDim2, position2);
		sX.resize_mode(dimension, newDim1, position1);
		sY.resize_mode(dimension, newDim2, position2);
		
		ttA.resize_mode(dimension+d, newDim1, position1);
		ttA.resize_mode(dimension, newDim1, position1);
		ttB.resize_mode(dimension+d, newDim1, position1);
		ttB.resize_mode(dimension, newDim2, position2);
		ttX.resize_mode(dimension, newDim1, position1);
		ttY.resize_mode(dimension, newDim2, position2);
		
		tnA.resize_mode(dimension+d, newDim1, position1);
		tnA.resize_mode(dimension, newDim1, position1);
		tnB.resize_mode(dimension+d, newDim1, position1);
		tnB.resize_mode(dimension, newDim2, position2);
		tnX.resize_mode(dimension, newDim1, position1);
		tnY.resize_mode(dimension, newDim2, position2);
		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, tnA, 1e-14));
		TEST(approx_equal(A, ttA, 1e-14));
		TEST(approx_equal(sA, tnA, 1e-14));
		TEST(approx_equal(sA, ttA, 1e-14));
		
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, tnB, 1e-14));
		TEST(approx_equal(B, ttB, 1e-14));
		TEST(approx_equal(sB, tnB, 1e-14));
		TEST(approx_equal(sB, ttB, 1e-14));
		
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, tnX, 1e-14));
		TEST(approx_equal(X, ttX, 1e-14));
		TEST(approx_equal(sX, tnX, 1e-14));
		TEST(approx_equal(sX, ttX, 1e-14));
		
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, tnY, 1e-14));
		TEST(approx_equal(Y, ttY, 1e-14));
		TEST(approx_equal(sY, tnY, 1e-14));
		TEST(approx_equal(sY, ttY, 1e-14));
		
		
		
		C(i&0) = A(i/2,j/2)*X(j&0);
		sC(i&0) = sA(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttA(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnA(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC, 1e-14));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(i&0) = B(i/2,j/2)*X(j&0);
		sC(i&0) = sB(i/2,j/2)*sX(j&0);
		ttC(i&0) = ttB(i/2,j/2)*ttX(j&0);
		tnC(i&0) = tnB(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(j&0) = B(i/2,j/2)*Y(i&0);
		sC(j&0) = sB(i/2,j/2)*sY(i&0);
		ttC(j&0) = ttB(i/2,j/2)*ttY(i&0);
		tnC(j&0) = tnB(i/2,j/2)*tnY(i&0);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
		
		
		C(k&0) = B(k/2,i/2)*A(i/2,j/2)*X(j&0);
		sC(k&0) = sB(k/2,i/2)*sA(i/2,j/2)*sX(j&0);
		ttC(k&0) = ttB(k/2,i/2)*ttA(i/2,j/2)*ttX(j&0);
		tnC(k&0) = tnB(k/2,i/2)*tnA(i/2,j/2)*tnX(j&0);
		
		TEST(approx_equal(C, sC, 1e-14));
		TEST(approx_equal(C, tnC, 1e-14));
		TEST(approx_equal(C, ttC, 1e-14));
		TEST(approx_equal(sC, tnC, 1e-14));
		TEST(approx_equal(sC, ttC, 1e-14));
	}
});


static misc::UnitTest cons_entrywise_prod("Consistency", "entrywise_product", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dims3, dimsX, dimsY, dimsA, dimsB;
	
	const Index i, j, k;
	
	for(size_t d = 1; d <= 6; ++d) {
		if ((xerus::misc::product(dimsA)+xerus::misc::product(dimsB)) * 9 > MAX_RAM_REQUIREMENT) {
			break;
		}
		
		Tensor A = Tensor::random(dimsA);
		Tensor B = Tensor::random(dimsB);
		Tensor X = Tensor::random(dimsX);
		Tensor Y = Tensor::random(dimsY);
		Tensor C;
		
		TTOperator ttA(A, 0.33);
		TTOperator ttB(B, 0.33);
		TTOperator ttoC;
		TTTensor ttX(X, 0.33);
		TTTensor ttY(Y, 0.33);
		TTTensor tttC;
		
		A = ttA;
		B = ttB;
		X = ttX;
		Y = ttY;
		
		Tensor sA = A.sparse_copy();
		Tensor sB = B.sparse_copy();
		Tensor sX = X.sparse_copy();
		Tensor sY = Y.sparse_copy();
		Tensor sC;

		
		TEST(approx_equal(A, sA));
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(sA, ttA));
		
		TEST(approx_equal(B, sB));
		TEST(approx_equal(B, ttB));
		TEST(approx_equal(sB, ttB));
		
		TEST(approx_equal(X, sX));
		TEST(approx_equal(X, ttX));
		TEST(approx_equal(sX, ttX));
		
		TEST(approx_equal(Y, sY));
		TEST(approx_equal(Y, ttY));
		TEST(approx_equal(sY, ttY));
		
		
		C = entrywise_product(X, X);
		sC = entrywise_product(sX, sX);
		tttC = entrywise_product(ttX, ttX);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		C = entrywise_product(X, Y);
		sC = entrywise_product(sX, sY);
		tttC = entrywise_product(ttX, ttY);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		C = entrywise_product(entrywise_product(X, Y), X);
		sC = entrywise_product(entrywise_product(sX, sY), sX);
		tttC = entrywise_product(entrywise_product(ttX, ttY), ttX);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		
		C = entrywise_product(entrywise_product(entrywise_product(3.7*X, Y), X), -3*Y);
		sC = entrywise_product(entrywise_product(entrywise_product(3.7*sX, sY), sX), -3*sY);
		tttC = entrywise_product(entrywise_product(entrywise_product(3.7*ttX, ttY), ttX), -3*ttY);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, tttC, 1e-14));
		TEST(approx_equal(sC, tttC, 1e-14));
		
		
		C = entrywise_product(A, A);
		sC = entrywise_product(sA, sA);
		ttoC = entrywise_product(ttA, ttA);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		C = entrywise_product(A, B);
		sC = entrywise_product(sA, sB);
		ttoC = entrywise_product(ttA, ttB);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		
		C = entrywise_product(entrywise_product(3.7*A, B), -1.2*A);
		sC = entrywise_product(entrywise_product(3.7*sA, sB), -1.2*sA);
		ttoC = entrywise_product(entrywise_product(3.7*ttA, ttB), -1.2*ttA);
		
		TEST(approx_equal(C, sC));
		TEST(approx_equal(C, ttoC, 1e-14));
		TEST(approx_equal(sC, ttoC, 1e-14));
		
		
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsY = dims1;
		dimsA = dims1 | dims1;
		dimsB = dims1 | dims1;
	}
});



static misc::UnitTest cons_named_constructors("Consistency", "named_constructors", []() {
	std::mt19937_64 &rnd = xerus::misc::randomEngine;
	std::uniform_int_distribution<size_t> dimDist(1, 3);
	
	std::vector<size_t> dims1, dims2, dimsX, dimsA;
	
	for(size_t d = 1; d <= 6; ++d) {
		Tensor A = Tensor::ones(dimsA);
		Tensor X = Tensor::ones(dimsX);
		
		TTOperator ttA = TTOperator::ones(dimsA);
		TTTensor ttX = TTTensor::ones(dimsX);
		
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(X, ttX));
		
		
		A = Tensor::identity(dimsA);
		ttA = TTOperator::identity(dimsA);
		
		TEST(approx_equal(A, ttA));
		
		
		A = Tensor::kronecker(dimsA);
		X = Tensor::kronecker(dimsX);
		ttA = TTOperator::kronecker(dimsA);
		ttX = TTTensor::kronecker(dimsX);
		
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(X, ttX));
		
		std::uniform_int_distribution<size_t> posADist(0, misc::product(dimsA)-1);
		std::uniform_int_distribution<size_t> posXDist(0, misc::product(dimsX)-1);
		const auto posA = posADist(rnd);
		const auto posX = posXDist(rnd);
		
		A = Tensor::dirac(dimsA, posA);
		X = Tensor::dirac(dimsX, posX);
		ttA = TTOperator::dirac(dimsA, posA);
		ttX = TTTensor::dirac(dimsX, posX);
		
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(X, ttX));
		
		A = Tensor::dirac(dimsA, Tensor::position_to_multiIndex(posA, dimsA));
		X = Tensor::dirac(dimsX, Tensor::position_to_multiIndex(posX, dimsX));
		ttA = TTOperator::dirac(dimsA, Tensor::position_to_multiIndex(posA, dimsA));
		ttX = TTTensor::dirac(dimsX, Tensor::position_to_multiIndex(posX, dimsX));
		
		TEST(approx_equal(A, ttA));
		TEST(approx_equal(X, ttX));
		
		// Add a new dimension
		dims1.push_back(dimDist(rnd));
		dimsX = dims1;
		dimsA = dims1 | dims1;
	}
});
