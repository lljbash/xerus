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


#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;


static misc::UnitTest alg_retr("Algorithm", "retractions", [](){
	std::vector<size_t> stateDims(8,4);
	std::vector<size_t> stateRank(7,2);
	
	TTTensor X = TTTensor::random(stateDims, stateRank);
	TTTensor zero = TTTensor::random(stateDims, stateRank);
	zero *= 1e-16;
	
	TTTangentVector tangentZero(X, zero);
	for (size_t i=0; i<stateDims.size(); ++i) {
		MTEST(frob_norm(tangentZero.components[i]) < 1e-14, frob_norm(tangentZero.components[i]));
	}
	
	{
		TTTensor Y(X);
		ALSRetractionI(Y, tangentZero);
		MTEST(frob_norm(X-Y) < 1e-8, "alsI " << frob_norm(X-Y));
	}
	{
		TTTensor Y(X);
		ALSRetractionII(Y, zero);
		MTEST(frob_norm(X-Y) < 1e-8, "alsII " << frob_norm(X-Y));
	}
	{
		TTTensor Y(X);
		SubmanifoldRetractionI(Y, tangentZero);
		MTEST(frob_norm(X-Y) < 1e-8, "submaniI " << frob_norm(X-Y));
	}
	{
		TTTensor Y(X);
		SubmanifoldRetractionII(Y, zero);
		MTEST(frob_norm(X-Y) < 1e-8, "submaniII " << frob_norm(X-Y));
	}
	HOSVDRetraction hosvd(2);
	{
		TTTensor Y(X);
		hosvd(Y, tangentZero);
		MTEST(frob_norm(X-Y) < 1e-8, "hosvdI " << frob_norm(X-Y));
	}
	{
		TTTensor Y(X);
		hosvd(Y, zero);
		MTEST(frob_norm(X-Y) < 1e-8, "hosvdII " << frob_norm(X-Y));
	}
	
	TTTensor change = TTTensor::random(stateDims, stateRank);
	const value_t EPS = 1e-3;
	change *= EPS / frob_norm(change);
	
	Index i1,i2,i3,i4;
	TTTangentVector tangentChange(X, change);
	for (size_t i=0; i<stateDims.size(); ++i) {
		TEST(frob_norm(tangentChange.components[i]) > 1e-10);
		if (i!=0) {
			TEST(frob_norm(tangentChange.components[i](i1,i2,i3) * X.component(i)(i4,i2,i3)) < 1e-15);
		}
	}
	
	TTTensor riemannianChange = TTTensor(tangentChange);
	riemannianChange /= frob_norm(riemannianChange);
	
	{
		TTTensor Y(X);
		ALSRetractionI(Y, tangentChange);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "alsI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "alsI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "alsI " << fnorm << " " << angle << " " << EPS);
	}
	{
		TTTensor Y(X);
		ALSRetractionII(Y, change);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "alsII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "alsII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "alsII " << fnorm << " " << angle << " " << EPS);
	}
	{
		TTTensor Y(X);
		SubmanifoldRetractionI(Y, tangentChange);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "submaniI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "submaniI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "submaniI " << fnorm << " " << angle << " " << EPS);
	}
	{
		TTTensor Y(X);
		SubmanifoldRetractionII(Y, change);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "submaniII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "submaniII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "submaniII " << fnorm << " " << angle << " " << EPS);
	}
	{
		TTTensor Y(X);
		hosvd(Y, tangentChange);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "hosvdI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "hosvdI " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "hosvdI " << fnorm << " " << angle << " " << EPS);
	}
	{
		TTTensor Y(X);
		hosvd(Y, change);
		TTTensor actualChange = Y-X;
		value_t fnorm = frob_norm(actualChange);
		value_t angle = 1 - (value_t(riemannianChange(i1&0) * actualChange(i1&0)) / fnorm);
		MTEST(angle < 100 * EPS * EPS, "hosvdII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm > EPS * 1e-4, "hosvdII " << fnorm << " " << angle << " " << EPS);
		MTEST(fnorm < EPS, "hosvdII " << fnorm << " " << angle << " " << EPS);
	}
});


static misc::UnitTest tttv_ortho("TTTangentVector", "orthogonality", [](){
	std::vector<size_t> stateDims({2,3,5,4,3,2,4,1,2});
	std::vector<size_t> stateRank({ 2,2,4,1,3,4,2,2});
	Index j;
	
	TTTensor X = TTTensor::random(stateDims, stateRank);
	TTTensor delta = TTTensor::random(stateDims, stateRank);
	TTTangentVector tangentChange(X, delta);
	TTTensor Pdelta = TTTensor(tangentChange);
	TTTensor deltaPdelta = delta - Pdelta;
	
	// construct all possible basis vectors for the tangent plane
	for (size_t n=0; n<stateDims.size(); ++n) {
		for (size_t i=0; i<X.get_component(n).size; ++i) {
			TTTensor b(X);
			b.set_component(n, Tensor::dirac(b.component(n).dimensions, i));
			b.component(n).use_dense_representation();
			b.move_core(0);
			// check Pb = b
			TTTensor pb = TTTensor(TTTangentVector(X, b));
			MTEST(frob_norm(b-pb)/frob_norm(b) < 1e-13, n << " " << i << " " << frob_norm(b-pb)/frob_norm(b));
			
			// check orthogonality
			value_t scalarProd = value_t(deltaPdelta(j&0) * b(j&0));
			MTEST(std::abs(scalarProd) < 1e-9, n << " " << i << " prod: " << scalarProd << " <delta|b> = " << value_t(delta(j&0) * b(j&0)));
			
			MTEST(misc::approx_equal(value_t(delta(j&0) * b(j&0)), value_t(Pdelta(j&0) * b(j&0)), 2e-10),  n << " " << i << " " << value_t(delta(j&0) * b(j&0)) << " " << value_t(Pdelta(j&0) * b(j&0)));
		}
	}
});


static misc::UnitTest tttv_creation("TTTangentVector", "creation", [](){
	std::vector<size_t> stateDims({2,3,5,4,3,2,4,1,2});
	std::vector<size_t> stateRank({ 2,2,4,1,3,4,2,2});
	Index j;
	
	TTTensor X = TTTensor::random(stateDims, stateRank);
	TTTensor change = TTTensor::random(stateDims, stateRank);
	TTTangentVector tangentChange1(X, change);
	// projection should decrease norm
	MTEST(frob_norm(change) > frob_norm(TTTensor(tangentChange1)), frob_norm(change) << " " << frob_norm(TTTensor(tangentChange1)));
	
	// test of scalar products in tangent space
	value_t scalarProdInTangentSpace = std::sqrt(tangentChange1.scalar_product(tangentChange1));
	value_t scalarProdInEmbeddingSpace = frob_norm(TTTensor(tangentChange1));
	MTEST(misc::approx_equal(scalarProdInEmbeddingSpace, scalarProdInTangentSpace, 1e-14), 
		  "norm1 " << scalarProdInEmbeddingSpace << " " << scalarProdInTangentSpace << " diff " << (scalarProdInEmbeddingSpace-scalarProdInTangentSpace));
	
	scalarProdInTangentSpace = tangentChange1.frob_norm();
	MTEST(misc::approx_equal(scalarProdInEmbeddingSpace, scalarProdInTangentSpace, 1e-14), 
		  "norm2 " << scalarProdInEmbeddingSpace << " " << scalarProdInTangentSpace << " diff " << (scalarProdInEmbeddingSpace-scalarProdInTangentSpace));
	
	TTTensor change2 = TTTensor::random(stateDims, stateRank);
	TTTangentVector tangentChange2(X, change2);
	
	scalarProdInTangentSpace = tangentChange1.scalar_product(tangentChange2);
	scalarProdInEmbeddingSpace = value_t(TTTensor(tangentChange1)(j&0) * TTTensor(tangentChange2)(j&0));
	MTEST(misc::approx_equal(scalarProdInEmbeddingSpace, scalarProdInTangentSpace, 1e-10), 
		  "scalarProd " << scalarProdInEmbeddingSpace << " " << scalarProdInTangentSpace << " diff " << (scalarProdInEmbeddingSpace-scalarProdInTangentSpace));
	
	// test whether copy assignment, += and * work as intended
	tangentChange2 = tangentChange1;
	tangentChange2 += tangentChange1 * (-1);
	for (size_t i=0; i<stateDims.size(); ++i) {
		TEST(frob_norm(tangentChange2.components[i]) < 1e-14);
	}
	
	// projection P*P = P
	tangentChange2 = TTTangentVector(X, TTTensor(tangentChange1));
	MTEST((tangentChange2.frob_norm() - tangentChange1.frob_norm())/tangentChange1.frob_norm() < 1e-14, "PP " << tangentChange2.frob_norm() - tangentChange1.frob_norm()/tangentChange1.frob_norm());
	MTEST((frob_norm(TTTensor(tangentChange1) - TTTensor(tangentChange2)))/tangentChange1.frob_norm() < 1e-14, "PP2 " << (frob_norm(TTTensor(tangentChange1) - TTTensor(tangentChange2)))/tangentChange1.frob_norm());
	
	// tangent space of 10*X should be equal to tangent space of X
	tangentChange2 = TTTangentVector(10 * X, change);
	MTEST((frob_norm(TTTensor(tangentChange1) - TTTensor(tangentChange2)))/tangentChange1.frob_norm() < 1e-14, "10X " << (frob_norm(TTTensor(tangentChange1) - TTTensor(tangentChange2)))/tangentChange1.frob_norm());
});


static misc::UnitTest alg_vecTrans("Algorithm", "vectorTransport", [](){
	std::vector<size_t> stateDims(8,4);
	std::vector<size_t> stateRank(7,2);
	Index j;
	
	TTTensor X = TTTensor::random(stateDims, stateRank);
	TTTensor change = TTTensor::random(stateDims, stateRank);
	TTTangentVector tangentChange1(X, change);
	TTTangentVector tangentChange2(tangentChange1);
	
	// vector transport from X to X should not change the vector at all
	ProjectiveVectorTransport(X, tangentChange2);
	tangentChange2 -= tangentChange1;
	for (size_t i=0; i<stateDims.size(); ++i) {
		MTEST(frob_norm(tangentChange2.components[i]) < 1e-13, frob_norm(tangentChange2.components[i]));
	}
	
	value_t normOld = tangentChange1.frob_norm();
	for (value_t eps = 1e-10; eps <= 1e-4; eps *= 10) {
		TTTensor newX(X);
		SubmanifoldRetractionI(newX, tangentChange1 * eps);
		tangentChange2 = tangentChange1;
		ProjectiveVectorTransport(newX, tangentChange2);
		value_t normNew = tangentChange2.frob_norm();
		value_t changeAngle = (value_t(TTTensor(tangentChange1)(j&0) * TTTensor(tangentChange2)(j&0)) / normOld / normNew);
		MTEST(1 - std::abs(changeAngle) < std::max(eps*eps, 1e-15), eps << " 1-angle: " << (1-changeAngle));
		MTEST(normNew <= normOld + 1e-13, eps << " norm: " << normNew << " vs " << normOld << " diff " << normNew-normOld);
	}
});


