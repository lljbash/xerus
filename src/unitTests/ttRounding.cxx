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
#include "../../include/xerus/misc/internal.h"
using namespace xerus;

static misc::UnitTest tt_round("TT", "TTTensor_Rounding", [](){
    Index i,j,k;
    
    Tensor A1 = Tensor::random({2});
    Tensor B1;
    TTTensor TTA1(A1, 1e-14);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    TTA1.round(1e-14);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    TTA1.round(1);
    B1(i) = TTA1(i);
    TEST(approx_equal(B1,A1, 1e-14));
    
    
    Tensor A2 = Tensor::random({2,2});
    Tensor B2;
    TTTensor TTA2(A2, 1e-14);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    TTA2.round(1e-14);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    TTA2.round(2);
    B2(j,i) = TTA2(j,i);
    TEST(approx_equal(B2,A2, 1e-14));
    
    
    Tensor A3 = Tensor::random({2,7});
    Tensor B3;
    TTTensor TTA3(A3, 1e-14);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    TTA3.round(1e-14);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    TTA3.round(2);
    B3(j,i) = TTA3(j,i);
    TEST(approx_equal(B3,A3, 1e-14));
    
    
    Tensor A4 = Tensor::random({2,2,2,2,2,2,2,2});
    Tensor B4;
    TTTensor TTA4(A4, 1e-14);
    B4(j,i^7) = TTA4(j,i^7);
    TEST(approx_equal(B4,A4, 1e-14));
    
    TTA4.round(1e-14);
    B4(j,i^7) = TTA4(j,i^7);
    TEST(approx_equal(B4,A4, 1e-14));
    
    TTA4.round(512);
    B4(j,i^7) = TTA4(j,i^7);
    TEST(approx_equal(B4,A4, 1e-14));
    
    
    Tensor A5 = Tensor::random({5,6,3,1,4,2,8,1});
    Tensor B5;
    TTTensor TTA5(A5, 1e-14);
    B5(j,i^7) = TTA5(j,i^7);
    TEST(approx_equal(B5,A5, 1e-14));
    
    TTA5.round(1e-14);
    B5(j,i^7) = TTA5(j,i^7);
    TEST(approx_equal(B5,A5, 1e-14));
    
    TTA5.round(576);
    B5(j,i^7) = TTA5(j,i^7);
    TEST(approx_equal(B5,A5, 1e-14));
});


static misc::UnitTest tt_noround("TT", "no_rounding", [](){
    Index i,j,k;
	TTTensor a = TTTensor::random({2,2,2,2,2,2,2}, {2,2,2,2,2,2});
	TTTensor b(a);
	a.round(2);
	TEST(approx_equal(Tensor(a),Tensor(b)));
	TTTensor c = TTTensor::random({2,2,2,2,2,2,2}, {2,2,2,2,2,2});
	a(i&0) = a(i&0) + 0.0*c(i&0);
	LOG(unit_test, a.ranks());
	a.round(2);
	TEST(approx_equal(Tensor(a), Tensor(b), 1e-14));
});
