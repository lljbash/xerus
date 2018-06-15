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

static misc::UnitTest tensor_arith_neg("Tensor", "Arithmetic_Negatives", [](){
    Tensor A({2,2,2});
    Tensor B({2,2});
    Tensor B2({3,3});
    Tensor B3({3,2});
    Tensor B4({2,3});
    Tensor B5({2,2,2});
    Tensor C({2});
    
    Index i,j,k;

    FAILTEST(C(i) = B(i,j) * B2(j,j));
    FAILTEST(C(i) = B(i,j) * B3(j,j));
    FAILTEST(C(i) = B(i,j) * B4(j,j));
    FAILTEST(C(i) = B(i,j) * B5(j,j,j));
    FAILTEST(B(i,j) = B(i,j) + B2(j,j));
    FAILTEST(B(i,j) = B(i,j) + B3(j,j));
    FAILTEST(B(i,j) = B(i,j) + B4(j,j));
    FAILTEST(B(i,j) = B(i,j) + B5(j,j,j));
});

static misc::UnitTest tensor_triple_idx("Tensor", "triple_indices", [](){
	Tensor A;
	Tensor B;
	Tensor C;
	Tensor D;
	Tensor F;
	Tensor E0;
	Tensor E1;
	Tensor E2;
	Index i1,i2,i3,i4;
	
	FAILTEST(E0()   = A(i1,i1,i2/3)*B(i2/2,i2));
	FAILTEST(E1(i2) = A(i1,i1,i2)*B(i2,i2));
	FAILTEST(E0()   = A(i1,i2,i2)*B(i2,i1));
	FAILTEST(E1(i2) = A(i1,i2,i2)*B(i2,i1));
	FAILTEST(E0()   = A(i2,i2,i2)*B(i1,i1));
	FAILTEST(E1(i2) = A(i2,i2,i2)*B(i1,i1));
	FAILTEST(E0()   = A(i1,i2,i2)*B(i1,i3)*C(i3,i2));
// 	FAILTEST(E1(i2) = A(i1,i2,i2)*B(i1,i3)*C(i3,i2)); //FEATURE
	FAILTEST(E0()      = B(i1,i2)*C(i2,i3)*D(i3,i2));
// 	FAILTEST(E2(i1,i2) = B(i1,i2)*C(i2,i3)*D(i3,i2)); //FEATURE
	FAILTEST(E0()      = B(i1,i2)*C(i2,i3)*D(i1,i2));
// 	FAILTEST(E2(i2,i3) = B(i1,i2)*C(i2,i3)*D(i1,i2)); //FEATURE
	FAILTEST(E0()      = B(i1,i2)*C(i2,i3)*D(i3,i4)*F(i4,i2));
// 	FAILTEST(E2(i1,i2) = B(i1,i2)*C(i2,i3)*D(i3,i4)*F(i4,i2)); //FEATURE
});

    
