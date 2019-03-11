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

static misc::UnitTest sparse_arith_neg("SparseTensor", "Arithmetic_Negatives", [](){
    Tensor A({2,2,2}, Tensor::Representation::Sparse);
    Tensor B({2,2}, Tensor::Representation::Sparse);
    Tensor B2({3,3}, Tensor::Representation::Sparse);
    Tensor B3({3,2}, Tensor::Representation::Sparse);
    Tensor B4({2,3}, Tensor::Representation::Sparse);
    Tensor B5({2,2,2}, Tensor::Representation::Sparse);
    Tensor C({2}, Tensor::Representation::Sparse);
    
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

static misc::UnitTest sparse_tripleIdx("SparseTensor", "triple_indices", [](){
	Tensor A({1,1,1}, Tensor::Representation::Sparse);
	Tensor B({1,1}, Tensor::Representation::Sparse);
	Tensor C({1,1}, Tensor::Representation::Sparse);
	Tensor D({1,1}, Tensor::Representation::Sparse);
	Tensor F({1,1}, Tensor::Representation::Sparse);
	Tensor E0( Tensor::Representation::Sparse );
	Tensor E1({1}, Tensor::Representation::Sparse);
	Tensor E2({1}, Tensor::Representation::Sparse);
	Index i1,i2,i3,i4;
	
	FAILTEST(E0()   = A(i1,i1,i2)*B(i2,i2));
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
    
