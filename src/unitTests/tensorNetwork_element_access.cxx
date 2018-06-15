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


#include <xerus.h>

#include "../../include/xerus/test/test.h"

using namespace xerus;

static misc::UnitTest tn_elem_access("TensorNetwork", "element_access", [](){
    Tensor A({1,2});
    Tensor B({2,3});
    TensorNetwork res;

    Index i,j,k,l;
    
    A[{0,0}] = 1;
    A[{0,1}] = 2;
    
    B[{0,0}] = 3;
    B[{0,1}] = 4;
    B[{0,2}] = 5;
    B[{1,0}] = 6;
    B[{1,1}] = 7;
    B[{1,2}] = 8;
    
    //No Index contracted
    res(i,j,k,l) = A(i,j) * B(k,l);
    std::vector<value_t> resX({3,4,5,6,7,8,6,8,10,12,14,16});
    for(size_t t = 0; t < misc::product(res.dimensions); ++t) {
        TEST(misc::approx_equal(res[t], resX[t]));
    }
    TEST(misc::approx_equal(res[{0,1,1,1}], 14.0));
    

    //One Index contracted
    res(k,i) = B(j,k) * A(i,j);
    
    TEST(misc::approx_equal(res[0], 15.0));
    TEST(misc::approx_equal(res[1], 18.0));
    TEST(misc::approx_equal(res[2], 21.0));
    TEST(misc::approx_equal(res[{0,0}], 15.0));
    TEST(misc::approx_equal(res[{1,0}], 18.0));
    TEST(misc::approx_equal(res[{2,0}], 21.0));
});


static misc::UnitTest tn_many_access("TensorNetwork", "many_element_access", [](){
    Tensor A({1,2});
    Tensor B({2,3});
    TensorNetwork res;

    Index i,j,k,l;
    
    A[{0,0}] = 1;
    A[{0,1}] = 2;
    
    B[{0,0}] = 3;
    B[{0,1}] = 4;
    B[{0,2}] = 5;
    B[{1,0}] = 6;
    B[{1,1}] = 7;
    B[{1,2}] = 8;
    
    //No Index contracted
    res(i,j,k,l) = A(i,j) * B(k,l);

	SinglePointMeasurementSet measurments;
	measurments.add({0,0,0,0}, 0);
	measurments.add({0,0,0,1}, 0);
	measurments.add({0,0,0,2}, 0);
	measurments.add({0,0,1,0}, 0);
	measurments.add({0,0,1,1}, 0);
	measurments.add({0,0,1,2}, 0);
	measurments.add({0,1,0,0}, 0);
	measurments.add({0,1,0,1}, 0);
	measurments.add({0,1,0,2}, 0);
	measurments.add({0,1,1,0}, 0);
	measurments.add({0,1,1,1}, 0);
	measurments.add({0,1,1,2}, 0);
	
	measurments.measure(res);
	
	for(size_t m = 0; m < measurments.size(); ++m) {
		TEST(misc::approx_equal(measurments.measuredValues[m], res[measurments.positions[m]]));
    }
});
