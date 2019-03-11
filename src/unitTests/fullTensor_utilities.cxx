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

static misc::UnitTest tensor_remove_slate("Tensor", "remove_slate", [](){
	double n = 0.0;
	Tensor A({3,3,3}, [&](const std::vector<size_t> &){ n += 1.0; return n; } );

	TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18, 19,20,21,22,23,24,25,26,27}, 1e-14));

	A.remove_slate(0,1);
	TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,9, 19,20,21,22,23,24,25,26,27}, 1e-14));

	A.resize_mode(0,3,1);
	TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,9, 0,0,0,0,0,0,0,0,0, 19,20,21,22,23,24,25,26,27}, 1e-14));

	A.remove_slate(1,0);
	TEST(approx_entrywise_equal(A, {4,5,6,7,8,9, 0,0,0,0,0,0, 22,23,24,25,26,27}, 1e-14));

	A.resize_mode(1,3,1);
	TEST(approx_entrywise_equal(A, {4,5,6,0,0,0,7,8,9, 0,0,0,0,0,0,0,0,0, 22,23,24,0,0,0,25,26,27}, 1e-14));
});

static misc::UnitTest tensor_fix_mode("Tensor", "fix_mode", [](){
	double n = 0.0;
	Tensor A({3,3,3}, [&](const std::vector<size_t> &){ n += 1.0; return n; } );

	TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,9, 10,11,12,13,14,15,16,17,18, 19,20,21,22,23,24,25,26,27}, 1e-14));

	A.fix_mode(0,1);
	TEST(approx_entrywise_equal(A, {10,11,12,13,14,15,16,17,18}, 1e-14));

	A.fix_mode(1,2);
	TEST(approx_entrywise_equal(A, {12,15,18}, 1e-14));
});


static misc::UnitTest tensor_dim_reduc("Tensor", "dimension_reduction", [](){
    Tensor A({2,2,2});
    A[{0,0,0}] = 1;
    A[{0,0,1}] = 2;
    A[{0,1,0}] = 3;
    A[{0,1,1}] = 4;
    A[{1,0,0}] = 5;
    A[{1,0,1}] = 6;
    A[{1,1,0}] = 7;
    A[{1,1,1}] = 8;
    
    Tensor B = A;
    Tensor C;
    C = A;
    
    A.resize_mode(0,1);
    TEST(approx_entrywise_equal(A, {1,2,3,4}, 1e-13));
    TEST(A.dimensions[0] == 1);
    TEST(A.size == 4);
    
    B.resize_mode(1,1);
    TEST(approx_entrywise_equal(B, {1,2,5,6}, 1e-13));
    TEST(B.dimensions[1] == 1);
    TEST(B.size == 4);
    
    C.resize_mode(2,1);
    TEST(approx_entrywise_equal(C, {1,3,5,7}, 1e-13));
    TEST(C.dimensions[2] == 1);
    TEST(C.size == 4);
});

static misc::UnitTest tensor_dim_exp("Tensor", "dimension_expansion", [](){
    Tensor A({2,2,2});
    A[{0,0,0}] = 1;
    A[{0,0,1}] = 2;
    A[{0,1,0}] = 3;
    A[{0,1,1}] = 4;
    A[{1,0,0}] = 5;
    A[{1,0,1}] = 6;
    A[{1,1,0}] = 7;
    A[{1,1,1}] = 8;
    
    Tensor B = A;
    Tensor C;
    C = A;
    
    A.resize_mode(0,3);
    TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,0,0,0,0}, 1e-13));
    TEST(A.dimensions[0] == 3);
    TEST(A.size == 12);
    
    B.resize_mode(1,3);
    TEST(approx_entrywise_equal(B, {1,2,3,4,0,0,5,6,7,8,0,0}, 1e-13));
    TEST(B.dimensions[1] == 3);
    TEST(B.size == 12);
    
    C.resize_mode(2,3);
    TEST(approx_entrywise_equal(C, {1,2,0,3,4,0,5,6,0,7,8,0}, 1e-13));
    TEST(C.dimensions[2] == 3);
    TEST(C.size == 12);
});

static misc::UnitTest tensor_modify_elem("Tensor", "modify_entries", [](){
    Tensor A({4,4});
    Tensor C({2,8});
    
    A[{0,0}] = 1;
    A[{0,1}] = 2;
    A[{0,2}] = 3;
    A[{0,3}] = 4;
    A[{1,0}] = 5;
    A[{1,1}] = 6;
    A[{1,2}] = 7;
    A[{1,3}] = 8;
    A[{2,0}] = 9;
    A[{2,1}] = 10;
    A[{2,2}] = 11;
    A[{2,3}] = 12;
    A[{3,0}] = 13;
    A[{3,1}] = 14;
    A[{3,2}] = 15;
    A[{3,3}] = 16;
    
    A.modify_diagonal_entries([](value_t& _entry){});
    TEST(approx_entrywise_equal(A, {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}));
    
    A.modify_diagonal_entries([](value_t& _entry){_entry = 73.5*_entry;});
    TEST(approx_entrywise_equal(A, {73.5*1,2,3,4,5,73.5*6,7,8,9,10,73.5*11,12,13,14,15,73.5*16}));
    
    A.modify_diagonal_entries([](value_t& _entry, const size_t _position){_entry = 73.5*_entry - static_cast<value_t>(_position);});
    TEST(approx_entrywise_equal(A, {73.5*73.5*1,2,3,4,5,73.5*73.5*6-1.0,7,8,9,10,73.5*73.5*11-2.0,12,13,14,15,73.5*73.5*16-3.0}));
    
    A.reinterpret_dimensions({2,8});
    
    A.modify_diagonal_entries([](value_t& _entry){_entry = 0;});
    TEST(approx_entrywise_equal(A, {0,2,3,4,5,73.5*73.5*6-1.0,7,8,9,0,73.5*73.5*11-2.0,12,13,14,15,73.5*73.5*16-3.0}));
});


