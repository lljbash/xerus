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

#include <cstring>

using namespace xerus;

static misc::UnitTest tensor_prod0("Tensor", "Product_Order_0", [](){
    Tensor A({});
    Tensor B({});
    Tensor res1({});
    
    A[{}] = 42;
    B[{}] = 73;
    
    res1() = A() * B();
    TEST(approx_entrywise_equal(res1, {42*73}));
});

static misc::UnitTest tensor_prod1("Tensor", "Product_Order_1", [](){
    Tensor A({2});
    Tensor B({2});
    Tensor C({3});
    Tensor res1({2,2});
    Tensor res2({});
    Tensor res3({2,3});
    Tensor res4({3,2});
    
    Index i,j;
    
    A[0] = 1;
    A[1] = 2;
    
    B[0] = 3;
    B[1] = 4;
    
    C[0] = 5;
    C[1] = 6;
    C[2] = 7;
    
    // Same dimensions
    // Contraction with no index being contracted
    res1(i,j) = A(i) * B(j);
    TEST(approx_entrywise_equal(res1, {3,4,6,8}));
    res1(j,i) = A(j) * B(i);
    TEST(approx_entrywise_equal(res1, {3,4,6,8}));
    res1(i,j) = A(j) * B(i);
    TEST(approx_entrywise_equal(res1, {3,6,4,8}));
    res1(j,i) = A(i) * B(j);
    TEST(approx_entrywise_equal(res1, {3,6,4,8}));
    
    // Contraction with one index being contracted
    res2() = A(i) * B(i);
    TEST(approx_entrywise_equal(res2, {11}));
    
    // Different dimensions
    // Contraction with no index being contracted
    res3(i,j) = A(i) * C(j);
    TEST(approx_entrywise_equal(res3, {5,6,7,10,12,14}));
    res3(j,i) = A(j) * C(i);
    TEST(approx_entrywise_equal(res3, {5,6,7,10,12,14}));
    res4(i,j) = C(i) * A(j);
    TEST(approx_entrywise_equal(res4, {5,10,6,12,7,14}));
    res4(j,i) = C(j) * A(i);
    TEST(approx_entrywise_equal(res4, {5,10,6,12,7,14}));
    
    res4(i,j) = A(j) * C(i);
    TEST(approx_entrywise_equal(res4, {5,10,6,12,7,14}));
    res4(j,i) = A(i) * C(j);
    TEST(approx_entrywise_equal(res4, {5,10,6,12,7,14}));
    res3(i,j) = C(j) * A(i);
    TEST(approx_entrywise_equal(res3, {5,6,7,10,12,14}));
    res3(j,i) = C(i) * A(j);
    TEST(approx_entrywise_equal(res3, {5,6,7,10,12,14}));
});

static misc::UnitTest tensor_prod2_same("Tensor", "Product_Order_2_Same_Dimensions", [](){
    Tensor A({2,2});
    Tensor B({2,2});
    Tensor res1({2,2,2,2});
    Tensor res2({2,2});
    Tensor res3({});

    Index i,j,k,l;
    
    A[{0,0}]=1;
    A[{0,1}]=2;
    A[{1,0}]=3;
    A[{1,1}]=4;
    
    B[{0,0}]=5;
    B[{0,1}]=6;
    B[{1,0}]=7;
    B[{1,1}]=8;
    
    //Most possible contractions with no index being contracted
    //Switch pairs (i,j) and (k,l)
    res1(i,j,k,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,10,12,14,16,15,18,21,24,20,24,28,32}));
    res1(k,l,i,j) = A(k,l) * B(i,j);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,10,12,14,16,15,18,21,24,20,24,28,32}));
    res1(i,j,k,l) = A(k,l) * B(i,j);
    TEST(approx_entrywise_equal(res1, {5,10,15,20,6,12,18,24,7,14,21,28,8,16,24,32}));
    res1(k,l,i,j) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,10,15,20,6,12,18,24,7,14,21,28,8,16,24,32}));
    
    //Switch i/j or k/l
    res1(i,j,k,l) = A(j,i) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,15,18,21,24,10,12,14,16,20,24,28,32}));
    res1(j,i,k,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,15,18,21,24,10,12,14,16,20,24,28,32}));
    res1(i,j,k,l) = A(i,j) * B(l,k);
    TEST(approx_entrywise_equal(res1, {5,7,6,8,10,14,12,16,15,21,18,24,20,28,24,32}));
    res1(i,j,l,k) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,7,6,8,10,14,12,16,15,21,18,24,20,28,24,32}));
    
    //Switch i/l or j/k
    res1(i,j,k,l) = A(l,j) * B(k,i);
    TEST(approx_entrywise_equal(res1, {5,15,7,21,10,20,14,28,6,18,8,24,12,24,16,32}));
    res1(l,j,k,i) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,15,7,21,10,20,14,28,6,18,8,24,12,24,16,32}));
    res1(i,j,k,l) = A(i,k) * B(j,l);
    TEST(approx_entrywise_equal(res1, {5,6,10,12,7,8,14,16,15,18,20,24,21,24,28,32}));
    res1(i,k,j,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,6,10,12,7,8,14,16,15,18,20,24,21,24,28,32}));
    
    //Switch i/k or j/l
    res1(i,j,k,l) = A(k,j) * B(i,l);
    TEST(approx_entrywise_equal(res1, {5,6,15,18,10,12,20,24,7,8,21,24,14,16,28,32}));
    res1(k,j,i,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,6,15,18,10,12,20,24,7,8,21,24,14,16,28,32}));
    res1(i,j,k,l) = A(i,l) * B(k,j);
    TEST(approx_entrywise_equal(res1, {5,10,7,14,6,12,8,16,15,20,21,28,18,24,24,32}));
    res1(i,l,k,j) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {5,10,7,14,6,12,8,16,15,20,21,28,18,24,24,32}));
    
    //Switch pairs using multi indices
    res1(i^2, j^2) = A(i^2) * B(j^2);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,10,12,14,16,15,18,21,24,20,24,28,32}));
    res1(j^2, i^2) = A(j^2) * B(i^2);
    TEST(approx_entrywise_equal(res1, {5,6,7,8,10,12,14,16,15,18,21,24,20,24,28,32}));
    res1(i^2, j^2) = A(j^2) * B(i^2);
    TEST(approx_entrywise_equal(res1, {5,10,15,20,6,12,18,24,7,14,21,28,8,16,24,32}));
    res1(j^2, i^2) = A(i^2) * B(j^2);
    TEST(approx_entrywise_equal(res1, {5,10,15,20,6,12,18,24,7,14,21,28,8,16,24,32}));
    
    
    //All possible contractions with one index being contracted
    res2(i, k) = A(i, j) * B(j, k);
    TEST(approx_entrywise_equal(res2, {19,22,43,50}));
    res2(i, k) = A(j, i) * B(j, k);
    TEST(approx_entrywise_equal(res2, {26,30,38,44}));
    res2(i, k) = A(i, j) * B(k, j);
    TEST(approx_entrywise_equal(res2, {17,23,39,53}));
    res2(i, k) = A(j, i) * B(k, j);
    TEST(approx_entrywise_equal(res2, {23,31,34,46}));
    
    res2(k, i) = A(i, j) * B(j, k);
    TEST(approx_entrywise_equal(res2, {19,43,22,50}));
    res2(k, i) = A(j, i) * B(j, k);
    TEST(approx_entrywise_equal(res2, {26,38,30,44}));
    res2(k, i) = A(i, j) * B(k, j);
    TEST(approx_entrywise_equal(res2, {17,39,23,53}));
    res2(k, i) = A(j, i) * B(k, j);
    TEST(approx_entrywise_equal(res2, {23,34,31,46}));
    
    //All possible contractions with two indices being contracted
    res3() = A(i, j) * B(i, j);
    TEST(approx_entrywise_equal(res3, {70}));
    res3() = A(i, j) * B(j, i);
    TEST(approx_entrywise_equal(res3, {69}));
    res3() = A(j, i) * B(i, j);
    TEST(approx_entrywise_equal(res3, {69}));
    res3() = A(i^2) * B(i^2);
    TEST(approx_entrywise_equal(res3, {70}));
});

static misc::UnitTest tensor_prod2_diff("Tensor", "Product_Order_2_Different_Dimensions", [](){
    Tensor A({1,2});
    Tensor B({2,3});
    Tensor res1({1,2,2,3});
    Tensor res2({1,3});
    Tensor res3({3,1});

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
    res1(i,j,k,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {3,4,5,6,7,8,6,8,10,12,14,16}));
    res1(i,j,k,l) = A(i,k) * B(j,l);
    TEST(approx_entrywise_equal(res1, {3,4,5,6,8,10,6,7,8,12,14,16}));
    res1(i,k,j,l) = A(i,j) * B(k,l);
    TEST(approx_entrywise_equal(res1, {3,4,5,6,8,10,6,7,8,12,14,16}));
    
    //One Index contracted
    res2(i,k) = A(i,j) * B(j,k);
    TEST(approx_entrywise_equal(res2, {15,18,21}));
    res3(k,i) = A(i,j) * B(j,k);
    TEST(approx_entrywise_equal(res2, {15,18,21}));
    res2(i,k) = B(j,k) * A(i,j);
    TEST(approx_entrywise_equal(res2, {15,18,21}));
    res3(k,i) = B(j,k) * A(i,j);
    TEST(approx_entrywise_equal(res2, {15,18,21}));
});

static misc::UnitTest tensor_prod3_same("Tensor", "Product_Order_3_Same_Dimensions", [](){
    Tensor A({2,2,2});
    Tensor B({2,2,2});
    Tensor res2({2,2});
    Tensor res2a({2});
    Tensor res3({});

    Index i,j,k,l,m;
    
    A[{0,0,0}]=1;
    A[{0,1,0}]=2;
    A[{1,0,0}]=3;
    A[{1,1,0}]=4;
    A[{0,0,1}]=5;
    A[{0,1,1}]=6;
    A[{1,0,1}]=7;
    A[{1,1,1}]=8;
    
    B[{0,0,0}]=5;
    B[{0,0,1}]=9;
    B[{0,1,0}]=6;
    B[{0,1,1}]=10;
    B[{1,0,0}]=7;
    B[{1,0,1}]=11;
    B[{1,1,0}]=8;
    B[{1,1,1}]=12;
    
    res2(i,j) = A(l,m,j) * B(m,i,l);
    TEST(approx_entrywise_equal(res2, {5+2*7+3*9+4*11, 5*5+6*7+7*9+8*11, 6+2*8+3*10+4*12, 5*6+6*8+7*10+8*12}));
    
    res2a(i) = A(l,m,0) * B(m,i,l);
    TEST(approx_entrywise_equal(res2a, {5+2*7+3*9+4*11, 6+2*8+3*10+4*12}));
    
    res2a(j) = A(l,m,j)* B(m,1,l);
    TEST(approx_entrywise_equal(res2a, {6+2*8+3*10+4*12, 5*6+6*8+7*10+8*12}));
    
    res3() = A(i,j,k) * B(i,k,j);
    TEST(misc::approx_equal(res3[0], 5.0+5*6+2*9+6*10+3*7+7*8+4*11+8*12, 1e-13));
});

static misc::UnitTest tensor_prod_multi("Tensor", "Product_Multiindices", [](){
	Tensor res({2,2,2});
    Tensor res2({2,2});
    Tensor res4({2});
    Tensor res3({});
    Tensor B({2,2,2});
    Tensor C({2,2,2});
    

    Index i,j,k,l,m;
    
    B[{0,0,0}]=1;
    B[{0,1,0}]=2;
    B[{1,0,0}]=3;
    B[{1,1,0}]=4;
    B[{0,0,1}]=5;
    B[{0,1,1}]=6;
    B[{1,0,1}]=7;
    B[{1,1,1}]=8;
    
    C[{0,0,0}]=5;
    C[{0,0,1}]=9;
    C[{0,1,0}]=6;
    C[{0,1,1}]=10;
    C[{1,0,0}]=7;
    C[{1,0,1}]=11;
    C[{1,1,0}]=8; 
    C[{1,1,1}]=12;
    
    res2(i,j, k&2) = B(l,m,j, k^0) * C(m,i,l, k&3);
    TEST(approx_entrywise_equal(res2, {5+2*7+3*9+4*11, 5*5+6*7+7*9+8*11, 6+2*8+3*10+4*12, 5*6+6*8+7*10+8*12}));
    res3(j&0) = B(i&0) * C(i^3);
    TEST(misc::approx_equal(res3[0],5.0+2*6+3*7+4*8+5*9+6*10+7*11+8*12 , 1e-13));
    res3(j^0) = B(i&0) * C(i^3);
    TEST(misc::approx_equal(res3[0],5.0+2*6+3*7+4*8+5*9+6*10+7*11+8*12 , 1e-13));
});

static misc::UnitTest tensor_prod_three("Tensor", "Product_Threefold",  [](){
    Tensor res({2,2});
    Tensor B({2,2});
    Tensor C({2,2});
    Tensor D({2,2});

    Index i, j, k, l;
    
    B[{0,0}]=1;
    B[{0,1}]=2;
    B[{1,0}]=3;
    B[{1,1}]=4;
    
    C[{0,0}]=5;
    C[{0,1}]=6;
    C[{1,0}]=7;
    C[{1,1}]=8;
    
    D[{0,0}]=9;
    D[{0,1}]=10;
    D[{1,0}]=11;
    D[{1,1}]=12;
    
    res(i,l) = B(i,j) * C(j,k) * D(k,l);
    TEST(approx_entrywise_equal(res, {413, 454, 937, 1030}));
    res(i,l) = B(i,j) * C(k,l) * D(j,k);
    TEST(approx_entrywise_equal(res, {393, 458, 901, 1050}));
    res(i,l) = B(k,l) * C(i,j) * D(j,k);
    TEST(approx_entrywise_equal(res, {477, 710, 649, 966}));
    res(i,l) = B(j,k) * C(k,l) * D(i,j);
    TEST(approx_entrywise_equal(res, {601, 698, 725, 842}));
});

static misc::UnitTest tensor_prod_many_2("Tensor", "Product_Many_Degree_2", [](){
    Tensor A({2,2});
    Tensor B({2,2});
    Tensor C({2,2});
    Tensor D({2,2});
    Tensor E({2,2});
    Tensor F({2,2});
    Tensor G({2,2});
    Tensor res1({2,2});

    Index i,j,k,l,m,n,o;
    
    A[{0,0}]=1;
    A[{0,1}]=2;
    A[{1,0}]=3;
    A[{1,1}]=4;
    
    B[{0,0}]=5;
    B[{0,1}]=6;
    B[{1,0}]=7;
    B[{1,1}]=8;
    
    C[{0,0}]=9;
    C[{0,1}]=10;
    C[{1,0}]=11;
    C[{1,1}]=12;
    
    D[{0,0}]=13;
    D[{0,1}]=14;
    D[{1,0}]=15;
    D[{1,1}]=16;
    
    E[{0,0}]=17;
    E[{0,1}]=18;
    E[{1,0}]=19;
    E[{1,1}]=20;
    
    F[{0,0}]=21;
    F[{0,1}]=22;
    F[{1,0}]=23;
    F[{1,1}]=24;
    
    res1(i,o) = A(i,j) * B(j,k) * C(k,l) * D(l,m) * E(m,n) * F(n,o); 
    TEST(approx_entrywise_equal(res1, {20596523, 21531582, 46728183, 48849590}));
    res1(i,o) = (A(i,j) * B(j,k)) * (C(k,l) * (D(l,m) * E(m,n)) * F(n,o)); 
    TEST(approx_entrywise_equal(res1, {20596523, 21531582, 46728183, 48849590}));
    res1(i,o) = E(m,n) * A(i,j) * F(n,o) * C(k,l) * B(j,k) * D(l,m); 
    TEST(approx_entrywise_equal(res1, {20596523, 21531582, 46728183, 48849590}));
    res1(i,o) = E(m,n) * (B(j,k) * A(i,j) * F(n,o)) * (C(k,l) * D(l,m)); 
    TEST(approx_entrywise_equal(res1, {20596523, 21531582, 46728183, 48849590}));
});

static misc::UnitTest tensor_prod1000("Tensor", "Product_1000x1000", [](){
	Tensor res({1000,1000});
    Tensor A({1000,1000}, [] (const std::vector<size_t> &_idx) { return double(_idx[0] + _idx[1]); });
    Tensor B({1000,1000}, [] (const std::vector<size_t> &_idx) { return double((1000-_idx[0]) * (1000-_idx[1])); });
    Tensor C({1000,1000}, Tensor::Representation::Dense);
    Index i, j, k;

    blasWrapper::matrix_matrix_product(C.get_dense_data(), 1000, 1000, 1.0, A.get_dense_data(), false, 1000, B.get_dense_data(), false);
    res(i,k) = A(i,j) * B(j,k);
    MTEST(memcmp(res.get_dense_data(), C.get_dense_data(), sizeof(value_t)*1000*1000)==0, "Error: " << frob_norm(res-C));
    
    blasWrapper::matrix_matrix_product(C.get_dense_data(), 1000, 1000, 1.0, A.get_dense_data(), false, 1000, B.get_dense_data(), true);
    res(i,k) = A(i,j) * B(k,j);
    TEST(memcmp(res.get_dense_data(), C.get_dense_data(), sizeof(value_t)*1000*1000)==0);
    
    blasWrapper::matrix_matrix_product(C.get_dense_data(), 1000, 1000, 1.0, A.get_dense_data(), true, 1000, B.get_dense_data(), true);
    res(i,k) = A(j,i) * B(k,j);
    TEST(memcmp(res.get_dense_data(), C.get_dense_data(), sizeof(value_t)*1000*1000)==0);
});



static misc::UnitTest dyadic_factor_product("Tensor", "Dyadic_factor_product", []() {
	Index i,j;
	Tensor s = Tensor::ones({10})/10.0;
	
	Tensor s1; s1(i,j) = s(i)*s(j);
	
	for(size_t k = 0; k < 10; ++k) {
		for(size_t l = 0; l < 10; ++l) {
			TEST(misc::approx_equal(s1[{k,l}], 0.01));
		}
	}
});
