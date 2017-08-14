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
using namespace xerus;

static misc::UnitTest tensor_svd_id("Tensor", "SVD_Identity", [](){
    Tensor A({2,2,2,2});
    Tensor res1({2,2,4});
    Tensor res2({4,4});
    Tensor res3({4,2,2});
    
    A[{0,0,0,0}] = 1;
    A[{0,1,0,1}] = 1;
    A[{1,0,1,0}] = 1;
    A[{1,1,1,1}] = 1;    
    
    Index i, j, k, l, m, n;
    
    (res1(i,j,m), res2(m,n), res3(n,k,l)) = SVD(A(i,j,k,l));
	res2.reinterpret_dimensions({2,2,2,2});
	TEST(approx_entrywise_equal(res2, A));
	res2.reinterpret_dimensions({4,4});
	MTEST(frob_norm(res1(i^2, m)*res1(i^2, n) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(m, i^2)*res3(n, i^2) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
	
	(res1(m,i,j), res2(n,m), res3(k,n,l)) = SVD(A(i,j,k,l));
	res2.reinterpret_dimensions({2,2,2,2});
	TEST(approx_entrywise_equal(res2, A));
	res2.reinterpret_dimensions({4,4});
	MTEST(frob_norm(res1(m, i^2)*res1(n, i^2) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(k,m,l)*res3(k,n,l) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
});

static misc::UnitTest tensor_svd_zero("Tensor", "SVD_zero", [](){
    Tensor A({2,2,2,2});
    Tensor res1({2,2,4});
    Tensor res2({4,4});
    Tensor res3({4,2,2});
    
    Index i, j, k, l, m, n;
    
    (res1(i,j,m), res2(m,n), res3(n,k,l)) = SVD(A(i,j,k,l));
	MTEST(res2.dimensions[0] == 1, res2.dimensions[0]);
	MTEST(std::abs(res2[0]) < 1e-15, res2[0]);
	MTEST(frob_norm(res1(i^2, m)*res1(i^2, n) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(m, i^2)*res3(n, i^2) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
});

static misc::UnitTest tensor_svd_rnd512("Tensor", "SVD_Random_512x512", [](){
    Tensor A = Tensor::random({8,8,8,8,8,8});
    Tensor res1;
    Tensor res2;
    Tensor res3;
    Tensor res4;
     
    
    Index i, j, k, l, m, n, o, p, r, s;
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(A(i,j,k,l,m,n));
    res4(i,j,k,l,m,n) = res1(i,j,k,o)*res2(o,p)*res3(p,l,m,n);
    TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(i^3, m)*res1(i^3, n) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(m, i^3)*res3(n, i^3) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(A(l,k,m,i,j,n));
    res4(l,k,m,i,j,n) =  res1(i,j,k,o)*res2(o,p)*res3(p,l,m,n);
    TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(i^3, m)*res1(i^3, n) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(m, i^3)*res3(n, i^3) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(A(l,i,m,k,j,n));
    res4(k,i,m,l,j,n) =  res1(i,j,l,o)*res2(o,p)*res3(p,k,m,n);
    TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(i^3, m)*res1(i^3, n) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(m, i^3)*res3(n, i^3) - Tensor::identity(res2.dimensions)(m, n)) < 1e-12, " Vt not orthogonal");
	
	(res1(i,o,k,j), res2(p,o), res3(l,n,m,p)) = SVD(A(l,i,m,k,j,n));
	res4(l,k,m,i,j,n) =  res1(k,o,i,j)*res2(p,o)*res3(l,n,m,p);
	TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(k,o,i,j)*res1(k,p,i,j) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(l,n,m,o)*res3(l,n,m,p) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " Vt not orthogonal");
	
	
	(res1(i,o,k,j), res2(p,o), res3(l,n,m,p)) = SVD(-1*A(l,i,m,k,j,n));
	res4(l,k,m,i,j,n) =  res1(k,o,i,j)*res2(p,o)*res3(l,n,m,p);
	TEST(approx_equal(-1*res4, A, 1e-13));
	MTEST(frob_norm(res1(k,o,i,j)*res1(k,p,i,j) - Tensor::identity(res2.dimensions)(o, p)) < 1e-11, " U not orthogonal " << frob_norm(res1(k,o,i,j)*res1(k,p,i,j) - Tensor::identity(res2.dimensions)(o, p)));
	MTEST(frob_norm(res3(l,n,m,o)*res3(l,n,m,p) - Tensor::identity(res2.dimensions)(o, p)) < 1e-11, " Vt not orthogonal " << frob_norm(res3(l,n,m,o)*res3(l,n,m,p) - Tensor::identity(res2.dimensions)(o, p)));
});

static misc::UnitTest tensor_svd_soft("Tensor", "SVD_soft_thresholding", [](){
    Tensor A = 10*Tensor::random({3,5,2,7,3,12});
    Tensor Ax, U, V, Us, Vs;
    Tensor S(Tensor::Tensor::Representation::Sparse);
	Tensor Ss(Tensor::Tensor::Representation::Sparse);
    
    Index i, j, k, l, m, n, o, p, r, s;
    
    (U(i,j,k,o), S(o,p), V(p,l,m,n)) = SVD(A(i,j,k,l,m,n));
    (Us(i,j,k,o), Ss(o,p), Vs(p,l,m,n)) = SVD(A(i,j,k,l,m,n), 7.3);
	
	U.resize_mode(U.degree()-1, Ss.dimensions[0]);
	V.resize_mode(0, Ss.dimensions[0]);
	
    TEST(approx_equal(U, Us, 1e-12));
    TEST(approx_equal(V, Vs, 1e-12));
	
	for(size_t x = 0; x < S.dimensions[0]; ++x) {
		if(x < Ss.dimensions[0]) {
			TEST(misc::approx_equal(Ss[{x,x}], std::max(0.0, S[{x, x}]-7.3), 3e-13));
		} else {
			TEST(S[{x,x}] <= 7.3);
		}
	}
	
	Ax(i,j,k,l,m,n) = U(i,j,k,o)* S(o,p)* V(p,l,m,n);
	TEST(approx_equal(A, Ax, 1e-12));
	MTEST(frob_norm(U(i,j,k,o)*U(i,j,k,p) - Tensor::identity(S.dimensions)(o, p)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(V(o,l,m,n)*V(p,l,m,n) - Tensor::identity(S.dimensions)(o, p)) < 1e-12, " Vt not orthogonal");
});

static misc::UnitTest tensor_svd_order_6("Tensor", "SVD_Random_Order_Six", [](){
    Tensor A = Tensor::random({9,7,5,5,9,7});
    Tensor res1;
    Tensor res2;
    Tensor res3;
    Tensor res4;
     
    
    Index i, j, k, l, m, n, o, p, r, s;
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(A(i,j,k,l,m,n));
    res4(i,j,k,l,m,n) = res1(i,j,k,o)*res2(o,p)*res3(p,l,m,n);
    TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(i,j,k,o)*res1(i,j,k,p) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(o,l,m,n)*res3(p,l,m,n) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " Vt not orthogonal");
    
    (res1(i,j,k,o), res2(o,p), res3(p,l,m,n)) = SVD(A(m,j,k,l,i,n));
    res4(m,j,k,l,i,n) = res1(i,j,k,o)*res2(o,p)*res3(p,l,m,n);
    TEST(approx_equal(res4, A, 1e-14));
	MTEST(frob_norm(res1(i,j,k,o)*res1(i,j,k,p) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " U not orthogonal");
	MTEST(frob_norm(res3(o,l,m,n)*res3(p,l,m,n) - Tensor::identity(res2.dimensions)(o, p)) < 1e-12, " Vt not orthogonal");
});

static misc::UnitTest tensor_qr_rq_rnd6("Tensor", "QR_AND_RQ_Random_Order_Six", [](){
    Tensor A = Tensor::random({7,5,9,7,5,9});
    Tensor Q;
    Tensor R;
    Tensor Q2;
    Tensor R2;
    Tensor Q3;
    Tensor R3;
    Tensor Q4;
    Tensor res4;
    
    Index i, j, k, l, m, n, o, p, q, r;

    
    (Q(i,j,k,l), R(l,m,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,j,k,o)*R(o,m,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "1 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
	
	res4(l,m) = Q(i,j,k,l) * Q(i,j,k,m);
	res4.modify_diagonal_entries([](value_t &entry){entry -= 1;});
	TEST(frob_norm(res4) < 1e-12);
    
    (Q(i,j,k,l), R(l,m,n,r)) = QR(A(i,n,k,m,j,r));
    res4(i,n,k,m,j,r) = Q(i,j,k,o)*R(o,m,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "2 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
    
    (Q2(i,k,l), R2(l,m,j,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q2(i,k,o)*R2(o,m,j,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "3 " << frob_norm(res4-A) << " / " << frob_norm(A));
    
    (Q3(i,m,j,k,l), R3(l,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q3(i,m,j,k,o)*R3(o,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "4 " << frob_norm(res4-A) << " / " << frob_norm(A));
	
	(Q(i,l,j,k,m), R(l,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,o,j,k,m)*R(o,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "5 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(i,l,j,k,m)*Q(i,q,j,k,m) - Tensor::identity({Q.dimensions[1], Q.dimensions[1]})(l, q)) < 1e-12, " Q not orthogonal");
    
    
    (R(i,j,k,l), Q(l,m,n,r)) = RQ(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = R(i,j,k,o)*Q(o,m,n,r);
    MTEST(approx_equal(res4, A, 5e-15), "6 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(p,m,n,r)*Q(q,m,n,r) - Tensor::identity({Q.dimensions[0], Q.dimensions[0]})(p, q)) < 1e-12, " Q not orthogonal");
    
    (R(i,j,k,l), Q(l,m,n,r)) = RQ(A(i,n,k,m,j,r));
    res4(i,n,k,m,j,r) = R(i,j,k,o)*Q(o,m,n,r);
    MTEST(approx_equal(res4, A, 5e-15), "7 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(p,m,n,r)*Q(q,m,n,r) - Tensor::identity({Q.dimensions[0], Q.dimensions[0]})(p, q)) < 1e-12, " Q not orthogonal");
    
    (R2(i,m,j,k,l), Q2(l,n,r)) = RQ(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = R2(i,m,j,k,l)*Q2(l,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "8 " << frob_norm(res4-A) << " / " << frob_norm(A));
    
    (R3(i,k,l), Q3(l,m,j,n,r)) = RQ(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = R3(i,k,o)*Q3(o,m,j,n,r);
    MTEST(approx_equal(res4, A, 2e-15), "9 " << frob_norm(res4-A) << " / " << frob_norm(A));
	
	(R(l,i,k), Q(n,m,j,l,r)) = RQ(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = R(o,i,k)*Q(n,m,j,o,r);
    MTEST(approx_equal(res4, A, 2e-15), "10 " << frob_norm(res4-A) << " / " << frob_norm(A));
	MTEST(frob_norm(Q(n,m,j,q,r)*Q(n,m,j,p,r) - Tensor::identity({Q.dimensions[3], Q.dimensions[3]})(p, q)) < 1e-12, " Q not orthogonal");
    
    
    (R3(i,k,l), Q4(l,j,n,r)) = RQ(A(i,j,k,3,n,r));
    res4(i,j,k,n,r) = R3(i,k,o)*Q4(o,j,n,r);
    TEST(frob_norm(A(i,j,k,3,n,r) - res4(i,j,k,n,r)) < 1e-12);
	MTEST(frob_norm(Q4(q,j,n,r)*Q4(p,j,n,r) - Tensor::identity({Q4.dimensions[0], Q4.dimensions[0]})(p, q)) < 1e-12, " Q not orthogonal");
});


static misc::UnitTest tensor_qc("Tensor", "QC", [](){
    Tensor A = Tensor::random({2,2,2,2,2,2});
	Tensor B({2,3}, [](size_t i){return double(i);});
    Tensor Q;
    Tensor R;
    Tensor Q2;
    Tensor R2;
    Tensor Q3;
    Tensor R3;
    Tensor Q4;
    Tensor res4;
    
    Index i, j, k, l, m, n, o, p, q, r;
	
	(Q(i,j), R(j,k)) = QC(B(i,k));
    
    (Q(i,j,k,l), R(l,m,n,r)) = QC(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,j,k,o)*R(o,m,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
    
    (Q(i,j,k,l), R(l,m,n,r)) = QC(A(i,n,k,m,j,r));
    res4(i,n,k,m,j,r) = Q(i,j,k,o)*R(o,m,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
    
    (Q2(i,k,l), R2(l,m,j,n,r)) = QC(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q2(i,k,o)*R2(o,m,j,n,r);
    TEST(approx_equal(res4, A, 1e-12));
    
    (Q3(i,m,j,k,l), R3(l,n,r)) = QC(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q3(i,m,j,k,o)*R3(o,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	
	(Q(i,l,j,k,m), R(l,n,r)) = QC(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,o,j,k,m)*R(o,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,l,j,k,m)*Q(i,q,j,k,m) - Tensor::identity({Q.dimensions[1], Q.dimensions[1]})(l, q)) < 1e-12, " Q not orthogonal");
});

static misc::UnitTest tensor_sqr("Tensor", "Sparse_QR", [](){
    Tensor A =  Tensor::random({2,2,2,2,2,2}, 16);
	A.use_sparse_representation();
	Tensor B({2,3}, [](size_t i){return double(i);});
    Tensor Q;
    Tensor R;
    Tensor Q2;
    Tensor R2;
    Tensor Q3;
    Tensor R3;
    Tensor Q4;
    Tensor res4;
    
    Index i, j, k, l, m, n, o, p, q, r;
	
    (Q(i,j,k,l), R(l,m,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,j,k,o)*R(o,m,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
    
    (Q(i,j,k,l), R(l,m,n,r)) = QR(A(i,n,k,m,j,r));
    res4(i,n,k,m,j,r) = Q(i,j,k,o)*R(o,m,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,j,k,l)*Q(i,j,k,m) - Tensor::identity({R.dimensions[0], R.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
    
    (Q2(i,k,l), R2(l,m,j,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q2(i,k,o)*R2(o,m,j,n,r);
    TEST(approx_equal(res4, A, 1e-12));
    
    (Q3(i,m,j,k,l), R3(l,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q3(i,m,j,k,o)*R3(o,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	
	(Q(i,l,j,k,m), R(l,n,r)) = QR(A(i,j,k,m,n,r));
    res4(i,j,k,m,n,r) = Q(i,o,j,k,m)*R(o,n,r);
    TEST(approx_equal(res4, A, 1e-15));
	MTEST(frob_norm(Q(i,l,j,k,m)*Q(i,q,j,k,m) - Tensor::identity({Q.dimensions[1], Q.dimensions[1]})(l, q)) < 1e-12, " Q not orthogonal");
});


static misc::UnitTest tensor_scq("Tensor", "Sparse_CQ", [](){
    Tensor A = Tensor::random({3,3,3,3,3,3}, 16);
	A.use_sparse_representation();
	Tensor Af(A);
	Af.use_dense_representation();
	A *= 2; Af *= 2;
    Tensor Qs, Qf, Cs, Cf, res;
    Index i, j, k, l, m, n, o, p, q, r;
	
    (Cs(i,j,k,l), Qs(l,m,n,r)) = CQ(A(i,j,k,m,n,r));
	(Cf(i,j,k,l), Qf(l,m,n,r)) = CQ(Af(i,j,k,m,n,r));
    res(i,j,k,m,n,r) = Cs(i,j,k,o)*Qs(o,m,n,r);
    TEST(approx_equal(res, A, 1e-15));
// 	TEST(approx_equal(Qs, Qf, 1e-15)); // NOTE apparently not true when there are small singular values
// 	TEST(approx_equal(Cs, Cf, 1e-15));// NOTE apparently not true when there are small singular values
	MTEST(frob_norm(Qs(l,i,j,k)*Qs(m,i,j,k) - Tensor::identity({Qs.dimensions[0], Qs.dimensions[0]})(l, m)) < 1e-12, " Q not orthogonal");
});

static misc::UnitTest sparse_svd("Tensor", "SparseSVD", [](){
	Index i,j,k,l;
	size_t m1=400, m2=400;
	for (size_t n=10; n<=1000; n+=100) {
		Tensor A = Tensor::random({m1, m2}, 1*n);
		
		Tensor U, S, Vt;
// 		uint64 start = xerus::misc::uTime();
		(U(i,j), S(j,k), Vt(k,l)) = SVD(A(i,l));
// 		uint64 tSparse = xerus::misc::uTime() - start;
// 		XERUS_LOG(sparsity, U.is_sparse() << ' ' << S.is_sparse() << ' ' << Vt.is_sparse());
		Tensor T;
		T(i,l) = U(i,j)*S(j,k)*Vt(k,l);
		MTEST(approx_equal(T, A, 1e-14), n << ' ' << frob_norm(T-A) << '/' << frob_norm(A));
		U(i,j)=U(k,i)*U(k,j) - Tensor::identity(S.dimensions)(i,j);
		MTEST(frob_norm(U)<1e-10, n << ' ' << frob_norm(U));
		Vt(i,j)=Vt(i,k)*Vt(j,k) - Tensor::identity(S.dimensions)(i,j);
		MTEST(frob_norm(Vt)<1e-10, n << ' ' << frob_norm(Vt));
		
// 		A.use_dense_representation();
// 		start = xerus::misc::uTime();
// 		(U(i,j), S(j,k), Vt(k,l)) = SVD(A(i,l));
// 		uint64 tDense = xerus::misc::uTime() - start;
// 		XERUS_LOG(times, n << ' ' << tSparse << ' ' << tDense);
	}
});

