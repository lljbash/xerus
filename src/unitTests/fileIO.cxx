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


static misc::UnitTest tensor_rw("Tensor", "read_write_file", [](){
	Tensor A = Tensor::random({12,13,14});
	
	misc::save_to_file(A, "test.dat", misc::FileFormat::TSV);
	Tensor Ab = misc::load_from_file<Tensor>("test.dat");
	MTEST(approx_equal(A, Ab), "dense tsv " << frob_norm(A-Ab));
	
	misc::save_to_file(A, "test.dat", misc::FileFormat::BINARY);
	Ab = misc::load_from_file<Tensor>("test.dat");
	MTEST(approx_equal(A, Ab), "dense bin " << frob_norm(A-Ab));
	
	Tensor S({100,234,567}, Tensor::Representation::Sparse);
	S[{5,7,99}] = 1; S[{80,123,5}] = 6; S[{1,2,3}] = 4; S[{99,233,566}] = 5;
	S[{12,12,12}] = 8; S[{15,15,15}] = 7; S[{99,99,99}] = 3; S[{65,65,65}] = 2;
	
	save_to_file(S, "test.dat", misc::FileFormat::BINARY);
	Ab = misc::load_from_file<Tensor>("test.dat");
	MTEST(frob_norm(S - Ab) < 1e-16, "sparse bin " << frob_norm(S-Ab));
	
	save_to_file(S, "test.dat", misc::FileFormat::TSV);
	Ab = misc::load_from_file<Tensor>("test.dat");
	TEST(Ab.is_sparse());
	MTEST(frob_norm(S - Ab) < 1e-16, "sparse tsv " << frob_norm(S-Ab));
});

static misc::UnitTest tn_rw("TensorNetwork", "read_write_file", [](){
	Tensor A = Tensor::random({12,13,14});
	Tensor B = Tensor({12,13,14}, Tensor::Representation::Sparse);
	B[{1,1,1}] = 1; B[{2,3,4}] = 2; B[{7,10,3}] = 3; B[{5,12,13}] = 4; B[{8,7,6}] = 5;
	Tensor C = Tensor::random({12,13,14});
	Index i,j,k,l,m,n;
	TensorNetwork T;
	T(k,l,n) = A(i,j,k) * B(i,l,m) * C(n,j,m);
	save_to_file(T, "test.dat", misc::FileFormat::TSV);
	TensorNetwork Tb = misc::load_from_file<TensorNetwork>("test.dat");
	MTEST(T.dimensions == Tb.dimensions, T.dimensions << " vs " << Tb.dimensions);
	Tb.require_valid_network();
	MTEST(frob_norm(T(i&0)-Tb(i&0))/frob_norm(T) < 6e-16, frob_norm(T(i&0)-Tb(i&0))/frob_norm(T));
	
	save_to_file(T, "test.dat", misc::FileFormat::BINARY);
	Tb = misc::load_from_file<TensorNetwork>("test.dat");
	MTEST(T.dimensions == Tb.dimensions, T.dimensions << " vs " << Tb.dimensions);
	Tb.require_valid_network();
	MTEST(frob_norm(T(i&0)-Tb(i&0))/frob_norm(T) < 6e-16, frob_norm(T(i&0)-Tb(i&0))/frob_norm(T));
});


static misc::UnitTest tt_rw("TT", "read_write_file", [](){
	TTTensor A = TTTensor::random({7,8,9,10}, {2,2,2});
	
	misc::save_to_file(A, "test.dat");
	
	FAILTEST(TensorNetwork Tb = misc::load_from_file<TensorNetwork>("test.dat"));
	TTTensor Ab = misc::load_from_file<TTTensor>("test.dat");
	Index i;
	Ab.require_correct_format();
	MTEST(Ab.canonicalized && Ab.corePosition == 0, Ab.canonicalized << " " << Ab.corePosition);
	MTEST(A.dimensions == Ab.dimensions, A.dimensions << " vs " << Ab.dimensions);
	MTEST(frob_norm(A(i&0)-Ab(i&0))/frob_norm(A) < 6e-16, frob_norm(A(i&0)-Ab(i&0))/frob_norm(A));
});
