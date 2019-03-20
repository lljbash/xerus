#include<xerus.h>

#include "../../include/xerus/test/test.h"
using namespace xerus;

static misc::UnitTest tt_entryprod("TT", "entrywise_product", [](){
	Index i,j,k;

	TTTensor A = TTTensor::random(std::vector<size_t>(10,2), std::vector<size_t>(9,2));
	TTTensor B = TTTensor::random(std::vector<size_t>(10,2), std::vector<size_t>(9,2));
	
	Tensor Af(A);
	Tensor Bf(B);
	
	TTOperator Ao(Af);
	TTOperator Bo(Bf);
	
	TTTensor C = entrywise_product(A, B);
	TTOperator Co = entrywise_product(Ao, Bo);
	Tensor Cf = entrywise_product(Af, Bf);
	
	MTEST(frob_norm(Cf - Tensor(Co))/frob_norm(Cf) < 1e-13, frob_norm(Cf - Tensor(Co))/frob_norm(Cf));
	MTEST(frob_norm(Cf - Tensor(C))/frob_norm(Cf) < 1e-13, frob_norm(Cf - Tensor(C))/frob_norm(Cf));
	
	
	TTTensor D1 = entrywise_product(A, A);
	
	TTOperator Do1 = entrywise_product(Ao, Ao);
	
	Tensor Df = entrywise_product(Af, Af);

	MTEST(approx_equal(Df, Tensor(D1), 1e-13), frob_norm(Df-Tensor(D1)));
	MTEST(approx_equal(Df, Tensor(Do1), 1e-13), frob_norm(Df- Tensor(Do1)));
});

static misc::UnitTest tt_soft("TT", "soft_thresholding", [](){
	Index i,j,k;

	TTTensor A = TTTensor::random(std::vector<size_t>(10,2), std::vector<size_t>(9,2));
	TTTensor B = TTTensor::random(std::vector<size_t>(10,2), std::vector<size_t>(9,2));
	
	
	Tensor Af(A);
	Tensor Bf(B);
	
	TTOperator Ao(Af);
	TTOperator Bo(Bf);
	
	TTTensor C = entrywise_product(A, B);
	TTOperator Co = entrywise_product(Ao, Bo);
	Tensor Cf = entrywise_product(Af, Bf);
	
	TEST(frob_norm(Cf - Tensor(Co))/frob_norm(Cf) < 1e-14);
	TEST(frob_norm(Cf - Tensor(C))/frob_norm(Cf) < 1e-14);
});


static misc::UnitTest tt_pseudo_inv("TT", "Non-operator_Pseudo_Inverse", [](){
	Index i,j,k,r1,r2,r3,r4;
	
	const size_t d = 2;
	
	const TTTensor op = TTTensor::random(std::vector<size_t>(2*d, 10), std::vector<size_t>(2*d-1, 4));
	
	TTTensor tmp = op;
	tmp.move_core(d);
	std::pair<TensorNetwork, TensorNetwork> parts = tmp.chop(d);
	Tensor core = tmp.get_component(d);
	Tensor U,S,V;
	(U(i,r1), S(r1,r2), V(r2,j^2)) = SVD(core(i,j^2));
	S.modify_diagonal_entries([](double &_v){
		if (_v>1e-10) {
			_v = 1/_v;
		}
	});
	TensorNetwork pInv;
	pInv(j, i^(d-1), k^d) = parts.first(k^d,r1) * U(r1,r2) * S(r2,r3) * V(r3,j,r4) * parts.second(r4,i^(d-1));
	
	double error = frob_norm(pInv(i^d,r1^d)*op(r1^d,r2^d)*pInv(r2^d,j^d) - pInv(i^d,j^d));
	MTEST(error < 1e-10, "A^+ A A^+ != A^+ ? " << error);
	
	error = frob_norm(op(i^d,r1^d)*pInv(r1^d,r2^d)*op(r2^d,j^d) - op(i^d,j^d));
	MTEST(error < 1e-10, "A A^+ A != A ? " << error);
});
