#include <xerus.h>

using namespace xerus;

class InternalSolver {
	const size_t d;
	
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;
	
	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;
	
	TTTensor& x;
	const TTOperator& A;
	const TTTensor& b;
	const double solutionsNorm;
public:
	size_t maxIterations;
	
	InternalSolver(const TTOperator& _A, TTTensor& _x, const TTTensor& _b) 
		: d(_x.order()), x(_x), A(_A), b(_b), solutionsNorm(frob_norm(_b)), maxIterations(1000)
	{ 
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		leftBStack.emplace_back(Tensor::ones({1,1}));
		rightBStack.emplace_back(Tensor::ones({1,1}));
	}
	
	
	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);
		
		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = leftBStack.back()(j1, j2)
				*xi(j1, k1, i1)*bi(j2, k1, i2);
		leftBStack.emplace_back(std::move(tmpB));
	}
	
	
	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);
		
		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = xi(i1, k1, j1)*bi(i2, k1, j2)
				*rightBStack.back()(j1, j2);
		rightBStack.emplace_back(std::move(tmpB));
	}
	
	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - b(i&0)) / solutionsNorm;
	}
	
	
	void solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}
		
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		std::vector<double> residuals(10, 1000.0);
		
		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			if (residuals.back()/residuals[residuals.size()-10] > 0.99) {
				XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
				return; // We are done!
			}
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back());
			
			
			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;
				
				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &bi = b.get_component(corePosition);
				
				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);
				rhs(i1, i2, i3) =            leftBStack.back()(i1, k1) *   bi(k1, i2, k2) *   rightBStack.back()(i3, k2);
				
				xerus::solve(x.component(corePosition), op, rhs);
				
				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
					rightBStack.pop_back();
				}
			}
			
			
			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
				leftBStack.pop_back();
			}
			
		}
	}
	
};

void simpleALS(const TTOperator& _A, TTTensor& _x, const TTTensor& _b)  {
	InternalSolver solver(_A, _x, _b);
	return solver.solve();
}


int main() {
	Index i,j,k;
	
	auto A = TTOperator::random(std::vector<size_t>(16, 4), std::vector<size_t>(7,2));
	A(i/2,j/2) = A(i/2, k/2) * A(j/2, k/2);
	
	auto solution = TTTensor::random(std::vector<size_t>(8, 4), std::vector<size_t>(7,3));
	TTTensor b;
	b(i&0) = A(i/2, j/2) * solution(j&0);
	
	auto x = TTTensor::random(std::vector<size_t>(8, 4), std::vector<size_t>(7,3));
	simpleALS(A, x, b);
	
	XERUS_LOG(info, "Residual: " << frob_norm(A(i/2, j/2) * x(j&0) - b(i&0))/frob_norm(b));
	XERUS_LOG(info, "Error: " << frob_norm(solution-x)/frob_norm(x));
}
