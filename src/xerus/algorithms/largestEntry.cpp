
#include <xerus/algorithms/largestEntry.h>
#include <xerus/misc/internal.h>

namespace xerus {
	template<bool isOperator>
	size_t find_largest_entry_rank_one(const TTNetwork<isOperator> &_T) {
		INTERNAL_CHECK(misc::sum(_T.ranks())+1 == _T.order(), "Ie");
		
		const size_t numComponents = _T.order()/(isOperator?2:1);
		size_t position = 0;
		size_t factor = misc::product(_T.dimensions);
		for(size_t c = 0; c < numComponents; ++c) {
			const size_t localSize = isOperator ? _T.dimensions[c]*_T.dimensions[numComponents+c] : _T.dimensions[c];
			factor /= localSize;
			
			size_t maxPos = 0;
			for(size_t i = 1; i < localSize; ++i) {
				if(std::abs(_T.get_component(c)[i]) > std::abs(_T.get_component(c)[maxPos])) {
					maxPos = i;
				}
			}
			position += maxPos*factor;
		}
		return position;
	}
	
	
	template<bool isOperator>
	size_t find_largest_entry(const TTNetwork<isOperator> &_T, const double _accuracy, const value_t _lowerBound) {
		_T.require_correct_format();
		
		const value_t numSVDs = double(_T.order()-1);
		const value_t gamma = (1-_accuracy)*_accuracy/2.0;
		
		TTNetwork<isOperator> X = _T;
		TTNetwork<isOperator> Y = X;
		
		Y.round(1);
		double Xn = std::max(_T[find_largest_entry_rank_one(Y)], _lowerBound);
		double tau = gamma*Xn*Xn/numSVDs;
		
		while(misc::sum(X.ranks()) >= _T.order()) {
			X = entrywise_product(X, X);
			X.soft_threshold(tau);
			
			Y = X; Y.round(1);
			
			Xn = std::max(X[find_largest_entry_rank_one(Y)], (1-gamma)*Xn*Xn);
			
			const double fNorm = X.frob_norm();
			Xn /= fNorm;
			X /= fNorm;
			tau = gamma*Xn*Xn/numSVDs;
		}
		
		return find_largest_entry_rank_one(X);
	}
	
	template size_t find_largest_entry(const TTNetwork<true> &, double, value_t);
	template size_t find_largest_entry(const TTNetwork<false> &, double, value_t);
	
} // namespace xerus
