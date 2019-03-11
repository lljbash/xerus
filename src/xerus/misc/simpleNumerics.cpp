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

/**
 * @file
 * @brief Implementation of the Romberg integration, polynomial class and limit extractors.
 */

#include <xerus/misc/simpleNumerics.h>
 
#include <xerus/misc/math.h>
#include <xerus/misc/containerSupport.h>
#include <xerus/misc/exceptions.h>
#include <xerus/misc/namedLogger.h>
#include <xerus/misc/check.h>
#include <xerus/misc/internal.h>

namespace xerus {
    namespace misc {

        double integrate(const std::function<double(double)>& _f, const double _a, const double _b, double _eps, const uint _minIter, const uint _maxIter, const uint _branchFactor, const uint _maxRecursion, const bool _relativeError) {
            REQUIRE(_minIter > 1, "");
            REQUIRE(_branchFactor > 1, "");
            double min = std::min(_a,_b);
            double max = std::max(_a,_b);
            if (_relativeError) { _eps = std::max(_eps, std::numeric_limits<double>::epsilon()); }
            std::vector<double> iterants;
            iterants.push_back((max - min)*(_f(min) + _f(max))/2);
            double h = (max-min);
            double sum;
            double error = 1;
            double maxVal = std::abs(iterants[0]);
            for (uint iter=0; iter<_maxIter; ++iter) {
                sum = 0;
                for (double x = min+h/2; x<max; x += h) {
                    double f = _f(x);
                    sum += f;
                    maxVal = std::max(maxVal, std::abs(f));
                }
                
                h /= 2;
                sum *= h;
                sum += iterants.back()/2;
                iterants.push_back(sum);
                double oldIt0 = iterants[0];
                for (size_t k=0; k<iterants.size()-1; ++k) {
                    size_t i=iterants.size()-1-k;
                    iterants[i-1] = iterants[i] + (iterants[i]-iterants[i-1])/(misc::pow(2.0, 2*(k+1))-1);
                }
                if (_relativeError) {
                    error = std::abs((iterants[0]-oldIt0)/oldIt0);
                    if (std::isnan(error)) { error = std::abs(iterants[0]-oldIt0); }
                } else {
                    error = std::abs(iterants[0]-oldIt0);
                }
                if (iter >= _minIter && error <= _eps) {
                    return (_a>_b?-1:1)*iterants[0];
                }
            }
            if (_maxRecursion == 0) {
                return (_a>_b?-1:1)*iterants[0];
            }
            // divide and conquer
            // this dynamicaly divides those parts that did not converge easily
            LOG(debug, (_relativeError?"relative":"absolute") << " error " << error << " is larger than eps = " << _eps);
            sum = 0;
            h = (max-min) / _branchFactor;
            double newEps = (_relativeError
                                ? std::max(std::abs(iterants[0]), maxVal)*_eps 
                                : std::max(
                                    _eps, // /std::sqrt(_branchFactor), 
                                    std::sqrt(_branchFactor)*std::numeric_limits<double>::epsilon()*std::max(std::abs(iterants[0]), maxVal)
                                )
                            );
            for (uint i=0; i<_branchFactor; ++i) {
                sum += integrate(_f, min+i*h, min+(i+1)*h, newEps, _minIter, _maxIter, _branchFactor, _maxRecursion-1, false);
            }
            return (_a>_b?-1:1)*sum;
        }


        double integrate_segmented(const std::function<double(double)> &_f, const double _a, const double _b, const double _segmentation, const double _eps, const uint _minIter, const uint _maxIter, const uint _branchFactor, const uint _maxRecursion) {
            const double min = std::min(_a, _b);
            const double max = std::max(_a, _b);
            double res = 0;
            for (double x = min; x < max; x += _segmentation) {
                res += integrate(_f, x, std::min(x+_segmentation, max), _eps, _minIter, _maxIter, _branchFactor, _maxRecursion);
            }
            return (_a>_b?-1:1)*res;
        }

        
        double find_root_bisection(const std::function<double(double)> &_f, double _min, double _max, const double _epsilon) {
			const double nm = std::max(_min, _max);
			_min = std::min(_min, _max);
			_max = nm;
			
			double fmin = _f(_min);
			double fmax = _f(_max);
			
			REQUIRE(fmin * fmax <= 0, "bisection requires inputs to both sides of the root (and no NaNs)");
			
			if (std::fpclassify(fmin) == FP_ZERO) {
				return _min;
			}
			
			if (std::fpclassify(fmax) == FP_ZERO) {
				return _max;
			}
			
			while (_max-_min > _epsilon) {
				double mid = (_max+_min)/2.0;
				double fmid = _f(mid);
				if (std::fpclassify(fmid) == FP_ZERO) {
					return mid;
				}
				REQUIRE(std::isfinite(fmid), "invalid function value f("<<mid<<") = " << fmid << " reached in bisection");
				if (fmin * fmid < 0) {
// 					fmax = fmid; // never needed again
					_max = mid;
				} else {
					fmin = fmid;
					_min = mid;
				}
			}
			
			return (_max + _min) / 2.0;
		}
	
        // - - - - - - - - - - - - - - - Polynomial - - - - - - - - - - - - - - -
        
        Polynomial::Polynomial() {}
        
        Polynomial::Polynomial(std::vector<double> _coeff) : coefficients(std::move(_coeff)) {}
        
        size_t Polynomial::terms() const {
            return coefficients.size();
        }
        
        Polynomial& Polynomial::operator+=(const Polynomial &_rhs) {
			coefficients.resize(std::max(terms(), _rhs.terms()));
			
			for(size_t i = 0; i < _rhs.terms(); ++i) {
				coefficients[i] += _rhs.coefficients[i];
			}

            return *this;
        }
        
        Polynomial& Polynomial::operator-=(const Polynomial &_rhs) {
			coefficients.resize(std::max(terms(), _rhs.terms()));
			
			for(size_t i = 0; i < _rhs.terms(); ++i) {
				coefficients[i] -= _rhs.coefficients[i];
			}

            return *this;
        }
        
        
        Polynomial& Polynomial::operator/=(double _rhs) {
            for (size_t i=0; i<terms(); ++i) {
                coefficients[i] /= _rhs;
            }
            return *this;
        }
        
        Polynomial& Polynomial::operator*=(double _rhs) {
            for (size_t i=0; i<terms(); ++i) {
                coefficients[i] *= _rhs;
            }
            return *this;
        }
        
        Polynomial Polynomial::operator*(const Polynomial &_rhs) const {
            Polynomial result;
            result.coefficients.resize(_rhs.terms()+terms()-1);
            for (size_t i=0; i<terms(); ++i) {
                for (size_t j=0; j<_rhs.terms(); ++j) {
                    result.coefficients[i+j] += coefficients[i]*_rhs.coefficients[j];
                }
            }
            return result;
        }
        
        Polynomial Polynomial::operator*(double _rhs) const {
            Polynomial result;
            result.coefficients.resize(terms());
            for (size_t i=0; i<terms(); ++i) {
                result.coefficients[i] = coefficients[i]*_rhs;
            }
            return result;
        }
        
        double Polynomial::operator()(double x) const {
            double result = 0;
            for (size_t i=terms(); i>0; --i) {
                result *= x;
                result += coefficients[i-1];
            }
            return result;
        }
        
        double Polynomial::scalar_product(const Polynomial &_rhs, const std::function<double (double)> &_weight, double _minX, double _maxX) const {
            return xerus::misc::integrate([&](double x){
                return (*this)(x) * _rhs(x) * _weight(x);
            }, _minX, _maxX, 1e-10);
        }
        
        double Polynomial::norm(const std::function<double (double)> &_weight, double _minX, double _maxX) const {
            return std::sqrt(scalar_product(*this, _weight, _minX, _maxX));
        }
        
        /// orthogonalizes this polynomial with respect to the provided (@note orthogonal!) basis
        Polynomial& Polynomial::orthogonolize(const std::vector<Polynomial> &_orthoBase, const std::function<double (double)> &_weight, double _minX, double _maxX) {
            for (size_t i=0; i<_orthoBase.size(); ++i) {
                (*this)-=_orthoBase[i]*scalar_product(_orthoBase[i], _weight, _minX, _maxX);
                REQUIRE(std::abs(scalar_product(_orthoBase[i], _weight, _minX, _maxX)) < 1e-12, i << " " << std::abs(scalar_product(_orthoBase[i], _weight, _minX, _maxX)));
            }
            (*this)/=norm(_weight, _minX, _maxX);
            return *this;
        }
        
        /// returns @a _N pairwise orthogonal polynomials w.r.t. a scalar product defined by the @a _weight
        std::vector<Polynomial> Polynomial::build_orthogonal_base(size_t _N, const std::function<double (double)> &_weight, double _minX, double _maxX) {
            std::vector<Polynomial> base;
            while (base.size() < _N) {
                Polynomial next;
                next.coefficients.resize(base.size()+1);
                next.coefficients.back() = 1; // next = x^(base.size)
                next.orthogonolize(base, _weight, _minX, _maxX);
                base.push_back(next);
                LOG(debug, "next basis polynomial " << next.coefficients);
            }
            return base;
        }
        
        
        // - - - - - - - - - - - - - - - ShanksTransformation - - - - - - - - - - - - - - -
        
        template<class ft_type>
        ft_type ShanksTransformation<ft_type>::shanks(ft_type x1, ft_type x2, ft_type x3) {
            ft_type denominator = x1 - 2*x2 + x3;
            if (std::abs(denominator) < epsilon * std::max(x1, std::max(x2, x3))) {
                return x2;
            }
            return (x1*x3 - x2*x2) / denominator;
        }
        
        template<class ft_type>
        void ShanksTransformation<ft_type>::push_back(ft_type _val) {
            values.push_back(_val);
            for (size_t i=values.size()-1; i>=2; i-=2) {
                values[i-2] = shanks(values[i-2], values[i-1], values[i]);
            }
        }
        
        template<class ft_type>
        ft_type ShanksTransformation<ft_type>::best_estimate() const {
            if (values.empty()) {
                XERUS_THROW(generic_error() << "tried to extract limit of empty sequence");
            } else {
                return values[(values.size()-1) % 2];
            }
        }
        
        template<class ft_type>
        ft_type ShanksTransformation<ft_type>::error_approximate() const {
            size_t i = (values.size()-1) % 2;
            if (i+1 >= values.size()) {
                return 1;
            }
            return std::abs(values[i] - values[i+1]);
        }
        
        template<class ft_type>
        void ShanksTransformation<ft_type>::reset() {
            values.clear();
        }
        
        template class ShanksTransformation<float>;
        template class ShanksTransformation<double>;

        
        
            
        // - - - - - - - - - - - - - - - ShanksTransformation - - - - - - - - - - - - - - -

        template<class ft_type>
        ft_type RichardsonExtrapolation<ft_type>::richard(size_t n, ft_type x1, ft_type x2) {
            return ft_type(n+1)*x2 - ft_type(n)*x1;
        }
        
        template<class ft_type>
        void RichardsonExtrapolation<ft_type>::push_back(ft_type _val) {
            values.push_back(_val);
            for (size_t i=values.size()-1; i>=1; i-=1) {
                values[i-1] = richard(i-1, values[i-1], values[i]);
            }
        }
        
        template<class ft_type>
        ft_type RichardsonExtrapolation<ft_type>::best_estimate() const {
            if (values.empty()) {
                XERUS_THROW(generic_error() << "tried to extract limit of empty sequence");
            } else {
                return values.front();
            }
        }
        
        template<class ft_type>
        ft_type RichardsonExtrapolation<ft_type>::error_approximate() const {
            if (values.size() < 2) {
                return 1;
            }
            return std::abs(values[0] - values[1]);
        }
        
        template<class ft_type>
        void RichardsonExtrapolation<ft_type>::reset() {
            values.clear();
        }
        
        
        template class RichardsonExtrapolation<float>;
        template class RichardsonExtrapolation<double>;
    
    } // namespace misc
} // namespace xerus
