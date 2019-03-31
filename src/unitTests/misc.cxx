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


#include <xerus.h> // NOTE xerus.h header file is necessary for below check of internal header export

#include "../../include/xerus/test/test.h"
using namespace xerus;


static misc::UnitTest misc_internal("Misc", "internal_header", [](){
	#ifdef REQUIRE
		MTEST(false, "internal header file xerus/misc/internal.h is being exported in the library!");
	#else
		TEST(true);
	#endif
});


#include "../../include/xerus/misc/internal.h"

static misc::UnitTest misc_romberg("Misc", "romberg_integration", [](){
	double npi;
	npi = 2*misc::integrate([](double x){ return std::sqrt(1-x*x); }, -1, 1, 1e-16);
	LOG(unit_test, npi << " err: " << std::abs(npi - M_PI));
	TEST(misc::approx_equal(npi,M_PI,2e-15));
	
	npi = 2*misc::integrate([](double x){ return -std::sqrt(1-x*x); }, -1, 1, 1e-16);
	LOG(unit_test, npi << " err: " << std::abs(npi + M_PI));
	TEST(misc::approx_equal(npi,-M_PI,2e-15));
	
	npi = misc::integrate([](double x){ return (1-x); }, -1, 1, 1e-14);
	LOG(unit_test, std::abs(npi - 2));
	TEST(misc::approx_equal(npi,2.0,2e-14));
	
	npi = misc::integrate([](double x){ return (x*x*x+1e-14); }, -1, 1, 1e-14);
	LOG(unit_test, std::abs(npi - 2e-14));
	TEST(std::abs(npi - 2e-14) < 1e-14);
	
	npi = misc::integrate([](double x){ return std::cos(x); }, 0, 1, 1e-14);
	LOG(unit_test, npi << " " << std::sin(1) << " " << std::abs(npi - std::sin(1)));
	TEST(misc::approx_equal(npi,std::sin(1),2e-14));
	
	npi = misc::integrate([](double x){ return (x>0&&x<=1?1:0); }, -2, 2, 1e-14, 3);
	LOG(unit_test, npi << " " << 1 << " " << std::abs(npi-1));
	TEST(misc::approx_equal(npi,1.0,2e-14));
	
	npi = misc::integrate([](double x){ return (x>0&&x<=1?1:0); }, -2, 2, 1e-14, 3);
	LOG(unit_test, npi << " " << 1 << " " << std::abs(npi-1));
	TEST(misc::approx_equal(npi,1.0,2e-14));
});


static misc::UnitTest misc_poly("Misc", "polynomial", [](){
	auto weight = [](double _x){
		return std::abs(std::sin(_x));
	};
	std::vector<misc::Polynomial> base = misc::Polynomial::build_orthogonal_base(10, weight, -1, 1);
	for (size_t i=0; i<base.size(); ++i) {
		TEST(base[i].terms() == i+1);
		TEST(misc::approx_equal(base[i].norm(weight, -1, 1), 1.0, 1e-12));
		for (size_t j=0; j<base.size(); ++j) {
			if (i==j) continue;
			TEST(std::abs(base[i].scalar_product(base[j], weight, -1, 1)) < 1e-10);
		}
	}
});


static misc::UnitTest misc_exc("Misc", "exceptions", [](){
	try {
		XERUS_THROW(misc::generic_error());
	} catch (misc::generic_error &e) {
		MTEST(true, "1");
	} catch (...) {
		MTEST(false, "1");
	}
	
	try {
		XERUS_THROW(misc::generic_error() << "test");
	} catch (misc::generic_error &e) {
		MTEST(true, "2");
	} catch (...) {
		MTEST(false, "2");
	}
	
	struct test_error : public misc::generic_error {};
	
	try {
		XERUS_THROW(test_error());
	} catch (test_error &e) {
		MTEST(true, "3");
	} catch (...) {
		MTEST(false, "3");
	}
	
	try {
		XERUS_THROW(test_error() << "test");
	} catch (test_error &e) {
		MTEST(true, "4");
	} catch (...) {
		MTEST(false, "4");
	}
});
