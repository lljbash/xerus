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
using namespace xerus;

static misc::UnitTest failtest("Index", "invalid_spans", [](){
#ifdef XERUS_DISABLE_RUNTIME_CHECKS
	LOG(warning, "failtests skipped due to XERUS_DISABLE_RUNTIME_CHECKS");
#else
	Index i,j;
	Tensor c;
	Tensor A = Tensor::random({10,10});
	Tensor b = Tensor::random({10});
	
	try {
		A(i,j^2) = A(j^2,i);
		MTEST(false, "A(j^2,i) did not throw!");
	} catch (misc::generic_error &_e) {
		TEST(true);
	} catch (...) {
		MTEST(false, "threw a wrong exception class!");
	}
	
	try {
		A(i,j) = A(j,i&0);
		MTEST(false, "A(j^2,i) did not throw!");
	} catch (misc::generic_error &_e) {
		TEST(true);
	} catch (...) {
		MTEST(false, "threw a wrong exception class!");
	}
	
	try {
		A(i,j) = A(j,i/1);
		MTEST(false, "A(j^2,i) did not throw!");
	} catch (misc::generic_error &_e) {
		TEST(true);
	} catch (...) {
		MTEST(false, "threw a wrong exception class!");
	}
	
	try {
		A(i,j) = A(j,i/3);
		MTEST(false, "A(j^2,i) did not throw!");
	} catch (misc::generic_error &_e) {
		TEST(true);
	} catch (...) {
		MTEST(false, "threw a wrong exception class!");
	}
	
	try {
		c(j) = A(i,j) * b(j);
		MTEST(false, "c(j)=... did not throw!");
	} catch (misc::generic_error &_e) {
		TEST(true);
	} catch (...) {
		MTEST(false, "threw a wrong exception class!");
	}
#endif
});
