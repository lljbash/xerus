## How to contribute to Xerus

#### **You found a bug**

* Please see if we are already aware of the problem by searching our [issue tracker](https://git.hemio.de/xerus/xerus/issues).

* If you can't find an open issue addressing the problem, [open a new one](https://git.hemio.de/xerus/xerus/issues/new). Please include all relevant information, as well as a **code sample** and if possible an **unit-test** demonstrating the expected behavior that is not occurring. This helps to ensure that no code breaking the tested functionality will be merged in the future.

* A unit test can be as simple as
```
#include <xerus.h>
#include <xerus/test/test.h>

int main() {
	int a = 2;
	int b = 3;
	TEST(a+b == 5);
}
```


#### **You wrote a patch that fixes a bug**

* Make sure that the code compiles without warning and all unit-tests are passed.

* Create a merge request against the [xerus development branch](https://git.hemio.de/xerus/xerus/tree/development). 

* Ensure the PR description clearly describes the problem and the solution. Include the relevant issue number if applicable.


#### **You want to add a new feature to xerus or change an existing one**

* Espacially for changes to existing code, we recommend to [open an issue](https://git.hemio.de/xerus/xerus/issues/new) detailing your plan, to collect feedback before start writing code.

* If the feedback is positive feel free to create a new branch and implement you changes.

* We have a few code and style conventions we ask our contributers to respect:

  - As a general rule we loosly follow the [C++ Core Guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md) and recommend everyone to read them before starting to work on larger contributions.

  - We use an aggressive set of warnings (including `-Wall -Wextra -pedantic` and many more) to ensure less error prone code. Any submission should resprect these warnings and will otherwise directly fail our CI tests.

  - As for naming conventions, classes are upper camel-case (e.g. `TensorNetwork`), variables are lower camel-case (e.g. `TensorNetwork myNetwork`), functions are lower case with underscores (e.g. `void do_something()`) and macros as well as constants are upper case with underscores (e.g. `XERUS_REQUIRE(...)`).
  
  - Indention should be done with tabs rather then spaces.

* If you changes are complete, the code compiles without warnings and all unit-tests are passed, you can create a merge request against the [xerus development branch](https://git.hemio.de/xerus/xerus/tree/development).


#### **You have any other questions about xerus**

* If you have any questions please do not hesitate to write us a mail to contact[at]libxerus.org.



Thank you for you interest. We look forward to hear from you.
