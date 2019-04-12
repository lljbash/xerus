---
layout: post
title: "Building Xerus"
date: 2000-12-20 16:25:06 +0000
topic: "Introduction"
section: "Documentation"
---


# Building Xerus

## Obtaining Xerus

You can get the source-code of the `xerus` library via [git](https://git.hemio.de/xerus/xerus/tree/master) or as an [archiv](https://git.hemio.de/xerus/xerus/repository/archive.tar.gz?ref=master).
For example to clone the repository with the latest stable version under linux simply type
~~~ bash
git clone https://git.hemio.de/xerus/xerus.git
~~~

## Dependencies and Configuration
`Xerus` depends on several well established libraries, usually pre-installed or available through the standard package managers. In particular `lapacke`, `CXSparse`, `binutils`, `boost`
and their dependencies. Also note that at least version 4.8 of the `GCC` is required, as this is the first version to offer support for all `C++11` functionality used in `xerus`. 
Make sure that all these are installed on your system before proceeding.

E.g. to install all dependencies on a fedora system execute
~~~ bash
dnf install gcc-c++ openblas-devel suitesparse-devel lapack-devel boost-devel binutils-devel
~~~

Or to install all dependencies for Ubuntu (tested for 18.04) execute:
~~~ bash
apt-get update
apt install make g++ libboost-all-dev binutils-dev libsuitesparse-dev libz-dev libiberty-dev libopenblas-dev liblapack-dev liblapacke-dev gfortran
~~~

To build the optional ARPACK extension, you need to install ARPACK and the c++ headers by hand. First clone the git project to an appropriate place, enter the project directory, configure the sources and make the library:
~~~ bash
git clone https://github.com/opencollab/arpack-ng.git
cd arpack-ng
apt-get install autoconf
sh bootstrap
./configure F77=gfortran --enable-icb
make
make check
~~~
You can add your own installation path with --prefix=<installation-path> for the configure command. 
Unfortunately, the C wrapper is broken. You need to modify the file_exists arpack.hpp:
~~~ bash
cd ICB/
vi arpack.hpp
~~~
In arpack.hpp comment out all 4 instances of the function neupd.

After that, to install the packages type:
~~~ bash
make install
~~~

Once ARPACK is installed you need to modify the xerus config file. Just remove the hashtags in front of ARPACK_LIBRARIES and the addtional flag below:
~~~ bash
# (optional) Uncomment if needed, iterative eigenvalue solver ARPACK-ng, see https://github.com/opencollab/arpack-ng
ARPACK_LIBRARIES = -larpack
OTHER += -DARPACK_LIBRARIES
~~~

To build the python bindings you will furthermore need the python development headers, `numpy` as well as `boost-python` or 
`boost-python3` depending on the python version you wish to use. E.g. for a fedora system and if you want to use python 2 simply execute
~~~ bash
dnf install python2-numpy python-devel
~~~

To build the documentation (i.e. this homepage) locally you will need `jekyll` and `doxygen`.

After downloading the source it is necessary to set a number of options that are somewhat individual to your system and needs. All following build steps assume that these
options are set in the `config.mk` file. Reasonable default values can be found in the `config.mk.default` file.

In particular the optimization level is interesting and the paths or standard flags for the required libraries need to be set. This way the `xerus` library can be compiled with any
`blas` compatible library for the matrix operations. For more details see the description in the `config.mk.default` file
~~~ bash
cp config.mk.default config.mk
nano config.mk
~~~

## Ensure Everything Works as Intended
The sources include a number of unit tests that ensure that everything works as intended. To perform them simply input
~~~ bash
make test -j4
~~~
(where `-j4` allows make to use up to 4 threads for the compilation). If all options were set correctly in the `config.mk` it should compile a test executable and launch it.
The output of this executable should then list a number of passed tests and end with something similar to 
~~~
|
|                       total summary 149 of 149 passed
-------------------------------------------------------------------------------
|
|                       Total time elapsed: 5594.088 ms
-------------------------------------------------------------------------------
~~~
Note in particular, that all tests were passed. Should this not be the case please file a bug report with as many details as you 
can in our [issuetracker](https://git.hemio.de/xerus/xerus/issues) or write us an [email](mailto:contact@libxerus.org).


## Building the Library

If all tests were passed you can build the library simply by calling `make shared` or `make static` depending on whether you want
to build shared or static library objects. To build all shared objects as well as the python bindings use `make python`. After
the compilation you should have a `build/` directory with a `libxerus_misc.a/.so` (suffix depending on build type), a `libxerus.a/.so`
and the `xerus.so` for python in case you also built the python bindings.

You can use these objects directly by setting your environment variables accordingly or you can install the library to a path
already included in these variables. To do so simply set the path in `config.mk` and call (as root if necessary) `make install`.


## Compiling your own Applications Using Xerus

If `xerus` is properly installed on your system, compiling your own applications using `xerus` is as simple as using any other library. Just include `-lxerus` in your linker call and make sure to use
`-std=c++11` or `-std=c++14` in all compilation units that include `xerus.h` and everything should work fine.

If you want to use the static version of `xerus` you also have to include all libraries `xerus` depends on in your linker call. In particular these are lapacke (`-llapacke`), 
lapack (`-llapack`), blas (`-lblas` or `-lopenblas` or similar), suitesparse (`-lcholmod -lspqr`) and binutils (`-lbfd`). On some old system one has to manually include the dependencys of binutils (`-liberty -lz -ldl`).

You can test this by trying to compile one of the examples (here without the installed `xerus` library)
~~~ bash
g++ -std=c++11 doc/_include/examples/qtt.cpp build/libxerus.a build/libxerus_misc.a -llapacke -llapack -lopenblas -lgfortran -lcholmod -lspqr -lbfd -liberty -lz -ldl
~~~
or with the shared objects and correctly set environment variables
~~~ bash
g++ -std=c++11 doc/_include/examples/qtt.cpp -lxerus
~~~

Within python you should be able to simply use
~~~ python
import xerus
~~~
if everything is set up correctly.
