CXX = g++
COMPATIBILITY = -std=c++17
COMPILE_THREADS = 8                     # Number of threads to use during link time optimization.
HIGH_OPTIMIZATION = TRUE                # Activates -O3 -march=native and some others
OTHER += -fopenmp

LOGGING += -D XERUS_LOG_INFO                     # Information that is not linked to any unexpected behaviour but might nevertheless be of interest.
LOGGING += -D XERUS_LOGFILE               # Use `error.log` file instead of cerr
LOGGING += -D XERUS_LOG_ABSOLUTE_TIME     # Print absolute times instead of relative to program time

INSTALL_LIB_PATH = ${PREFIX}/lib         # Path where to install the libxerus.so shared library.
INSTALL_HEADER_PATH = ${PREFIX}/include      # Path where to install the xerus header files.

BLAS_LIBRARIES = -lopenblas -lgfortran                      # Openblas, serial
LAPACK_LIBRARIES = -llapacke -llapack                       # Standard Lapack + Lapacke libraries
SUITESPARSE = -lcholmod -lspqr
