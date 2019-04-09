all: measuredTensor

# main: main.cpp rankOneMeasuredTensor.cpp
# 	g++-8 -std=c++17 -fopenmp -I ~/wias/libs/xerus/include -o $@ $^ -L ~/wias/libs/xerus/build -lxerus -lxerus_misc
# 	#$(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) -lbfd -liberty -lz -ldl

measuredTensor: rankOneMeasuredTensor.cpp measuredTensor.cpp
	# g++-8 -std=c++17 -fopenmp -O3 -Wall -shared -fPIC -I ~/wias/libs/xerus/include `python3 -m pybind11 --includes` $^ -L ~/wias/libs/xerus/build -lxerus -lxerus_misc -o $@`python3-config --extension-suffix`
	g++-8 -std=c++17 -fopenmp -O3 -Wall -shared -fPIC -I pybind11/include -I ~/xerus/include `python3-config --includes` $^ -L ~/xerus/build -lxerus -lxerus_misc -o $@`python3-config --extension-suffix`
