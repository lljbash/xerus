.PHONY: help shared static python2 python3 doc install test clean version opt warn printBoostVersion

# ------------------------------------------------------------------------------------------------------
#				Default rule should be the help message
# ------------------------------------------------------------------------------------------------------
help:
	@printf "Possible make targets are:\n \
	\t\tversion \t -- Print the version of xerus.\n \
	\t\tshared  \t -- Build xerus as a shared library.\n \
	\t\tstatic  \t -- Build xerus as a static library.\n \
	\t\tpython2 \t -- Build the xerus python2 wrappers.\n \
	\t\tpython3 \t -- Build the xerus python3 wrappers.\n \
	\t\tdoc     \t -- Build the html documentation for the xerus library.\n \
	\t\tinstall \t -- Install the shared library and header files (may require root).\n \
	\t\ttest    \t -- Build and run the xerus unit tests.\n \
	\t\tclean   \t -- Remove all object, library and executable files.\n"


# ------------------------------------------------------------------------------------------------------
#				Set the names of the resulting binary codes
# ------------------------------------------------------------------------------------------------------

# Name of the test executable
TEST_NAME = XerusTest


# ------------------------------------------------------------------------------------------------------
#			Extract Xerus version from VERSION file and the git repository
# ------------------------------------------------------------------------------------------------------

VERSION_STRING = $(shell git describe --tags --always 2>/dev/null || cat VERSION)
VERSION = -D XERUS_VERSION="$(VERSION_STRING)"

XERUS_MAJOR_V = $(word 1, $(subst ., ,$(VERSION_STRING)) )
ifneq (,$(findstring v, $(XERUS_MAJOR_V)))
	XERUS_MAJOR_V := $(strip $(subst v, ,$(XERUS_MAJOR_V)) )
endif
VERSION += -D XERUS_VERSION_MAJOR=$(XERUS_MAJOR_V)
XERUS_MINOR_V = $(word 2, $(subst ., ,$(VERSION_STRING)) )
VERSION += -D XERUS_VERSION_MINOR=$(XERUS_MINOR_V)
XERUS_REVISION_V = $(word 3, $(subst ., ,$(VERSION_STRING)) )
ifneq (,$(findstring -, $(XERUS_REVISION_V)))
	XERUS_COMMIT_V := $(word 2, $(subst -, ,$(XERUS_REVISION_V)) )
	XERUS_REVISION_V := $(word 1, $(subst -, ,$(XERUS_REVISION_V)) )
else
	XERUS_COMMIT_V = 0
endif
VERSION += -D XERUS_VERSION_REVISION=$(XERUS_REVISION_V)
VERSION += -D XERUS_VERSION_COMMIT=$(XERUS_COMMIT_V)


# ------------------------------------------------------------------------------------------------------
#				Register source files for the xerus library
# ------------------------------------------------------------------------------------------------------

# Register the source files
XERUS_SOURCES = $(wildcard src/xerus/*.cpp)
XERUS_SOURCES += $(wildcard src/xerus/algorithms/*.cpp)
XERUS_SOURCES += $(wildcard src/xerus/applications/*.cpp)
XERUS_SOURCES += $(wildcard src/xerus/examples/*.cpp)

MISC_SOURCES = $(wildcard src/xerus/misc/*.cpp)

PYTHON_SOURCES = $(wildcard src/xerus/python/*.cpp)

TEST_SOURCES = $(wildcard src/xerus/test/*.cpp)

UNIT_TEST_SOURCES = $(wildcard src/unitTests/*.cxx)

TUTORIAL_SOURCES = $(wildcard tutorials/*.cpp)

# Create lists of the corresponding objects and dependency files
XERUS_OBJECTS = $(XERUS_SOURCES:%.cpp=build/.libObjects/%.o)
XERUS_DEPS    = $(XERUS_SOURCES:%.cpp=build/.libObjects/%.d)

MISC_OBJECTS = $(MISC_SOURCES:%.cpp=build/.miscObjects/%.o)
MISC_DEPS    = $(MISC_SOURCES:%.cpp=build/.miscObjects/%.d)

# PYTHON_OBJECTS = $(PYTHON_SOURCES:%.cpp=build/.pythonObjects/%.o)
# PYTHON_DEPS    = $(PYTHON_SOURCES:%.cpp=build/.pyhtonObjects/%.d)

TEST_OBJECTS = $(TEST_SOURCES:%.cpp=build/.testObjects/%.o)
TEST_DEPS    = $(TEST_SOURCES:%.cpp=build/.testObjects/%.d)

UNIT_TEST_OBJECTS = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.o)
UNIT_TEST_DEPS    = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.d)

TUTORIALS	= $(TUTORIAL_SOURCES:%.cpp=build/.tutorialObjects/%)
TUTORIAL_DEPS   = $(TUTORIAL_SOURCES:%.cpp=build/.tutorialObjects/%.d)


# ------------------------------------------------------------------------------------------------------
#		Load the configurations provided by the user and set up general options
# ------------------------------------------------------------------------------------------------------
include config.mk

include makeIncludes/general.mk
include makeIncludes/warnings.mk
include makeIncludes/optimization.mk


# ------------------------------------------------------------------------------------------------------
#					Set additional compiler options
# ------------------------------------------------------------------------------------------------------
ifdef ACTIVATE_CODE_COVERAGE
	DEBUG += -D XERUS_TEST_COVERAGE
endif

# ------------------------------------------------------------------------------------------------------
#					Convinience variables
# ------------------------------------------------------------------------------------------------------

# Small hack to get newlines...
define \n


endef

FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(OPTIMIZE) $(VERSION) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER))
PYTHON_FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(VERSION) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER) -fno-var-tracking-assignments)
MINIMAL_DEPS = Makefile config.mk makeIncludes/general.mk makeIncludes/warnings.mk makeIncludes/optimization.mk


# ------------------------------------------------------------------------------------------------------
#					Load dependency files
# ------------------------------------------------------------------------------------------------------

-include $(XERUS_DEPS)
-include $(MISC_DEPS)
-include $(TEST_DEPS)
-include $(UNIT_TEST_DEPS)
-include $(TUTORIAL_DEPS)
-include build/.preCompileHeaders/xerus.h.d


# ------------------------------------------------------------------------------------------------------
#					Custom functions
# ------------------------------------------------------------------------------------------------------

# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
	$(if $(value $1),, \
		$(error Undefined $1$(if $2, ($2))$(if $(value @), \
				required by target `$@')))

# Check that given variables are declared (they may have empty values),
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_declared = \
	$(strip $(foreach 1,$1, \
		$(call __check_defined,$1,$(strip $(value 2)))))
__check_declared = \
	$(if $(filter undefined,$(origin $1)), \
		$(error Undeclared $1$(if $2, ($2))$(if $(value @), \
                required by target '$@' (variable may be empty but must be defined))))

# Check that a variable specified through the stem is defined and has
# a non-empty value, die with an error otherwise.
#
#   %: The name of the variable to test.
#
check-defined-%: __check_defined_FORCE
    @:$(call check_defined, $*, target-specific)

# Since pattern rules can't be listed as prerequisites of .PHONY,
# we use the old-school and hackish FORCE workaround.
# You could go without this, but otherwise a check can be missed
# in case a file named like `check-defined-...` exists in the root
# directory, e.g. left by an accidental `make -t` invocation.
.PHONY: __check_defined_FORCE
__check_defined_FORCE:


# ------------------------------------------------------------------------------------------------------
#					Make Rules
# ------------------------------------------------------------------------------------------------------

opt:
	$(CXX) $(FLAGS) -Q --help=optimizers


warn:
	$(CXX) $(FLAGS) -Q --help=warnings


# Fake rule to create arbitary headers, to prevent errors if files are moved/renamed
%.h:

shared: build/libxerus_misc.so build/libxerus.so

.PHONY: libxerus_misc_dependencies
libxerus_misc_dependencies:
	@:$(call check_defined, BOOST_LIBS, include and link paths)
build/libxerus_misc.so: $(MINIMAL_DEPS) $(MISC_SOURCES) | libxerus_misc_dependencies
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,libxerus_misc.so $(FLAGS) -I include $(MISC_SOURCES) -Wl,--as-needed $(CALLSTACK_LIBS) $(BOOST_LIBS) -o build/libxerus_misc.so

.PHONY: libxerus_dependencies
libxerus_dependencies:
	@:$(call check_defined, SUITESPARSE LAPACK_LIBRARIES BLAS_LIBRARIES, include and link paths)
build/libxerus.so: $(MINIMAL_DEPS) $(XERUS_SOURCES) build/libxerus_misc.so | libxerus_dependencies
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,libxerus.so $(FLAGS) -I include $(XERUS_SOURCES) -L ./build/ -Wl,--as-needed -lxerus_misc $(SUITESPARSE) $(LAPACK_LIBRARIES) $(ARPACK_LIBRARIES) $(BLAS_LIBRARIES) -o build/libxerus.so


python2: build/python2/xerus.so
python3: build/python3/xerus.so

build/python2/xerus.so: $(MINIMAL_DEPS) $(PYTHON_SOURCES) src/xerus/python/misc.h build/libxerus.so
	@:$(call check_defined, PYTHON2_CONFIG BOOST_PYTHON2, include and link paths)
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,xerus.so $(PYTHON2_CONFIG) $(PYTHON_FLAGS) -I include -I 3rdParty/pybind11/include $(PYTHON_SOURCES) -L ./build/ -Wl,--as-needed -lxerus $(BOOST_PYTHON2) -o $@

build/python3/xerus.so: $(MINIMAL_DEPS) $(PYTHON_SOURCES) src/xerus/python/misc.h build/libxerus.so
	@:$(call check_defined, PYTHON3_CONFIG BOOST_PYTHON3, include and link paths)
	mkdir -p $(dir $@)
	@# -fpermissive is needed because of a bug in the definition of BOOST_PYTHON_MODULE_INIT in <boost/python/module_init.h>
	$(CXX) -shared -fPIC -Wl,-soname,xerus.so $(PYTHON3_CONFIG) $(PYTHON_FLAGS) -fpermissive -I include -I 3rdParty/pybind11/include $(PYTHON_SOURCES) -L ./build/ -Wl,--as-needed -lxerus -o $@


static: build/libxerus_misc.a build/libxerus.a

build/libxerus_misc.a: $(MINIMAL_DEPS) $(MISC_OBJECTS) | libxerus_misc_dependencies
	mkdir -p $(dir $@)
ifdef USE_LTO
	gcc-ar rcs ./build/libxerus_misc.a $(MISC_OBJECTS)
else
	ar rcs ./build/libxerus_misc.a $(MISC_OBJECTS)
endif

build/libxerus.a: $(MINIMAL_DEPS) $(XERUS_OBJECTS) | libxerus_dependencies
	mkdir -p $(dir $@)
ifdef USE_LTO
	gcc-ar rcs ./build/libxerus.a $(XERUS_OBJECTS)
else
	ar rcs ./build/libxerus.a $(XERUS_OBJECTS)
endif


ifdef INSTALL_PYTHON2_PATH
install: build/python2/xerus.so
endif
ifdef INSTALL_PYTHON3_PATH
install: build/python3/xerus.so
endif
ifdef INSTALL_LIB_PATH
ifdef INSTALL_HEADER_PATH
install: shared
	test -d $(strip $(INSTALL_LIB_PATH));
	test -d $(strip $(INSTALL_HEADER_PATH));
	@printf "Installing libxerus.so to $(strip $(INSTALL_LIB_PATH)) and storing the header files in $(strip $(INSTALL_HEADER_PATH)).\n"
	cp include/xerus.h $(INSTALL_HEADER_PATH)
	cp -r include/xerus $(INSTALL_HEADER_PATH)
	cp build/libxerus_misc.so $(INSTALL_LIB_PATH)
	cp build/libxerus.so $(INSTALL_LIB_PATH)
ifdef INSTALL_PYTHON2_PATH
	test -d $(strip $(INSTALL_PYTHON2_PATH));
	@printf "Installing xerus.so to $(strip $(INSTALL_PYTHON2_PATH)).\n"
	cp build/python2/xerus.so $(INSTALL_PYTHON2_PATH)
endif
ifdef INSTALL_PYTHON3_PATH
	test -d $(strip $(INSTALL_PYTHON3_PATH));
	@printf "Installing xerus.so to $(strip $(INSTALL_PYTHON3_PATH)).\n"
	cp build/python3/xerus.so $(INSTALL_PYTHON3_PATH)
endif
else
install:
	@printf "Cannot install xerus: INSTALL_HEADER_PATH not set. Please set the path in config file.\n"
endif
else
install:
	@printf "Cannot install xerus: INSTALL_LIB_PATH not set.  Please set the path in config file.\n"
endif

$(TEST_NAME): $(MINIMAL_DEPS) $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS) build/libxerus.a build/libxerus_misc.a | libxerus_misc_dependencies libxerus_dependencies
	$(CXX) -D XERUS_UNITTEST $(FLAGS) $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS) build/libxerus.a build/libxerus_misc.a $(SUITESPARSE) $(LAPACK_LIBRARIES) $(ARPACK_LIBRARIES) $(BLAS_LIBRARIES) $(BOOST_LIBS) $(CALLSTACK_LIBS) -o $(TEST_NAME)

build/print_boost_version: src/print_boost_version.cpp
	@$(CXX) -o $@ $<

printBoostVersion: build/print_boost_version
	@build/print_boost_version

ifdef ACTIVATE_CODE_COVERAGE
ifdef BROCKEN_CI
test:
	mkdir -p build
	make $(TEST_NAME)
	@cat build/build_output.txt | grep "REQUIRE_TEST @" > build/required_tests.txt
	./$(TEST_NAME) all
else

MAKE_PID := $(shell echo $$PPID)
JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))

test:
	mkdir -p build
	make $(TEST_NAME) $(JOB_FLAG) &> build/build_output.txt || cat build/build_output.txt
	@cat build/build_output.txt | grep "REQUIRE_TEST @" > build/required_tests.txt
	./$(TEST_NAME) all
endif
else
test:  $(TEST_NAME)
	./$(TEST_NAME) all
endif


.PHONY: test_python2_dependencies
test_python2_dependencies:
	@:$(call check_defined, PYTEST2, pytest executable)
test_python2: build/libxerus.so build/python2/xerus.so | test_python2_dependencies
	@PYTHONPATH=build/python2:${PYTHONPATH} LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} $(PYTEST2) src/pyTests

.PHONY: test_python3_dependencies
test_python3_dependencies:
	@:$(call check_defined, PYTEST3, pytest executable)
test_python3: build/libxerus.so build/python3/xerus.so | test_python3_dependencies
	@PYTHONPATH=build/python3:${PYTHONPATH} LD_LIBRARY_PATH=build:${LD_LIBRARY_PATH} $(PYTEST3) src/pyTests


fullTest: $(TUTORIALS) $(TEST_NAME)
	$(foreach x,$(TUTORIALS),./$(x)$(\n))
	./$(TEST_NAME) all


doc:
	make -C doc doc


clean:
	rm -fr build
	-rm -f $(TEST_NAME)
	-rm -f include/xerus.h.gch
	make -C doc clean

version:
	@echo $(XERUS_MAJOR_V).$(XERUS_MINOR_V).$(XERUS_REVISION_V)



# Build rule for normal misc objects
build/.miscObjects/%.o: %.cpp $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -I include $< -c $(FLAGS) -MMD -o $@


# Build rule for normal lib objects
build/.libObjects/%.o: %.cpp $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -I include $< -c $(FLAGS) -MMD -o $@


# Build rule for test lib objects
build/.testObjects/%.o: %.cpp $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I include $< -c $(FLAGS) -MMD -o $@


# Build rule for unit test objects
ifdef USE_GCC
build/.unitTestObjects/%.o: %.cxx $(MINIMAL_DEPS) build/.preCompileHeaders/xerus.h.gch build/.preCompileHeaders/common.hxx.gch
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I build/.preCompileHeaders $< -c $(FLAGS) -MMD -o $@
else
build/.unitTestObjects/%.o: %.cxx $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I include -I src/unitTests $< -c $(FLAGS) -MMD -o $@
endif


# Build and execution rules for tutorials
build/.tutorialObjects/%: %.cpp $(MINIMAL_DEPS) build/libxerus.a build/libxerus_misc.a
	mkdir -p $(dir $@)
	$(CXX) -I include $< build/libxerus.a build/libxerus_misc.a $(SUITESPARSE) $(LAPACK_LIBRARIES) $(ARPACK_LIBRARIES) $(BLAS_LIBRARIES) $(BOOST_LIBS) $(CALLSTACK_LIBS) $(FLAGS) -MMD -o $@


# Build rule for the preCompileHeader
build/.preCompileHeaders/xerus.h.gch: include/xerus.h $(MINIMAL_DEPS) .git/ORIG_HEAD
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST $< -c $(FLAGS) -MMD -o $@


# Build rule for the other preCompileHeader
build/.preCompileHeaders/common.hxx.gch: src/unitTests/common.hxx $(MINIMAL_DEPS) .git/ORIG_HEAD
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST $< -c $(FLAGS) -MMD -o $@


# dummy rule in case files were downloaded without git
.git/ORIG_HEAD:
	mkdir -p .git
	touch .git/ORIG_HEAD

