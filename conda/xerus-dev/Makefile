.PHONY: test

TEST_NAME = XerusTest

TEST_SOURCES = $(wildcard src/xerus/test/*.cpp)

UNIT_TEST_SOURCES = $(wildcard src/unitTests/*.cxx)

TEST_OBJECTS = $(TEST_SOURCES:%.cpp=build/.testObjects/%.o)
TEST_DEPS    = $(TEST_SOURCES:%.cpp=build/.testObjects/%.d)

UNIT_TEST_OBJECTS = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.o)
UNIT_TEST_DEPS    = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.d)

# ------------------------------------------------------------------------------------------------------
#		Load the configurations provided by the user and set up general options
# ------------------------------------------------------------------------------------------------------
include config.mk

include makeIncludes/general.mk
include makeIncludes/warnings.mk
include makeIncludes/optimization.mk

FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(OPTIMIZE) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER))

# ------------------------------------------------------------------------------------------------------
#					Load dependency files
# ------------------------------------------------------------------------------------------------------

-include $(TEST_DEPS)
-include $(UNIT_TEST_DEPS)
-include build/.preCompileHeaders/xerus.h.d

# Fake rule to create arbitary headers, to prevent errors if files are moved/renamed
%.h:

test: $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS)
	$(CXX) -D XERUS_UNITTEST $(FLAGS) $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS) -lxerus -lxerus_misc $(SUITESPARSE) $(LAPACK_LIBRARIES) $(ARPACK_LIBRARIES) $(BLAS_LIBRARIES) $(CALLSTACK_LIBS) -o $(TEST_NAME)
	$(TEST_NAME) all
	pytest src/pyTests
