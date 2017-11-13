# ------------------------------------------------------------------------------------------------------
#				Default rule should be the help message
# ------------------------------------------------------------------------------------------------------
help:
	@printf "Possible make targets are:\n \
	\t\tshared \t\t -- Build xerus as a shared library.\n \
	\t\tstatic \t\t -- Build xerus as a static library.\n \
	\t\tpython \t\t -- Build the xerus python wrappers.\n \
	\t\tdoc \t\t -- Build the html documentation for the xerus library.\n \
	\t\tinstall \t -- Install the shared library and header files (may require root).\n \
	\t\ttest \t\t -- Build and run the xerus unit tests.\n \
	\t\tclean \t\t -- Remove all object, library and executable files.\n"

	
# ------------------------------------------------------------------------------------------------------
#				Set the names of the resulting binary codes 
# ------------------------------------------------------------------------------------------------------

# Name of the test executable
TEST_NAME = XerusTest

# xerus version from VERSION file
XERUS_VERSION = $(shell git describe --tags --always 2>/dev/null || cat VERSION)
DEBUG += -D XERUS_VERSION="$(XERUS_VERSION)"

XERUS_MAJOR_V = $(word 1, $(subst ., ,$(XERUS_VERSION)) )
ifneq (,$(findstring v, $(XERUS_MAJOR_V)))
	XERUS_MAJOR_V := $(strip $(subst v, ,$(XERUS_MAJOR_V)) )
endif
DEBUG += -DXERUS_VERSION_MAJOR=$(XERUS_MAJOR_V)
XERUS_MINOR_V = $(word 2, $(subst ., ,$(XERUS_VERSION)) )
DEBUG += -DXERUS_VERSION_MINOR=$(XERUS_MINOR_V)
XERUS_REVISION_V = $(word 3, $(subst ., ,$(XERUS_VERSION)) )
ifneq (,$(findstring -, $(XERUS_REVISION_V)))
	XERUS_COMMIT_V := $(word 2, $(subst -, ,$(XERUS_REVISION_V)) )
	XERUS_REVISION_V := $(word 1, $(subst -, ,$(XERUS_REVISION_V)) )
else
	XERUS_COMMIT_V = 0
endif
DEBUG += -DXERUS_VERSION_REVISION=$(XERUS_REVISION_V)
DEBUG += -DXERUS_VERSION_COMMIT=$(XERUS_COMMIT_V)


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

PYTHON_OBJECTS = $(PYTHON_SOURCES:%.cpp=build/.pythonObjects/%.o)
PYTHON_DEPS    = $(PYTHON_SOURCES:%.cpp=build/.pyhtonObjects/%.d)

TEST_OBJECTS = $(TEST_SOURCES:%.cpp=build/.testObjects/%.o)
TEST_DEPS    = $(TEST_SOURCES:%.cpp=build/.testObjects/%.d)

UNIT_TEST_OBJECTS = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.o)
UNIT_TEST_DEPS    = $(UNIT_TEST_SOURCES:%.cxx=build/.unitTestObjects/%.d)

TUTORIALS 	= $(TUTORIAL_SOURCES:%.cpp=build/.tutorialObjects/%)
TUTORIAL_DEPS   = $(TUTORIAL_SOURCES:%.cpp=build/.tutorialObjects/%.d)


# ------------------------------------------------------------------------------------------------------
#		Load the configurations provided by the user and set up general options
# ------------------------------------------------------------------------------------------------------
include config.mk

include makeIncludes/general.mk
include makeIncludes/warnings.mk
include makeIncludes/optimization.mk


# ------------------------------------------------------------------------------------------------------
#		  			Set additional compiler options 
# ------------------------------------------------------------------------------------------------------




# ------------------------------------------------------------------------------------------------------
#		  			Convinience variables
# ------------------------------------------------------------------------------------------------------

# Small hack to get newlines...
define \n


endef

FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(OPTIMIZE) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER))
PYTHON_FLAGS = $(strip $(COMPATIBILITY) $(WARNINGS) $(LOGGING) $(DEBUG) $(ADDITIONAL_INCLUDE) $(OTHER) -fno-var-tracking-assignments)
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
#					Make Rules      
# ------------------------------------------------------------------------------------------------------

opt:
	$(CXX) $(FLAGS) -Q --help=optimizers


warn:
	$(CXX) $(FLAGS) -Q --help=warnings


# Fake rule to create arbitary headers, to prevent errors if files are moved/renamed
%.h: 

ifdef BUILD_PYTHON_BINDINGS
shared: build/libxerus_misc.so build/libxerus.so build/xerus.so
else
shared: build/libxerus_misc.so build/libxerus.so 
endif

build/libxerus_misc.so: $(MINIMAL_DEPS) $(MISC_SOURCES)
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,libxerus_misc.so $(FLAGS) -I include $(MISC_SOURCES) -Wl,--as-needed $(CALLSTACK_LIBS) -lboost_filesystem -o build/libxerus_misc.so

build/libxerus.so: $(MINIMAL_DEPS) $(XERUS_SOURCES) build/libxerus_misc.so
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,libxerus.so $(FLAGS) -I include $(XERUS_SOURCES) -L ./build/ -Wl,--as-needed -lxerus_misc $(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) -o build/libxerus.so


python: build/xerus.so

build/xerus.so: $(MINIMAL_DEPS) $(PYTHON_SOURCES) build/libxerus.so
	mkdir -p $(dir $@)
	$(CXX) -shared -fPIC -Wl,-soname,xerus.so `python2-config --cflags --ldflags` $(PYTHON_FLAGS) -I include $(PYTHON_SOURCES) -L ./build/ -Wl,--as-needed -lxerus $(BOOST_PYTHON) -o build/xerus.so


static: build/libxerus_misc.a build/libxerus.a

build/libxerus_misc.a: $(MINIMAL_DEPS) $(MISC_OBJECTS)
	mkdir -p $(dir $@)
ifdef USE_LTO
	gcc-ar rcs ./build/libxerus_misc.a $(MISC_OBJECTS)
else
	ar rcs ./build/libxerus_misc.a $(MISC_OBJECTS)
endif

build/libxerus.a: $(MINIMAL_DEPS) $(XERUS_OBJECTS)
	mkdir -p $(dir $@)
ifdef USE_LTO
	gcc-ar rcs ./build/libxerus.a $(XERUS_OBJECTS)
else
	ar rcs ./build/libxerus.a $(XERUS_OBJECTS)
endif


ifdef DESTDIR
	INSTALL_LIB_PATH = $(DESTDIR)/lib/
	INSTALL_HEADER_PATH = $(DESTDIR)/include/
	INSTALL_PYTHON_PATH = $(DESTDIR)/lib/python/site-packages/
endif


ifdef INSTALL_LIB_PATH
ifdef INSTALL_HEADER_PATH
install: shared
	@printf "Installing libxerus.so to $(strip $(INSTALL_LIB_PATH)) and storing the header files in $(strip $(INSTALL_HEADER_PATH)).\n"
	mkdir -p $(INSTALL_LIB_PATH)
	mkdir -p $(INSTALL_HEADER_PATH)
	cp include/xerus.h $(INSTALL_HEADER_PATH)
	cp -r include/xerus $(INSTALL_HEADER_PATH)
	cp build/libxerus_misc.so $(INSTALL_LIB_PATH)
	cp build/libxerus.so $(INSTALL_LIB_PATH)
ifdef BUILD_PYTHON_BINDINGS
	cp build/xerus.so $(INSTALL_PYTHON_PATH)
endif
else
install:
	@printf "Cannot install xerus: INSTALL_HEADER_PATH not set. Please set the path in config file.\n"
endif
else
install:
	@printf "Cannot install xerus: INSTALL_LIB_PATH not set.  Please set the path in config file.\n"
endif


$(TEST_NAME): $(MINIMAL_DEPS) $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS) build/libxerus.a build/libxerus_misc.a
	$(CXX) -D XERUS_UNITTEST $(FLAGS) $(UNIT_TEST_OBJECTS) $(TEST_OBJECTS) build/libxerus.a build/libxerus_misc.a $(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) $(CALLSTACK_LIBS) -o $(TEST_NAME)


test:  $(TEST_NAME)
	./$(TEST_NAME) all


fullTest: $(TUTORIALS) $(TEST_NAME)
	$(foreach x,$(TUTORIALS),./$(x)$(\n))
	./$(TEST_NAME) all


.FORCE:
doc: .FORCE doc/parseDoxytags doc/findDoxytag
	make -C doc doc


clean:
	rm -fr build
	-rm -f $(TEST_NAME)
	-rm -f include/xerus.h.gch
	make -C doc clean



benchmark: $(MINIMAL_DEPS) $(LOCAL_HEADERS) benchmark.cxx $(LIB_NAME_STATIC)
	$(CXX) $(FLAGS) benchmark.cxx $(LIB_NAME_STATIC) $(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) $(CALLSTACK_LIBS) -lboost_filesystem -lboost_system -o Benchmark

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


# Build rule for benchmark objects
build/%.o: %.cpp $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I include $< -c $(FLAGS) -MMD -o $@


# Build rule for unit test objects
ifdef USE_GCC
build/.unitTestObjects/%.o: %.cxx $(MINIMAL_DEPS) build/.preCompileHeaders/xerus.h.gch
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I build/.preCompileHeaders $< -c $(FLAGS) -MMD -o $@
else
build/.unitTestObjects/%.o: %.cxx $(MINIMAL_DEPS)
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST -I include $< -c $(FLAGS) -MMD -o $@
endif


# Build and execution rules for tutorials
build/.tutorialObjects/%: %.cpp $(MINIMAL_DEPS) build/libxerus.a build/libxerus_misc.a
	mkdir -p $(dir $@)
	$(CXX) -I include $< build/libxerus.a build/libxerus_misc.a $(SUITESPARSE) $(LAPACK_LIBRARIES) $(BLAS_LIBRARIES) $(CALLSTACK_LIBS) $(FLAGS) -MMD -o $@

	
# Build rule for the preCompileHeader
build/.preCompileHeaders/xerus.h.gch: include/xerus.h $(MINIMAL_DEPS) .git/ORIG_HEAD
	mkdir -p $(dir $@)
	$(CXX) -D XERUS_UNITTEST $< -c $(FLAGS) -MMD -o $@


# dummy rule in case files were downloaded without git
.git/ORIG_HEAD:
	mkdir -p .git
	touch .git/ORIG_HEAD

