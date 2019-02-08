.PHONY: help doc clean serve

# ------------------------------------------------------------------------------------------------------
#				Default rule should be the help message
# ------------------------------------------------------------------------------------------------------
help:
	@printf "Possible make targets are:\n \
	\t\thelp \t\t -- Print this help.\n \
	\t\tdoc \t\t -- Build the html documentation for the xerus library.\n \
	\t\tserve \t\t -- Build the html documentation for the xerus library and offer it via 'jekyll serve'.\n \
	\t\tclean \t\t -- Remove all documentation files.\n"

doc: parseDoxytags findDoxytag
	-mkdir html
	doxygen doxygen/Doxyfile
	./parseDoxytags
	jekyll build --source jekyll/ --destination html/

clean:
	-rm -rf html
	-rm -f parseDoxytags findDoxytag
	-rm -f xerus.tags xerus.tagfile

serve: parseDoxytags findDoxytag
	-mkdir html
	doxygen doxygen/Doxyfile
	./parseDoxytags
	jekyll serve --source jekyll/ --destination html/

include ../config.mk
include ../makeIncludes/general.mk
include ../makeIncludes/warnings.mk
include ../makeIncludes/optimization.mk

FLAGS = $(strip $(WARNINGS) $(OPTIMIZE) $(OTHER)) -std=c++11

parseDoxytags: ../src/docHelper/parseDoxytags.cpp
	$(CXX) $(FLAGS) ../src/docHelper/parseDoxytags.cpp -o parseDoxytags


findDoxytag: ../src/docHelper/findDoxytag.cpp
	$(CXX) $(FLAGS) ../src/docHelper/findDoxytag.cpp -o findDoxytag
