# Sets WARNINGS
# Uses USE_CLANG, SUGGEST_ATTRIBUTES

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Activate/Disable Additinal Warnings - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
#General Warning & Error options
WARNINGS += -Wall						# Show "all" warnings
WARNINGS += -Wextra						# Show more warnings
WARNINGS += -pedantic						# Strict ISO C++ checks (warn everywhere the ISO says there should be a warning)


# Additionall Warnings not aktivated by any level
WARNINGS += -Wundef						# Warn if an undefined identifier is evaluated in an ‘#if’ directive. 
WARNINGS += -Wunreachable-code					# Warn about unreachable code
WARNINGS += -Wdisabled-optimization	                	# Warn if a requested optimization pass is disabled
WARNINGS += -Wcast-qual						# Warn whenever a pointer is cast so as to remove a type qualifier from the target type. 
WARNINGS += -Wsign-promo					# Warn when overload resolution chooses a promotion from unsigned or enumerated type to a signed type
WARNINGS += -Winit-self						# Warn about uninitialized variables that are initialized with themselves.
WARNINGS += -Wnon-virtual-dtor					# Warn when a class has virtual functions and an accessible non-virtual destructor.
WARNINGS += -Woverloaded-virtual				# Warn when a function declaration hides virtual functions from a base class.

ifndef USE_ICC
	WARNINGS += -Wconversion					# warn double -> uint conversions etc.
	WARNINGS += -Wfloat-equal					# Warn if floating-point values are used in equality comparisons. 
	WARNINGS += -Wshadow						# Warn about variables being shadowed (for ICC this is incompatible with openmp?)
	WARNINGS += -Wswitch-default					# Warn whenever a switch statement does not have a default case. 
	WARNINGS += -Wpacked						# misaligned structs (reordering of the fields might help)
	WARNINGS += -Wcast-align					# Warn whenever a pointer is cast such that the required alignment of the target is increased. 
	WARNINGS += -Wctor-dtor-privacy					# Warn when a class seems unusable.
	WARNINGS += -Wold-style-cast					# Warn if an old-style (C-style) cast to a non-void type is used within a C++ program. 
endif

ifdef USE_CLANG
	WARNINGS += -Wlogical-op-parentheses				# Warn about suspicious uses of logical operators in expressions
	WARNINGS += -Wno-error=return-type-c-linkage			# do not warn about c-incompatible return types in "extern-c" blocks
else ifdef USE_GCC
	WARNINGS += -Wuseless-cast 					# Warn when an expression is casted to its own type. 
	WARNINGS += -Wno-error=useless-cast				# fails on gcc 5.1.1 otherwise
	WARNINGS += -Wlogical-op					# Warn about suspicious uses of logical operators in expressions
	WARNINGS += -Wtrampolines					# Warn about trampolines
#	WARNINGS += -Wzero-as-null-pointer-constant			# Warn when a literal '0' is used as null pointer constant.
	WARNINGS += -Wnoexcept						# Warn when a noexcept-expression evaluates to false because of a call to a function that does not have noexcept but is known by the compiler to never throw. 
 	WARNINGS += -Wsuggest-override					# Suggest functions that do override

	ifdef SUGGEST_ATTRIBUTES
		WARNINGS += -Wsuggest-attribute=pure 			# Suggest functions that can be pure
		WARNINGS += -Wsuggest-attribute=const 			# Suggest functions that can be const
		WARNINGS += -Wsuggest-attribute=noreturn 		# Suggest functions that do not return
		WARNINGS += -Wsuggest-attribute=format 			# 
		WARNINGS += -Wsuggest-final-methods		 	# 
		WARNINGS += -Wsuggest-final-types		 	# 
		WARNINGS += -Wno-error=suggest-attribute=pure 		# Do not promote to Error
		WARNINGS += -Wno-error=suggest-attribute=const 		# Do not promote to Error
		WARNINGS += -Wno-error=suggest-attribute=noreturn 	# Do not promote to Error
		WARNINGS += -Wno-error=suggest-attribute=format 	# Do not promote to Error
		WARNINGS += -Wno-error=suggest-final-methods		# Do not promote to Error
		WARNINGS += -Wno-error=suggest-final-types		# Do not promote to Error
	endif
endif

# Disabled Warnings
WARNINGS += -Wno-comment					# Ignore warning about /* inside /*...*/
WARNINGS += -Wno-unknown-pragmas				# Allow unknwon pragmas (e.g. openmp)
WARNINGS += -Wno-unused-parameter				# No warning about unused parameters

ifdef USE_ICC
	WARNINGS += -wd2304					# disable ICC's "warning #2304: non-explicit constructor with single argument may cause implicit type conversion"
	WARNINGS += -wd2305					# disable ICC's "error #2305: declaration of 'explicit' constructor without a single argument is redundant"
endif

## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Error Options - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##
ifdef STRICT_WARNINGS

	WARNINGS += -Werror				        	# Promote ALL Warnings to Errors
	
	ifdef USE_CLANG
		WARNINGS += -ferror-limit=3				# Abort at third error
	else
		WARNINGS += -fmax-errors=3				# Abort at third error
	endif

	# Exceptions from Error Promotion
	WARNINGS += -Wno-error=unused-parameter				# Do not Promote to Error
	WARNINGS += -Wno-error=unused-variable				# Do not Promote to Error
	WARNINGS += -Wno-error=disabled-optimization			# Do not Promote to Error
	
	ifndef USE_CLANG
		WARNINGS += -Wno-error=unused-but-set-variable		# Do not Promote to Error
		WARNINGS += -Wno-error=maybe-uninitialized		# Do not Promote to Error
	endif
endif
