# Sets: OPTIMIZE
# Uses: USE_CLANG, USE_LTO, LOW_OPTIMIZATION, HIGH_OPTIMIZATION, DANGEROUS_OPTIMIZATION, RIDICULOUS_OPTIMIZATION, FREE_RAM, COMPILE_THREADS, GRAPHITE_AVAILABLE

# Set default values

ifndef COMPILE_THREADS
	COMPILE_THREADS=1
endif

ifndef FREE_RAM
    FREE_RAM = 6291456
endif

# Allow GCC to use all the Free Ram available or 6 GB if nothin is set.
ifdef USE_FREE_RAM
	OPTIMIZE += --param ggc-min-heapsize=$(FREE_RAM)    # Allow GCC to use X KB ram before bothering to free any
endif


## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - Set Optimization Options - - - - - - - - - - - - - - - - - - - - - - - - - - - - - ##

# Set LTO usage
ifdef USE_LTO
	DEBUG += -D USE_LTO
	ifdef USE_GCC
		OPTIMIZE += -flto=$(COMPILE_THREADS)	    # Use LTO at compiling using X threads
		OPTIMIZE += -fno-fat-lto-objects	        # No none LTO code in object files
	else ifdef USE_ICC
		OPTIMIZE += -ipo
		OPTIMZIE += -fno-fat-lto-objects		    # No none LTO code in object files
	else
		OPTIMIZE += -flto			                # Use LTO at compiling
	endif
endif

# Set optimization options
ifdef DEBUG_OPTIMIZATION
	DEBUG += -D DEBUG_OPTIMIZATION
	ifndef USE_CLANG
		OPTIMIZE += -Og -fno-inline -ggdb
	endif
else ifdef HIGH_OPTIMIZATION
	DEBUG += -D HIGH_OPTIMIZATION
	OPTIMIZE += -O3				# Even more optimization, using non iso conform C++ operations
	OPTIMIZE += -march=native			# Compile only for native architecture 
	
	# Stuff that is not enabled and might help
	ifdef USE_GCC
		OPTIMIZE += -fbranch-target-load-optimize	# Perform branch target register load optimization before prologue / epilogue threading. 
		OPTIMIZE += -fgcse-las				# When -fgcse-las is enabled, the global common subexpression elimination pass eliminates redundant loads that come after stores to the same memory location (both partial and full redundancies). 
		OPTIMIZE += -fgcse-sm				# When -fgcse-sm is enabled, a store motion pass is run after global common subexpression elimination.
		#Not working with LTO 		OPTIMIZE += -fipa-pta				# Perform interprocedural pointer analysis and interprocedural modification and reference analysis.
		OPTIMIZE += -fira-loop-pressure			# Use IRA to evaluate register pressure in loops for decisions to move loop invariants.
		
		#Optimizations that need Graphite
		ifdef GRAPHITE_AVAILABLE
			OPTIMIZE += -fgraphite-identity			# Enable the identity transformation for graphite.
			OPTIMIZE += -floop-block			# Perform loop blocking transformations on loops.
			OPTIMIZE += -floop-interchange			# Perform loop interchange transformations on loops.
			OPTIMIZE += -floop-nest-optimize		# Enable the ISL based loop nest optimizer.
			OPTIMIZE += -floop-strip-mine			# Perform loop strip mining transformations on loops.
		endif
# 		OPTIMIZE += -floop-unroll-and-jam		# Enable unroll and jam for the ISL based loop nest optimizer. 
		OPTIMIZE += -fmodulo-sched 			# Perform swing modulo scheduling immediately before the first scheduling pass. 
		OPTIMIZE += -fmodulo-sched-allow-regmoves 	# Perform more aggressive SMS-based modulo scheduling with register moves allowed. 
# 		OPTIMIZE += -fomit-frame-pointer		# Don't keep the frame pointer in a register for functions that don't need one.
		OPTIMIZE += -fsched-pressure			# Enable register pressure sensitive insn scheduling before register allocation. 
		OPTIMIZE += -fsched-spec-load			# Allow speculative motion of some load instructions. 
		OPTIMIZE += -fsched2-use-superblocks		# When scheduling after register allocation, use superblock scheduling.
		OPTIMIZE += -ftree-loop-distribution		# Perform loop distribution.
		OPTIMIZE += -ftree-vectorize			# Perform vectorization on trees.
	endif
else ifdef DANGEROUS_OPTIMIZATION
	DEBUG += -D DANGEROUS_OPTIMIZATION
	OPTIMIZE += -march=native			# Compile only for native architecture 
	OPTIMIZE += -Ofast				# Even more optimization, using non iso conform C++ operations
	
	# Stuff that is not enabled and might help
	ifdef USE_GCC
		OPTIMIZE += -fbranch-target-load-optimize	# Perform branch target register load optimization before prologue / epilogue threading. 
		OPTIMIZE += -fgcse-las				# When -fgcse-las is enabled, the global common subexpression elimination pass eliminates redundant loads that come after stores to the same memory location (both partial and full redundancies). 
		OPTIMIZE += -fgcse-sm				# When -fgcse-sm is enabled, a store motion pass is run after global common subexpression elimination.
		#Not working with LTO 		OPTIMIZE += -fipa-pta				# Perform interprocedural pointer analysis and interprocedural modification and reference analysis.
		OPTIMIZE += -fira-loop-pressure			# Use IRA to evaluate register pressure in loops for decisions to move loop invariants.
		#Optimizations that need Graphite
		ifdef GRAPHITE_AVAILABLE
			OPTIMIZE += -fgraphite-identity			# Enable the identity transformation for graphite.
			OPTIMIZE += -floop-block			# Perform loop blocking transformations on loops.
			OPTIMIZE += -floop-interchange			# Perform loop interchange transformations on loops.
			OPTIMIZE += -floop-nest-optimize		# Enable the ISL based loop nest optimizer.
			OPTIMIZE += -floop-strip-mine			# Perform loop strip mining transformations on loops.
		endif
# 		OPTIMIZE += -floop-unroll-and-jam		# Enable unroll and jam for the ISL based loop nest optimizer. 
		OPTIMIZE += -fmodulo-sched 			# Perform swing modulo scheduling immediately before the first scheduling pass. 
		OPTIMIZE += -fmodulo-sched-allow-regmoves 	# Perform more aggressive SMS-based modulo scheduling with register moves allowed. 
		OPTIMIZE += -fomit-frame-pointer		# Don't keep the frame pointer in a register for functions that don't need one.
		OPTIMIZE += -fsched-pressure			# Enable register pressure sensitive insn scheduling before register allocation. 
		OPTIMIZE += -fsched-spec-load			# Allow speculative motion of some load instructions. 
		OPTIMIZE += -fsched2-use-superblocks		# When scheduling after register allocation, use superblock scheduling.
		OPTIMIZE += -ftree-loop-distribution		# Perform loop distribution.
		OPTIMIZE += -ftree-vectorize			# Perform vectorization on trees.
		
		
		OPTIMIZE += -fcx-fortran-rules			# Complex multiplication and division follow Fortran rules.
		OPTIMIZE += -fsched-spec-load-dangerous 	# Allow speculative motion of more load instructions.
		OPTIMIZE += -funsafe-loop-optimizations 	# This option tells the loop optimizer to assume that loop indices do not overflow, and that loops with nontrivial exit condition are not infinite. 

		OPTIMIZE += --param max-vartrack-expr-depth=120
	endif
	
	# Potentially loose precicision
	OPTIMIZE += -freciprocal-math		# Allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations.
	
	# Potentially VERY dangerous
	OPTIMIZE += -fmerge-all-constants	# Attempt to merge identical constants and identical variables. 
	OPTIMIZE += -fno-math-errno 		# Do not set errno after calling math functions that are executed with a single instruction
	OPTIMIZE += -funsafe-math-optimizations # Allow optimizations for floating-point arithmetic that (a) assume that arguments and results are valid and (b) may violate IEEE or ANSI standards.
	OPTIMIZE += -ffinite-math-only		# Allow optimizations for floating-point arithmetic that assume that arguments and results are not NaNs or +-Infs. 
	OPTIMIZE += -fno-signed-zeros		# Allow optimizations for floating-point arithmetic that ignore the signedness of zero.
	OPTIMIZE += -fno-trapping-math		# Compile code assuming that floating-point operations cannot generate user-visible traps. These traps include division by zero, overflow, underflow, inexact result and invalid operation.
	ifndef USE_CLANG
		OPTIMIZE += -funsafe-loop-optimizations	# This option tells the loop optimizer to assume that loop indices do not overflow, and that loops with nontrivial exit condition are not infinite. 	
	endif
else ifdef RIDICULOUS_OPTIMIZATION
	DEBUG += -D RIDICULOUS_OPTIMIZATION
	OPTIMIZE += -march=native			# Compile only for native architecture 
	OPTIMIZE += -Ofast				# Even more optimization, using non iso conform C++ operations
	
	# Stuff that is not enabled and might help
	OPTIMIZE += -fgcse-sm				# attempts to move stores out of loops
	OPTIMIZE += -fgcse-las				# eliminates redundant loads that come after stores to the same memory location
	OPTIMIZE += -funswitch-loops			# Move branches with loop invariant conditions out of the loop
	OPTIMIZE += -fipa-pta				# cross-procedure optimization
	OPTIMIZE += -fbranch-target-load-optimize				
	OPTIMIZE += -fira-loop-pressure
	OPTIMIZE += -fsched-pressure
	OPTIMIZE += -fsched-spec-load-dangerous
	OPTIMIZE += -ftree-vectorize
	OPTIMIZE += -funroll-loops
	OPTIMIZE += -fopt-info-missed=horst.dat
	
	OPTIMIZE += -fsched-stalled-insns=1000		# Define how many insns (if any) can be moved prematurely from the queue of stalled insns into the ready list during the second scheduling pass.
	OPTIMIZE += -fsched-stalled-insns-dep=1000 	# Define how many insn groups (cycles) are examined for a dependency on a stalled insn that is a candidate for premature removal from the queue of stalled insns.
	OPTIMIZE += -fvariable-expansion-in-unroller	# With this option, the compiler creates multiple copies of some local variables when unrolling a loop, which can result in superior code. 
	
	OPTIMIZE += --param inline-min-speedup=5                        #The minimal estimated speedup allowing inliner to ignore inline-insns-single and inline-isnsns-auto 
	OPTIMIZE += --param max-inline-insns-single=100000              #The maximum number of instructions in a single function eligible for inlining 
	OPTIMIZE += --param max-inline-insns-auto=1000                  #The maximum number of instructions when automatically inlining 
	OPTIMIZE += --param max-inline-insns-recursive=100000           #The maximum number of instructions inline function can grow to via recursive inlining 
	OPTIMIZE += --param max-inline-insns-recursive-auto=1000        #The maximum number of instructions non-inline function can grow to via recursive inlining 
	OPTIMIZE += --param max-inline-recursive-depth=1000             #The maximum depth of recursive inlining for inline functions 
	OPTIMIZE += --param max-inline-recursive-depth-auto=1000        #The maximum depth of recursive inlining for non-inline functions 
	#OPTIMIZE += --param min-inline-recursive-probability=1000      #Inline recursively only when the probability of call being executed exceeds the parameter 
	OPTIMIZE += --param max-early-inliner-iterations=1000           #The maximum number of nested indirect inlining performed by early inliner 
	#OPTIMIZE += --param comdat-sharing-probability=1000            #Probability that COMDAT function will be shared with different compilation unit 
	#OPTIMIZE += --param partial-inlining-entry-probability=1000    #Maximum probability of the entry BB of split region (in percent relative to entry BB of the function) to make partial inlining happen 
	#OPTIMIZE += --param max-variable-expansions-in-unroller=1000   #If -fvariable-expansion-in-unroller is used, the maximum number of times that an individual variable will be expanded during loop unrolling 
#--->	OPTIMIZE += --param min-vect-loop-bound=0                       #If -ftree-vectorize is used, the minimal loop bound of a loop to be considered for vectorization 
	OPTIMIZE += --param max-delay-slot-insn-search=100000           #The maximum number of instructions to consider to fill a delay slot 
	OPTIMIZE += --param max-delay-slot-live-search=100000           #The maximum number of instructions to consider to find accurate live register information 
	OPTIMIZE += --param max-pending-list-length=100000              #The maximum length of scheduling's pending operations list 
	OPTIMIZE += --param max-modulo-backtrack-attempts=100000        #The maximum number of backtrack attempts the scheduler should make when modulo scheduling a loop 
	OPTIMIZE += --param large-function-insns=100000                 #The size of function body to be considered large 
	OPTIMIZE += --param large-function-growth=500                   #Maximal growth due to inlining of large function (in percent) 
	OPTIMIZE += --param large-unit-insns=100000                      #The size of translation unit to be considered large 
	OPTIMIZE += --param inline-unit-growth=500                      #How much can given compilation unit grow because of the inlining (in percent) 
	OPTIMIZE += --param ipcp-unit-growth=100                        #How much can given compilation unit grow because of the interprocedural constant propagation (in percent) 
	OPTIMIZE += --param early-inlining-insns=100                    #Maximal estimated growth of function body caused by early inlining of single call 
	#OPTIMIZE += --param large-stack-frame=256                      #The size of stack frame to be considered large 
	#OPTIMIZE += --param large-stack-frame-growth=1000              #Maximal stack frame growth due to inlining (in percent) 
	OPTIMIZE += --param max-gcse-memory=14508032                     #The maximum amount of memory to be allocated by GCSE 
	#OPTIMIZE += --param max-gcse-insertion-ratio=20                #The maximum ratio of insertions to deletions of expressions in GCSE 
	#OPTIMIZE += --param gcse-after-reload-partial-fraction=1000    #The threshold ratio for performing partial redundancy elimination after reload 
	#OPTIMIZE += --param gcse-after-reload-critical-fraction=1000   #The threshold ratio of critical edges execution count that permit performing redundancy elimination after reload 
	OPTIMIZE += --param gcse-cost-distance-ratio=100                #Scaling factor in calculation of maximum distance an expression can be moved by GCSE optimizations 
	OPTIMIZE += --param gcse-unrestricted-cost=1                    #Cost at which GCSE optimizations will not constraint the distance an expression can travel 
	OPTIMIZE += --param max-hoist-depth=0                           #Maximum depth of search in the dominator tree for expressions to hoist 
	OPTIMIZE += --param max-unrolled-insns=1000                     #The maximum number of instructions to consider to unroll in a loop 
	OPTIMIZE += --param max-average-unrolled-insns=1000             #The maximum number of instructions to consider to unroll in a loop on average 
	OPTIMIZE += --param max-unroll-times=1000                       #The maximum number of unrollings of a single loop 
	OPTIMIZE += --param max-peeled-insns=1000                       #The maximum number of insns of a peeled loop 
	OPTIMIZE += --param max-peel-times=1000                         #The maximum number of peelings of a single loop 
	OPTIMIZE += --param max-peel-branches=1000                      #The maximum number of branches on the path through the peeled sequence 
	OPTIMIZE += --param max-completely-peeled-insns=1000            #The maximum number of insns of a completely peeled loop 
	OPTIMIZE += --param max-completely-peel-times=1000              #The maximum number of peelings of a single loop that is peeled completely 
	OPTIMIZE += --param max-once-peeled-insns=1000                  #The maximum number of insns of a peeled loop that rolls only once 
	OPTIMIZE += --param max-completely-peel-loop-nest-depth=1000    #The maximum depth of a loop nest we completely peel 
	OPTIMIZE += --param max-unswitch-insns=1000                     #The maximum number of insns of an unswitched loop 
	OPTIMIZE += --param max-unswitch-level=1000                     #The maximum number of unswitchings in a single loop 
	OPTIMIZE += --param max-iterations-to-track=100000              #Bound on the number of iterations the brute force # of iterations analysis algorithm evaluates 
	OPTIMIZE += --param max-iterations-computation-cost=1000        #Bound on the cost of an expression to compute the number of iterations 
	#OPTIMIZE += --param sms-max-ii-factor=1000                     #A factor for tuning the upper bound that swing modulo scheduler uses for scheduling a loop 
	#OPTIMIZE += --param sms-min-sc=1000                            #The minimum value of stage count that swing modulo scheduler will generate. 
	OPTIMIZE += --param sms-dfa-history=1000                        #The number of cycles the swing modulo scheduler considers when checking conflicts using DFA 
	#OPTIMIZE += --param sms-loop-average-count-threshold=1000      #A threshold on the average loop count considered by the swing modulo scheduler 
	#OPTIMIZE += --param hot-bb-count-ws-permille=100               #A basic block profile count is considered hot if it contributes to the given permillage of the entire profiled execution 
	#OPTIMIZE += --param hot-bb-frequency-fraction=100              #Select fraction of the maximal frequency of executions of basic block in function given basic block needs to have to be considered hot 
	#OPTIMIZE += --param unlikely-bb-count-fraction=10              #The minimum fraction of profile runs a given basic block execution count must be not to be considered unlikely 
	#OPTIMIZE += --param align-threshold=10                         #Select fraction of the maximal frequency of executions of basic block in function given basic block get alignment 
	OPTIMIZE += --param align-loop-iterations=10                    #Loops iterating at least selected number of iterations will get loop alignement. 
	OPTIMIZE += --param max-predicted-iterations=1000               #The maximum number of loop iterations we predict statically 
	#OPTIMIZE += --param builtin-expect-probability=1000            #Set the estimated probability in percentage for builtin expect. The default value is 90% probability. 
	#OPTIMIZE += --param tracer-dynamic-coverage-feedback=1000      #The percentage of function, weighted by execution frequency, that must be covered by trace formation. Used when profile feedback is available 
	OPTIMIZE += --param tracer-dynamic-coverage=100                 #The percentage of function, weighted by execution frequency, that must be covered by trace formation. Used when profile feedback is not available 
	OPTIMIZE += --param tracer-max-code-growth=1000                 #Maximal code growth caused by tail duplication (in percent) 
	#OPTIMIZE += --param tracer-min-branch-ratio=1000               #Stop reverse growth if the reverse probability of best edge is less than this threshold (in percent) 
	#OPTIMIZE += --param tracer-min-branch-probability-feedback=1000 #Stop forward growth if the probability of best edge is less than this threshold (in percent). Used when profile feedback is available 
	#OPTIMIZE += --param tracer-min-branch-probability=1000         #Stop forward growth if the probability of best edge is less than this threshold (in percent). Used when profile feedback is not available 
	OPTIMIZE += --param max-crossjump-edges=100000                  #The maximum number of incoming edges to consider for crossjumping 
	#OPTIMIZE += --param min-crossjump-insns=1000                   #The minimum number of matching instructions to consider for crossjumping 
	#OPTIMIZE += --param max-grow-copy-bb-insns=1000                #The maximum expansion factor when copying basic blocks 
	OPTIMIZE += --param max-goto-duplication-insns=100              #The maximum number of insns to duplicate when unfactoring computed gotos 
	OPTIMIZE += --param max-cse-path-length=100000                  #The maximum length of path considered in cse 
	OPTIMIZE += --param max-cse-insns=100000                        #The maximum instructions CSE process before flushing 
	#OPTIMIZE += --param lim-expensive=1000                         #The minimum cost of an expensive expression in the loop invariant motion 
	OPTIMIZE += --param iv-consider-all-candidates-bound=100000     #Bound on number of candidates below that all candidates are considered in iv optimizations 
	OPTIMIZE += --param iv-max-considered-uses=100000               #Bound on number of iv uses in loop optimized in iv optimizations 
	#OPTIMIZE += --param iv-always-prune-cand-set-bound=1000        #If number of candidates in the set is smaller, we always try to remove unused ivs during its optimization 
	OPTIMIZE += --param scev-max-expr-size=100000                   #Bound on size of expressions used in the scalar evolutions analyzer 
	OPTIMIZE += --param scev-max-expr-complexity=1000               #Bound on the complexity of the expressions in the scalar evolutions analyzer 
	OPTIMIZE += --param omega-max-vars=100000                       #Bound on the number of variables in Omega constraint systems 
	OPTIMIZE += --param omega-max-geqs=100000                       #Bound on the number of inequalities in Omega constraint systems 
	OPTIMIZE += --param omega-max-eqs=100000                        #Bound on the number of equalities in Omega constraint systems 
	OPTIMIZE += --param omega-max-wild-cards=1000                   #Bound on the number of wild cards in Omega constraint systems 
	OPTIMIZE += --param omega-hash-table-size=1000                  #Bound on the size of the hash table in Omega constraint systems 
	OPTIMIZE += --param omega-max-keys=1000                         #Bound on the number of keys in Omega constraint systems 
	OPTIMIZE += --param omega-eliminate-redundant-constraints=1     #When set to 1, use expensive methods to eliminate all redundant constraints 
	OPTIMIZE += --param vect-max-version-for-alignment-checks=100   #Bound on number of runtime checks inserted by the vectorizer's loop versioning for alignment check 
	OPTIMIZE += --param vect-max-version-for-alias-checks=100       #Bound on number of runtime checks inserted by the vectorizer's loop versioning for alias check 
	OPTIMIZE += --param vect-max-peeling-for-alignment=64           #Max number of loop peels to enhancement alignment of data references in a loop 
	OPTIMIZE += --param max-cselib-memory-locations=100000          #The maximum memory locations recorded by cselib 
	OPTIMIZE += --param max-reload-search-insns=100000              #The maximum number of instructions to search backward when looking for equivalent reload 
	#OPTIMIZE += --param sink-frequency-threshold=1000              #Target block's relative execution frequency (as a percentage) required to sink a statement 
	OPTIMIZE += --param max-sched-region-blocks=1000                #The maximum number of blocks in a region to be considered for interblock scheduling 
	OPTIMIZE += --param max-sched-region-insns=100000    		#The maximum number of insns in a region to be considered for interblock scheduling 
	OPTIMIZE += --param max-pipeline-region-blocks=1000  		#The maximum number of blocks in a region to be considered for interblock scheduling 
	OPTIMIZE += --param max-pipeline-region-insns=100000 		#The maximum number of insns in a region to be considered for interblock scheduling 
	# OPTIMIZE += --param min-spec-prob=1000             		#The minimum probability of reaching a source block for interblock speculative scheduling 
	OPTIMIZE += --param max-sched-extend-regions-iters=10		#The maximum number of iterations through CFG to extend regions 
	#OPTIMIZE += --param max-sched-insn-conflict-delay=3 		#The maximum conflict delay for an insn to be considered for speculative motion 
	# OPTIMIZE += --param sched-spec-prob-cutoff=1000      		#The minimal probability of speculation success (in percents), so that speculative insn will be scheduled. 
	# OPTIMIZE += --param sched-state-edge-prob-cutoff=1000 	#The minimum probability an edge must have for the scheduler to save its state across it. 
	OPTIMIZE += --param selsched-max-lookahead=1000      		#The maximum size of the lookahead window of selective scheduling 
	OPTIMIZE += --param selsched-max-sched-times=1000    		#Maximum number of times that an insn could be scheduled 
	OPTIMIZE += --param selsched-insns-to-rename=10    		#Maximum number of instructions in the ready list that are considered eligible for renaming 
	# OPTIMIZE += --param sched-mem-true-dep-cost=1			#Minimal distance between possibly conflicting store and load 
	# OPTIMIZE += --param max-last-value-rtl=1000          		#The maximum number of RTL nodes that can be recorded as combiner's last value 
	OPTIMIZE += --param max-fields-for-field-sensitive=1000 	#Maximum number of fields in a structure before pointer analysis treats the structure as a single variable 
	OPTIMIZE += --param max-sched-ready-insns=100000 		#The maximum number of instructions ready to be issued to be considered by the scheduler during the first scheduling pass 
	OPTIMIZE += --param max-partial-antic-length=0			#Maximum length of partial antic set when performing tree pre optimization 
	OPTIMIZE += --param sccvn-max-scc-size=100000 		        #Maximum size of a SCC before SCCVN stops processing a function 
	OPTIMIZE += --param sccvn-max-alias-queries-per-access=100000   #Maximum number of disambiguations to perform per memory access 
	OPTIMIZE += --param ira-max-loops-num=1000	           	#Max loops number for regional RA 
	OPTIMIZE += --param ira-max-conflict-table-size=6000 		#Max size of conflict table in MB 
	# OPTIMIZE += --param ira-loop-reserved-regs=1000      		#The number of registers in each class kept unused by loop invariant motion 
	OPTIMIZE += --param lra-max-considered-reload-pseudos=1000 	#The max number of reload pseudos which are considered during spilling a non-reload pseudo 
	# OPTIMIZE += --param switch-conversion-max-branch-ratio=1000   #The maximum ratio between array size and switch branches for a switch conversion to take place 
	# OPTIMIZE += --param loop-block-tile-size=1000      		#size of tiles for loop blocking 
	OPTIMIZE += --param graphite-max-nb-scop-params=1000 		#maximum number of parameters in a SCoP 
	OPTIMIZE += --param graphite-max-bbs-per-function=10000		#maximum number of basic blocks per function to be analyzed by Graphite 
	OPTIMIZE += --param loop-max-datarefs-for-datadeps=100000  	#Maximum number of datarefs in loop for building loop data dependencies 
	OPTIMIZE += --param loop-invariant-max-bbs-in-loop=100000 	#Max basic blocks number in loop for loop invariant motion 
	OPTIMIZE += --param slp-max-insns-in-bb=1000         		#Maximum number of instructions in basic block to be considered for SLP vectorization 
	# OPTIMIZE += --param min-insn-to-prefetch-ratio=1000  		#Min. ratio of insns to prefetches to enable prefetching for a loop with an unknown trip count 
	# OPTIMIZE += --param prefetch-min-insn-to-mem-ratio=1000 	#Min. ratio of insns to mem ops to enable prefetching in a loop 
	OPTIMIZE += --param max-vartrack-size=0			        #Max. size of var tracking hash tables 
	# OPTIMIZE += --param max-vartrack-reverse-op-size=1000 	#Max. size of loc list for which reverse ops should be added 
	# OPTIMIZE += --param ipa-sra-ptr-growth-factor=1000   		#Maximum allowed growth of size of new parameters ipa-sra replaces a pointer to an aggregate with 
	OPTIMIZE += --param ipa-cp-value-list-size=100000 	        #Maximum size of a list of values associated with each parameter for interprocedural constant propagation 
	# OPTIMIZE += --param ipa-cp-eval-threshold=1000       		#Threshold ipa-cp opportunity evaluation that is still considered beneficial to clone. 
	OPTIMIZE += --param ipa-max-agg-items=1000           		#Maximum number of aggregate content items for a parameter in jump functions and lattices 
	# OPTIMIZE += --param ipa-cp-loop-hint-bonus=1000      		#Compile-time bonus IPA-CP assigns to candidates which make loop bounds or strides known. 
	# OPTIMIZE += --param ipa-cp-array-index-hint-bonus=1000 	#Compile-time bonus IPA-CP assigns to candidates which make an array index known. 
	OPTIMIZE += --param lto-partitions=64              		#Number of partitions the program should be split to 
	# OPTIMIZE += --param max-stores-to-sink=1000          		#Maximum number of conditional store pairs that can be sunk 
	OPTIMIZE += --param case-values-threshold=0       		#The smallest number of different values for which it is best to use a jump-table instead of a tree of conditional branches, if 0, use the default for the machine 
	OPTIMIZE += --param max-tail-merge-comparisons=1000  		#Maximum amount of similar bbs to compare a bb with 
	OPTIMIZE += --param max-tail-merge-iterations=10		#Maximum amount of iterations of the pass over a function 
	OPTIMIZE += --param max-tracked-strlens=1000         		#Maximum number of strings for which strlen optimization pass will track string lengths 
	OPTIMIZE += --param max-slsr-cand-scan=100000 		        #Maximum length of candidate scans for straight- line strength reduction 
	OPTIMIZE += --param uninit-control-dep-attempts=1000      	#Maximum number of nested calls to search for control dependencies during uninitialized variable analysis
	
	
	#Optimizations that need Graphite
	ifdef GRAPHITE_AVAILABLE
		OPTIMIZE += -floop-interchange		#Perform loop interchange transformations on loops.
		OPTIMIZE += -floop-strip-mine		#Perform loop strip mining transformations on loops.
		OPTIMIZE += -floop-block		#Perform loop blocking transformations on loops.
		OPTIMIZE += -fgraphite-identity
		OPTIMIZE += -floop-nest-optimize
	endif
	
	# Potentially loose precicision
	OPTIMIZE += -freciprocal-math		# Allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations.
	
	# Potentially VERY dangerous
	OPTIMIZE += -fmerge-all-constants	# Attempt to merge identical constants and identical variables. 
	OPTIMIZE += -funsafe-loop-optimizations	# This option tells the loop optimizer to assume that loop indices do not overflow, and that loops with nontrivial exit condition are not infinite. 
	OPTIMIZE += -fno-math-errno 		# Do not set errno after calling math functions that are executed with a single instruction
	OPTIMIZE += -funsafe-math-optimizations # Allow optimizations for floating-point arithmetic that (a) assume that arguments and results are valid and (b) may violate IEEE or ANSI standards.
	OPTIMIZE += -ffinite-math-only		# Allow optimizations for floating-point arithmetic that assume that arguments and results are not NaNs or +-Infs. 
	OPTIMIZE += -fno-signed-zeros		# Allow optimizations for floating-point arithmetic that ignore the signedness of zero.
	OPTIMIZE += -fno-trapping-math		# Compile code assuming that floating-point operations cannot generate user-visible traps. These traps include division by zero, overflow, underflow, inexact result and invalid operation.
endif
