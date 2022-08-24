# distutils: language=c++
from .datasets import DatasetInfo
from .extraction_problem cimport FeatureConditions, ExtractionProblem, ExtractionContext, FeatureBounds, SplitPoint, Solution, ConditionSide

cpdef str LOG_LEVEL
cpdef int instance_num

cdef str FeatureConditions_tostr(FeatureConditions & myconditions)
cdef str ExtractionProblem_tostr(ExtractionProblem & problem)
cdef str ExtractionContext_tostr(ExtractionContext & global_state)
cdef str FeatureBounds_tostr(FeatureBounds &bounds)
cdef str SplitPoint_tostr(SplitPoint &split)
cdef DatasetInfo_tostr(dataset_info: DatasetInfo)
cdef str Solution_tostr(Solution &solution)
cdef str ConditionSide_tostr(ConditionSide & condition_side)
