# distutils: language=c++
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, profile=True

from libcpp.vector cimport vector as cppvector
from libc.stdint cimport uint32_t
from libcpp cimport bool

cdef void sum_vector(cppvector[double] & labels, cppvector[double] & into) 
cdef void sub_vector(cppvector[double] & labels, cppvector[double] & into) 
cdef bool cmp(double v1, double v2, bool is_lte) 
cdef int double_argmax(cppvector[double] & v)
