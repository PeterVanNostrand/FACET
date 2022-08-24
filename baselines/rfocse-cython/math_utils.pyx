cdef void sum_vector(cppvector[double] & labels, cppvector[double] & into) :
    for i in range(labels.size()):
        into[i] = into[i] + labels[i]

cdef void sub_vector(cppvector[double] & labels, cppvector[double] & into) :
    for i in range(labels.size()):
        into[i] = into[i] - labels[i]


cdef bool cmp(double v1, double v2, bool is_lte) :
    if is_lte:
        return v1 <= v2
    else:
        return v1 > v2

cdef int double_argmax(cppvector[double] & v) :
    cdef int max_pos = -1
    cdef double max_value

    for i in range(v.size()):
        if max_pos == -1 or v[i] > max_value:
            max_pos = i
            max_value = v[i]

    return max_pos

def is_number(a):
    try:
        int(repr(a))
    except:
        return False

    return True