cdef str PartitionProb_tostr(PartitionProb & prob):
    # cdef struct PartitionProb:
    #     cppvector[cppvector[double]] tree_prob_sum
    #     cppvector[int] active_by_tree
    #     cppvector[double] prob
    #     int num_active
    out_str = ""
    out_str += "\ttree_prob_sum: [\n"
    for vec in prob.tree_prob_sum:
        out_str += "\t\t" + str(vec) + "\n"
    out_str += "\t]\n"
    out_str += "\tactive_by_tree: " + str(prob.active_by_tree) + "\n"
    out_str += "\tprob: " + str(prob.prob) + "\n"
    out_str += "\tnum_active: " + str(prob.num_active) + "\n"
    return out_str