from .splitter import PartitionProb
from .debug import numerical_list_tostr


def PartitionProb_tostr(prob: PartitionProb) -> str:
    # cdef struct PartitionProb:
    #     cppvector[cppvector[double]] tree_prob_sum
    #     cppvector[int] active_by_tree
    #     cppvector[double] prob
    #     int num_active
    out_str = ""
    out_str += "\ttree_prob_sum: [\n"
    for vec in prob.tree_prob_sum:
        out_str += "\t\t" + numerical_list_tostr(vec) + "\n"
    out_str += "\t]\n"
    out_str += "\tactive_by_tree: " + numerical_list_tostr(prob.active_by_tree) + "\n"
    out_str += "\tprob: " + numerical_list_tostr(prob.prob) + "\n"
    out_str += "\tnum_active: " + str(prob.num_active) + "\n"
    return out_str
