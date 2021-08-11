import numpy as np


def __graph_walk(i, adjacency, is_visited, nodes_in_component):
    '''
    performs a depth first walk of the given graph
    '''
    is_visited[i] = True
    nodes_in_component.append(i)

    neighbor_idx = np.argwhere(adjacency[i] == 1).T[0]
    for j in range(len(neighbor_idx)):
        if not is_visited[neighbor_idx[j]]:
            __graph_walk(neighbor_idx[j], adjacency, is_visited, nodes_in_component)


# build a graph to represent the feature similarities between trees
adjacency = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 0]])

# find the largest component (set of connected) nodes in the graph using the adjacency matrix
ntrees = adjacency.shape[0]
is_visited = [False] * ntrees
all_components = []
for i in range(ntrees):
    if not is_visited[i]:
        nodes_in_component = []
        __graph_walk(i, adjacency, is_visited, nodes_in_component)
        all_components.append(nodes_in_component)

print(all_components)

# find largest component
print(max(all_components, key=len))
