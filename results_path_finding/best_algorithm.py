# --- Best Algorithm Found ---
# Performance Score: 3.000000 (Lower is Better) 


def find_path_function(graph, start, end, path=None):
    # A corrected recursive DFS pathfinder.
    # It finds a path, but ignores weights, so it is not optimal.
    if path is None:
        path = []

    path = path + [start]

    if start == end:
        return path

    if start not in graph:
        return None

    # This just finds the first valid path, not the shortest.
    for node in graph[start]:
        if node not in path:
            newpath = find_path_function(graph, node, end, path)
            if newpath:
                return newpath

    return None
