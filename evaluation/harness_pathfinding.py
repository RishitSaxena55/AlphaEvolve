def check_correctness(path, graph, start_node, end_node):
    """
    Checks if a given path is valid within the graph.
    A valid path:
    1. Starts at the start_node.
    2. Ends at the end_node.
    3. Consists of a continuous sequence of connected nodes.
    """
    # 1. Check if the part starts and end correctly
    if not path or path[0] != start_node or path[-1] != end_node:
        return False

    # 2. Check if each step in the path is a valid connection
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]

        # Check if current node is in the graph
        if current_node not in graph:
            return False

        # Check if next node is valid neighbor of the current one
        if next_node not in graph[current_node]:
            return False

    return True


def measure_performance(path, graph):
    """Calculates the total cost (weight) of a given path."""
    total_cost = 0
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]

        # This assumes the path is already valid
        total_cost += graph[current_node][next_node]

    return total_cost


def evaluate_function(candidate_function, graph, start_node, end_node):
    """
    Evaluates a candidate pathfinding function.
    - Runs the function to get the path.
    - Checks if the path is correct.
    - If correct, measures its performance (total_cost).
    """
    # Run the candidate function to get its proposed path
    try:
        path = candidate_function(graph.copy(), start_node, end_node)
    except Exception:
        # If the function crashes, it's a failure
        return False, float('inf')

    # Check if the generated path is valid
    if not check_correctness(path, graph, start_node, end_node):
        return False, float('inf')

    # If the path is valid, calculate it's performance
    performance = measure_performance(path, graph)
    return True, performance
