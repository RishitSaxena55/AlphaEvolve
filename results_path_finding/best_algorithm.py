# --- Best Algorithm Found ---
# Performance Score: 3.000000 (Lower is Better) 


import collections


def find_path_function(graph, start, end):
    if start == end:
        return [start]

    queue = collections.deque([(start, [start])])
    visited = {start}

    while queue:
        current_node, current_path = queue.popleft()

        if current_node not in graph:
            continue

        for neighbor in graph[current_node]:
            if neighbor == end:
                return current_path + [neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_path + [neighbor]))

    return None
