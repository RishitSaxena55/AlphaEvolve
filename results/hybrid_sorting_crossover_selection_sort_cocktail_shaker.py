# --- Best Algorithm Found ---
# Performance Score: 0.000762 (Lower is Better) 

def sort_function(arr):
    n = len(arr)
    start = 0
    end = n - 1
    swapped = True

    while swapped:
        swapped = False
        min_idx = start
        max_idx = end
        min_val = arr[start]
        max_val = arr[start]

        # Single pass to find both min and max
        for i in range(start + 1, end + 1):
            if arr[i] < min_val:
                min_val = arr[i]
                min_idx = i
            elif arr[i] > max_val:
                max_val = arr[i]
                max_idx = i

        if min_idx != start:
            arr[start], arr[min_idx] = arr[min_idx], arr[start]
            swapped = True

        #Adjust max_idx if min was swapped to max's position
        if max_idx == start:
            max_idx = min_idx

        if max_idx != end:
            arr[end], arr[max_idx] = arr[max_idx], arr[end]
            swapped = True

        start += 1
        end -= 1

        if start >= end:
            break
    return arr