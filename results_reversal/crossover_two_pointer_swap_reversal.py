# --- Best Algorithm Found ---
# Performance Score: 0.000510 (Lower is Better) 

def reverse_function(s):
    n = len(s)
    char_list = list(s)
    left = 0
    right = n - 1
    while left < right:
        char_list[left], char_list[right] = char_list[right], char_list[left]
        left += 1
        right -= 1
    return "".join(char_list)