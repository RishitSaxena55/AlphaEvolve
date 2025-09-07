import time
import random
import string


def check_correctness(reverse_function_to_test):
    """Tests a string reversal function."""
    test_cases = [
        ("hello", "olleh"),
        ("racecar", "racecar"),
        ("Python", "nohtyP"),
        ("", ""),
    ]

    for input_str, expected_output in test_cases:
        # Pass through reverse function
        if reverse_function_to_test(input_str) != expected_output:
            return False

    return True


def measure_performance(reverse_function_to_test):
    """Measures the execution speed of a string reversal function."""
    # Generate 1000 random strings of length 10
    test_data = [''.join(random.choices(string.ascii_letters + string.digits, k=10)) for _ in range(1000)]

    start_time = time.perf_counter()

    for test_list in test_data:
        reverse_function_to_test(test_list)

    end_time = time.perf_counter()

    return end_time - start_time


def evaluate_function(candidate_function):
    """
    Evaluates a candidate function for both correctness and performance.
    :param candidate_function:
    :return: tuple: (is_correct, performance_score)
    """
    if not check_correctness(candidate_function):
        return False, float('inf')  # Incorrect, worst possible score

    # If correct, measure it's speed
    performance = measure_performance(candidate_function)
    return True, performance
