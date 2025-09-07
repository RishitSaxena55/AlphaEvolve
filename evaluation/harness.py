import time
import random


def check_correctness(sort_function_to_test):
    """Tests a sorting function against a list of test cases."""
    test_cases = [
        ([3, 1, 2, 5, 4], [1, 2, 3, 4, 5]),  # Standard case
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),  # Reversed case
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),  # Already sorted
        ([1, 5, 2, 5, 1], [1, 1, 2, 5, 5])  # Case with duplicates
    ]

    for input_list, expected_output in test_cases:
        # Pass through sorting function
        if sort_function_to_test(input_list[:]) != expected_output:
            return False

    return True


def measure_performance(sort_function_to_test):
    """Measures the execution speed of a sorting function."""
    # Generate 1000 random lists of 5 numbers
    performance_tests = [random.sample(range(100), 5) for _ in range(1000)]

    start_time = time.perf_counter()

    for test_list in performance_tests:
        sort_function_to_test(test_list[:])

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
