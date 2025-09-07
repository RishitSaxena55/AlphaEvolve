import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
import random
import matplotlib.pyplot as plt
import time

# Import the judge
from evaluation.harness_string import evaluate_function

NUM_GENERATIONS = 15
POPULATION_SIZE = 10

# Define the two starting algorithms
initial_code_1 = """
def reverse_function(s):
    reversed_s = ""
    for char in s:
        reversed_s = char + reversed_s
    return reversed_s
"""

initial_code_2 = """
def reverse_function(s):
    char_list = list(s)
    start, end = 0, len(char_list) - 1
    while start < end:
        char_list[start], char_list[end] = char_list[end], char_list[start]
        start += 1
        end -= 1
    return "".join(char_list)
"""

# --- Evaluation ---
population = []  # Start with an empty list

# Evaluate the first algorithm and add it to the population
print("Evaluating initial_code_1...")
temp_namespace_1 = {}
exec(initial_code_1, temp_namespace_1)
initial_function_1 = temp_namespace_1['reverse_function']
is_correct_1, performance_1 = evaluate_function(initial_function_1)
if is_correct_1:
    population.append((performance_1, initial_code_1))

# Evaluate the second algorithm and add it to the population
print("Evaluating initial_code_2...")
temp_namespace_2 = {}
exec(initial_code_2, temp_namespace_2)
initial_function_2 = temp_namespace_2['reverse_function']
is_correct_2, performance_2 = evaluate_function(initial_function_2)
if is_correct_2:
    population.append((performance_2, initial_code_2))

print(f"\nInitialized population with {len(population)} diverse algorithms.")


def create_prompt(code1, code2=None):
    """Creates a prompt for mutation (1 code) or crossover (2 codes) for STRING REVERSAL."""
    if code2 is None:
        # Mutation Prompt for string reversal
        return f"""You are an expert Python programmer. Your task is to take a given Python function for reversing a string and propose a functionally equivalent but different version.

            Your goal is to improve the function's efficiency or to use a different algorithmic approach.

            **CRITICAL CONSTRAINT**: You must write the string reversal logic from scratch using basic control flow (loops, etc.) and string/character manipulation. You are **NOT ALLOWED** to use high-level slicing shortcuts like `[::-1]`.

            **CRITICAL**: You must provide *only* the complete, new Python function in your response, named `reverse_function`. Do not include any explanations.

            Here is the function to improve:
            ```python
            {code1}
            ```
            """

    # Crossover Prompt for string reversal
    return f"""You are an expert Python programmer specializing in algorithm optimization. Your task is to analyze two different Python functions that reverse a string and create a new, hybrid function that combines the best ideas from both.

        Your goal is to create a new function that is more efficient or uses a more clever algorithmic approach than either of the two examples.

        **CRITICAL CONSTRAINT**: You must write the string reversal logic from scratch using basic control flow (loops, etc.) and string/character manipulation. You are **NOT ALLOWED** to use high-level slicing shortcuts like `[::-1]`.

        **CRITICAL**: You must provide *only* the complete, new Python function in your response, named `reverse_function`. Do not include any explanations.

        Here is the first function (Function A):
        ```python
        {code1}
        ```

        Here is the second function (Function B):
        ```python
        {code2}
        ```
        """


def get_llm_suggestion(prompt):
    """Sends a prompt to the Gemini API and returns the code suggestion."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    response = model.generate_content(prompt)

    # Clean up the response to get only the code
    cleaned_code = response.text.strip().replace("```python", "").replace("```", "").strip()

    return cleaned_code


def plot_history(history):
    """
    Takes a list of performance scores and plots them.
    """
    # Create the results directory if it doesn't exist
    os.makedirs('results_reversal', exist_ok=True)

    print("\n Generating performance plot...")

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(history, marker='o', linestyle='-')

    plt.title('Performance Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Performance Score (Lower is Better)')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('results_reversal/performance_history.png')
    print("Plot saved to results_reversal/performance_history.png")


def save_best_code(score, code, filename="best_algorithm.py"):
    """Saves the best performing code to a file in the results dir."""
    os.makedirs('results_reversal', exist_ok=True)

    filepath = os.path.join('results_reversal', filename)

    print(f"\n Saving best algorithm to {filepath}...")

    # Create a header with the performance score
    header = f"# --- Best Algorithm Found ---\n"
    header += f"# Performance Score: {score:.6f} (Lower is Better) \n\n"

    with open(filepath, 'w') as f:
        f.write(header)
        f.write(code)

    print("Code saved successfully.")


history = []

# --- Start of the main loop ---
for i in range(1, NUM_GENERATIONS + 1):
    print(f"\n --- Generation {i}/{NUM_GENERATIONS} ---")

    if len(population) > 1:
        # Get two unique parents at once
        parent1, parent2 = random.sample(population, 2)
        parent1_score, parent1_code = parent1
        parent2_score, parent2_code = parent2
        print(f"Selected parent 1 (score: {parent1_score:.6f}) and parent 2 (score: {parent2_score:.6f}")
    else:
        # Fallback for first few generations if population is small
        parent1_score, parent1_code = population[0]
        parent2_code = None  # No second parent available yet
        print(f"Selected single parent with score: {parent1_score:.6f}")

    # 2. Create the prompt
    prompt = create_prompt(parent1_code, parent2_code)

    # 3. Get the new "child" code form the LLM
    print("Asking the LLM for a new function...")
    child_code_string = get_llm_suggestion(prompt)
    print("Got a response!")
    print("\n--- LLM Suggestion ---")
    print(child_code_string)
    print("----------------------\n")

    # Rate limit
    print("Pausing for 5 seconds...")
    time.sleep(5)

    # 4. Create a runnable function from the child code string
    child_function = None
    try:
        # We create a temporary dictionary to hold the function
        # and then extract it
        temp_namespace = {}
        exec(child_code_string, temp_namespace)
        child_function = temp_namespace['reverse_function']
    except Exception as e:
        print(f"Failed to create a function from the LLM's response. Error: {e}")

    # 5. Evaluate the new function
    if child_function:
        print("Evaluating the new function...")
        is_correct, performance = evaluate_function(child_function)
        print(f"--- Result ---")
        print(f"Correctness: {is_correct}")
        print(f"Performance (total_time): {performance:.6f} seconds")

        # Add the child to the population if it's good
        if is_correct:
            # Add the new child to the population
            population.append((performance, child_code_string))

            # Sort the population by score (lower is better)
            population.sort(key=lambda x: x[0])

            # Trim the population to maintain its size
            population = population[:POPULATION_SIZE]

    # Log the best score and print the current best score in the population
    if population:
        best_score = population[0][0]
        history.append(best_score)
        print(f"Current Best Score: {population[0][0]:.6f}")

print("\n\n--- Experiment Finished ---")
if population:
    best_overall_score, best_overall_code = population[0]
    print(f"Best overall score: {best_overall_score:.6f}")
    print("--- Best algorithm found ---")
    print(best_overall_code)

    # Plot history
    if history:
        plot_history(history)

    # Save best code
    save_best_code(best_overall_score, best_overall_code, "best_reversal_algorithm.py")

else:
    print("No successful algorithms were found")
