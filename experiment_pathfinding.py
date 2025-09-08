import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
import random
import matplotlib.pyplot as plt
import time

# Import the judge
from evaluation.harness_pathfinding import evaluate_function

NUM_GENERATIONS = 15

graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start_node = 'A'
end_node = 'D'

# The known shortest path is A -> B -> C -> D with a total cost of 1 + 2 + 1 = 4.

initial_code = """
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
"""

POPULATION_SIZE = 10

# Initialize the population with a score
print("Evaluating initial population...")
population = []
temp_namespace = {}
exec(initial_code, temp_namespace)
initial_function = temp_namespace['find_path_function']

is_correct, performance = evaluate_function(initial_function, graph, start_node, end_node)

if is_correct:
    population.append((performance, initial_code))

print(f"Initialized population with {len(population)} algorithms(s).")


def create_prompt(code1, code2=None):
    """Creates a prompt for the pathfinding problem."""

    # The main task description is defined once to ensure consistency.
    task_description = """You are an expert programmer specializing in graph algorithms. Your task is to write a Python function `find_path_function(graph, start, end)` that finds the shortest path in a weighted graph.

The `graph` is a dictionary where keys are node names and values are dictionaries of neighbors and their weights (costs). The function must return a list of nodes representing the shortest path from `start` to `end`.

Your goal is to evolve the given algorithm(s) to be more efficient and to correctly consider the weights to find the optimal path. A good way to do this is often by keeping track of the shortest known distance to each node and a set of visited nodes.

**CRITICAL CONSTRAINT**: You must write the pathfinding logic from scratch. You are **NOT ALLOWED** to use any imported libraries like `networkx`.

**CRITICAL**: You must provide *only* the complete, new Python function in your response. Do not include any explanations."""

    if code2 is None:
        # Mutation Prompt: This is used when there is only one parent algorithm.
        return f"""{task_description}

Here is the function to improve:
```python
{code1}
```
"""
    else:
        # Crossover Prompt: This is used when combining ideas from two parent algorithms.
        return f"""{task_description}

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
    os.makedirs('results_path_finding', exist_ok=True)

    print("\n Generating performance plot...")

    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.plot(history, marker='o', linestyle='-')

    plt.title('Performance Improvement Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Performance Score (Lower is Better)')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig('results_path_finding/performance_history.png')
    print("Plot saved to results_path_finding/performance_history.png")


def save_best_code(score, code, filename="best_algorithm.py"):
    """Saves the best performing code to a file in the results dir."""
    os.makedirs('results_path_finding', exist_ok=True)

    filepath = os.path.join('results_path_finding', filename)

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
        child_function = temp_namespace['find_path_function']
    except Exception as e:
        print(f"Failed to create a function from the LLM's response. Error: {e}")

    # 5. Evaluate the new function
    if child_function:
        print("Evaluating the new function...")
        is_correct, performance = evaluate_function(child_function, graph, start_node, end_node)
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
    save_best_code(best_overall_score, best_overall_code)

else:
    print("No successful algorithms were found")
