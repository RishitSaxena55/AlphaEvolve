import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
import random
import matplotlib.pyplot as plt
import time

# Import the judge
from evaluation.harness_pathfinding import evaluate_function

NUM_GENERATIONS = 10
HALL_OF_FAME_INJECTION_RATE = 0.2

graph = {
    'A': {'B': 1, 'C': 10},
    'B': {'A': 1, 'C': 1, 'D': 100},
    'C': {'A': 10, 'B': 1, 'D': 1},
    'D': {'B': 100, 'C': 1}
}
# Correct shortest path is A->B->C->D (cost 3)

start_node = 'A'
end_node = 'D'

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
# Initialize Hall of Fame
hall_of_fame = []

# Initialize the population with a score
print("Evaluating initial population...")
population = []
temp_namespace = {}
exec(initial_code, temp_namespace)
initial_function = temp_namespace['find_path_function']

is_correct, performance = evaluate_function(initial_function, graph, start_node, end_node)

if is_correct:
    population.append((performance, initial_code))
    hall_of_fame.append((performance, initial_code))

print(f"Initialized population with {len(population)} algorithms(s).")


# --- Chain-of-Thought Prompting Functions ---

def create_planning_prompt(code1, code2=None):
    """Creates a prompt that asks the AI to generate a PLAN to improve an algorithm."""

    task_description = """You are an expert programmer specializing in graph algorithms. Your task is to provide a step-by-step plan to create or improve a function that finds the *shortest* path in a weighted graph.

Do NOT write any Python code. Only provide a numbered list of the logical steps required. For example:
1. Initialize a data structure to store distances to all nodes, setting them to infinity.
2. Set the starting node's distance to 0.
3. Create a priority queue to track nodes to visit.
4. Loop until the priority queue is empty..."""

    if code2 is None:
        # Planning prompt for MUTATION (improving one function)
        return f"""{task_description}

Analyze the following function and create a plan to improve it so it correctly considers edge weights.
```python
{code1}
```
"""
    else:
        # Planning prompt for CROSSOVER (combining two functions)
        return f"""{task_description}

Analyze the two following functions. Create a single, hybrid plan that combines the best ideas from both to create a superior algorithm.
Function A:
```python
{code1}
```
Function B:
```python
{code2}
```
"""


def create_coding_prompt(plan, code1, code2=None):
    """Creates a prompt that asks the AI to write CODE based on a plan."""

    if code2 is None:
        context = f"""**The Original Function for reference:**
```python
{code1}
```"""
    else:
        context = f"""**Function A for reference:**
```python
{code1}
```
**Function B for reference:**
```python
{code2}
```"""

    return f"""You are an expert Python programmer. Your task is to write a complete Python function that implements the following plan.

**The Plan:**
{plan}

{context}

**CRITICAL CONSTRAINT**: You must write the pathfinding logic from scratch. You are **NOT ALLOWED** to use any imported libraries like `networkx`.
**CRITICAL**: You must provide *only* the complete, new Python function in your response, named `find_path_function`. Do not include any explanations.
"""


def get_llm_suggestion(parent1_code, parent2_code=None):
    """Orchestrates the two-step "Plan and Code" process."""

    # Get the plan from LLM
    print("Asking the LLM for a plan...")
    planning_prompt = create_planning_prompt(parent1_code, parent2_code)

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')

    # Make the first API call to get the plan
    plan_response = model.generate_content(planning_prompt)
    plan_text = plan_response.text
    print("Got a plan!")
    print("\n--- The Plan ---")
    print(plan_text)
    print("----------------\n")

    time.sleep(5)

    # Get the code based on the Plan
    print("Asking the LLM to write the code for the plan...")
    coding_prompt = create_coding_prompt(plan_text, parent1_code, parent2_code)

    # Make the second API call to get the final code
    code_response = model.generate_content(coding_prompt)

    # Clean up the response to get only the code
    cleaned_code = code_response.text.strip().replace("```python", "").replace("```", "").strip()

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

    # --- NEW: Hall of Fame Parent Selection Logic ---
    parent1_code, parent2_code = None, None
    if not population:
        print("Population is empty. Halting experiment.")
        break

    # Decide whether to inject a parent from the Hall of Fame
    if random.random() < HALL_OF_FAME_INJECTION_RATE and hall_of_fame:
        print("Injecting a creative parent from the Hall of Fame...")
        # Parent 1 is from the Hall of Fame
        parent1_score, parent1_code = random.choice(hall_of_fame)

        # Parent 2 is from the high-performing population (if available)
        if len(population) > 1:
            parent2_score, parent2_code = random.choice(population)
            print(f"Selected parent 1 from HOF (cost: {parent1_score}) and parent 2 (cost: {parent2_score}).")
        else:
            print(f"Selected single parent (cost: {parent1_score}) from HOF for mutation.")
    else:
        # Standard selection from the high-performing population
        if len(population) > 1:
            parent1, parent2 = random.sample(population, 2)
            parent1_score, parent1_code = parent1
            parent2_score, parent2_code = parent2
            print(f"Selected parent 1 (cost: {parent1_score}) and parent 2 (cost: {parent2_score}) for crossover.")
        else:
            parent1_score, parent1_code = population[0]
            print(f"Selected single parent (cost: {parent1_score}) for mutation.")
    child_code_string = get_llm_suggestion(parent1_code, parent2_code)
    print("Got a response!")
    print("\n--- LLM Suggestion ---")
    print(child_code_string)
    print("----------------------\n")

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

        # ... (inside the 'if child_function:' block) ...
        if is_correct:
            # This part is the same: update the main population
            population.append((performance, child_code_string))
            population.sort(key=lambda x: x[0])
            population = population[:POPULATION_SIZE]

            # --- NEW: Hall of Fame Update Logic ---
            is_unique = True
            # Use line count as a simple way to check for structural difference
            child_line_count = len(child_code_string.splitlines())
            for _, fame_code in hall_of_fame:
                if len(fame_code.splitlines()) == child_line_count:
                    is_unique = False
                    break
            if is_unique:
                hall_of_fame.append((performance, child_code_string))
                print("A new unique algorithm was added to the Hall of Fame!")

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

# Print Hall of Fame
if hall_of_fame:
    print("\n--- 🏛️ Hall of Fame (All Unique Algorithms Discovered) ---")
    hall_of_fame.sort(key=lambda x: x[0]) # Sort by best score
    for score, code in hall_of_fame:
        print(f"\n--- Score: {score} ---")
        print(code)

