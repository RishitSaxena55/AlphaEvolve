import os
import google.generativeai as genai
from dotenv import load_dotenv

bubble_sort_code = """
def sort_function(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

population = [bubble_sort_code]


def create_prompt(code_string):
    """Takes a string of Python code and wraps it in a prompt for the LLM."""
    prompt = f"""You are an expert Python programmer. Your task is to take a given Python function and propose a functionally equivalent but different version.

    Your goal is to improve the function's efficiency or to use a different algorithmic approach.

    **CRITICAL**: You must provide *only* the complete, new Python function in your response. Do not include any explanations, introductory text, or markdown formatting like ```python.

    Here is the function to improve:
    ```python
    {code_string}
    """
    return prompt


def get_llm_suggestion(prompt):
    """Sends a prompt to the Gemini API and returns the code suggestion."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Please check your .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    response = model.generate_content(prompt)

    # Clean up the response to get only the code
    cleaned_code = response.text.strip().replace("```python", "").replace("```", "").strip()

    return cleaned_code


# Import the judge
from evaluation.harness import evaluate_function

# 1. Select the parent
parent_code = population[0]

# 2. Create the prompt
prompt = create_prompt(parent_code)

# 3. Get the new "child" code form the LLM
print("Asking the LLM for a new function...")
child_code_string = get_llm_suggestion(prompt)
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
    child_function = temp_namespace['sort_function']
except Exception as e:
    print(f"Failed to create a function from the LLM's response. Error: {e}")

# 5. Evaluate the new function
if child_function:
    print("Evaluating the new function...")
    is_correct, performance = evaluate_function(child_function)
    print(f"--- Result ---")
    print(f"Correctness: {is_correct}")
    print(f"Performance (total_time): {performance:.6f} seconds")


