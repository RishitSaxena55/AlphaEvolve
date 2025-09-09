# **Project AlphaEvolve: An LLM-driven Agent for Algorithmic Discovery 🧬**

<p align="center">
  <a href="https://ai-evolver-webapp-3svl.vercel.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Launch%20App-brightgreen?style=for-the-badge&logo=vercel" alt="Live Demo">
  </a>
</p>

This project is a hands-on exploration into the world of AI-driven scientific discovery, inspired by landmark papers like Google DeepMind's AlphaEvolve. It features a custom-built evolutionary agent that leverages the power of Large Language Models (LLMs) to discover, optimize, and invent computer algorithms from scratch.

The core question: **Can an AI agent, powered by Google's Gemini, evolve and discover novel, efficient algorithms for classic computer science problems?**

The answer, as demonstrated through a series of controlled experiments, is a resounding yes—with some fascinating limitations.

## **🚀 The Core Engine: How It Works**

The system is built on a classic evolutionary loop, but with a modern twist: the "mutation" and "crossover" events are powered by an LLM, allowing for intelligent, semantic changes to the code rather than random ones. This project is a practical implementation of the official AlphaEvolve architecture.

The core workflow consists of these steps:

1. **The Population:** A collection of the best-performing algorithms found so far.  
2. **Parent Selection:** One or two of the best "parent" algorithms are selected from the population.  
3. **The Prompt Engine:** The parent algorithms are embedded into a carefully crafted prompt, instructing the LLM to either mutate (improve one parent) or perform a crossover (combine the ideas of two parents).  
4. **The LLM "Creator":** The prompt is sent to the Gemini 1.5 Flash API, which generates a new "child" algorithm.  
5. **The Test Harness:** This automated "judge" rigorously tests the new algorithm for correctness and performance.  
6. **Survival of the Fittest:** If the new algorithm is correct and performs well, it is added to the population, and the weakest members are culled.

## **📊 Summary of Results**

This table summarizes the key findings from each experiment, showing the progression from simple problems to more complex challenges.

| Experiment | Strategy | Algorithm Discovered | Final Performance Score\* | View Code | View Plot |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Baseline** | Unconstrained | Built-in sorted() | \~0.000225 | N/A | N/A |
| **Constraint** | Mutation | Selection Sort | \~0.001313 | [Code](https://www.google.com/search?q=./results/selection_sort_suggestion.png) | N/A |
| **Crossover** | Crossover | **Hybrid Cocktail Sort** | **\~0.000762** | [Code](results/hybrid_sorting_crossover_selection_sort_cocktail_shaker.py) | N/A |
| **New Domain** | Crossover | Two-Pointer Swap | \~0.000510 | [Code](results_reversal/crossover_two_pointer_swap_reversal.py) | [Plot](results_reversal/crossover_two_pointer_swap_reversal.png) |
| **Pathfinding** | Adv. Strategies | Stuck in Local Minimum | \~11.0 (Non-optimal path) | [Code](https://www.google.com/search?q=./results_path_finding/best_algorithm.py) | [Plot](https://www.google.com/search?q=./results_path_finding/performance_history.png) |

*\*Lower score is better. Scores for sorting/reversal are time; score for pathfinding is path cost.*

## **🔬 The Research Journey: A Tale of Five Experiments**

The project evolved through a series of carefully designed experiments, each building on the last.

### **Experiment 1 & 2: Sorting (Baseline & Constraint)**

* **Finding:** The AI could find the optimal built-in sorted() function when unconstrained, and discovered a classic **Selection Sort** from scratch when constrained. This proved the core concept.

### **Experiment 3: The Crossover Strategy**

* **Finding:** Using a "crossover" strategy to combine ideas from two parents was more effective, leading to the discovery of sophisticated algorithms like **Cocktail Shaker Sort**.

### **Experiment 4: Generalizing to a New Domain (String Reversal)**

* **Finding:** The engine proved to be general-purpose by successfully adapting to a new problem and discovering the efficient **Two-Pointer Swap** algorithm for string reversal.

### **Experiment 5: The Pathfinding Challenge**

* **Research Question:** Can the engine make the significant creative leap required to discover a complex algorithm like Dijkstra's from a simple starting point?  
* **Result:** The system consistently found a valid path using a simple Depth-First Search (DFS), but failed to evolve it into a solution that correctly used weights to find the shortest path. Advanced strategies like "Chain-of-Thought" and "Hall of Fame" were implemented to encourage creativity.  
* **Finding:** This experiment successfully identified the limits of the current evolutionary strategies. The conceptual gap between a simple DFS and a multi-component algorithm like Dijkstra's was too large for the AI to cross in a single evolutionary step. This is a classic example of getting stuck in a **"local minimum"** and is a critical finding for future work.

## **🛠️ Tech Stack**

* **Language:** Python 3  
* **Core AI:** Google Gemini 2.0 Flash Lite API  
* **Data Analysis & Plotting:** Matplotlib  
* **Environment:** venv, python-dotenv

## **⚙️ Running the Experiments**

1. **Clone the Repository:**  
   ```
   git clone \[https://github.com/RishitSaxena55/AlphaEvolve.git\](https://github.com/RishitSaxena55/AlphaEvolve.git)  
   cd AlphaEvolve
   ```

2. **Set Up the Environment:**  
   ```
   python3 \-m venv venv  
   source venv/bin/activate  \# On Windows: .\\venv\\Scripts\\activate
   ```
   
3. **Install Dependencies:**  
   ```
   pip install \-r requirements.txt  
   ```

## **🌟 What I Learned (Key Takeaways)**

* **LLMs as Creative Engines:** LLMs are incredibly powerful tools for intelligent "mutation" and "crossover" in evolutionary algorithms.  
* **The Power of Constraints:** The most interesting discoveries happened when the AI was given clear constraints, forcing it to be genuinely creative.  
* **Identifying Creative Limits:** The pathfinding experiment was crucial in demonstrating that for complex, multi-component algorithms, simple evolution is not enough. The AI needs a better strategy to make large conceptual leaps.

## **🤝 Future Work & Collaboration Opportunities**

This project is a starting point, and the findings from the pathfinding experiment have paved the way for exciting future research. I am actively looking for opportunities to expand this work.

* **Overcoming Creative Gaps:** The key challenge is to help the AI bridge large conceptual gaps. I am interested in exploring:  
  * **Algorithmic Scaffolding:** Providing the AI with a structural "skeleton" of a complex algorithm (like Dijkstra's) and tasking it with filling in the logical gaps.  
  * **Multi-Stage Evolution:** Evolving individual components (like a priority queue) first, and then evolving a final algorithm that uses these pre-evolved components.  
* **Tackling New Domains:** The framework is ready to be tested against other complex problems like simple data compression or optimizing data structures.  
* **Enhancing the Engine:** There is significant room to improve the core engine, including adapting it to use open, locally-run models like Gemma.

I believe this research direction has immense potential. If you are a researcher, student, or engineer interested in this field, please feel free to open an issue on GitHub to discuss ideas or reach out to me directly.
