# Repository Structure

Below is an **alternative format** for your repository structure that’s more readable in various interfaces:

---



Tailoring Math Examples: 

Test Personalization: Each user “grade level” could equate to a different set of user preferences and complexity requirements.
Measure Adaptability: See how effectively the model’s chain-of-thought and final answers adjust to users of varying knowledge.
Demonstrate the Memory Advantage: For instance, a “Grade 3” user might require simpler language, more step-by-step breakdown, while a “Grade 10” user might already know the basics and need more advanced context.
Showcase Evolving Preferences: Over multiple sessions, the user might “graduate” from simpler tasks to more advanced tasks, and you can see if your memory-augmented system shifts its style accordingly.


## my-math-mCoT-IPO
- **README.md**  
  - High-level overview of the entire project
- **requirements.txt**  
  - Contains a list of dependencies for the project

---

## data
- **user_preferences/**
  - `example_user.json`  
    - JSON file for memory-based preferences
- **tasks/**
  - `math_tasks.json`  
    - JSON file storing sample math tasks

---

## docs
- `toy_math_environment.md`  
  - Guides and documentation related to the toy math environment

---

## src
- **memory/**
  - `memory_manager.py`  
    - Functions to load/save user memory in JSON
  - `retrieval.py`  
    - Logic for retrieving relevant memory chunks
- **prompting/**
  - `prompt_builder.py`  
    - Helper scripts for constructing prompts
  - `format_instructions.py`  
    - Common instruction templates for formatting
- **models/**
  - `mcot_inference.py`  
    - mCoT demonstration code
  - `ipo_inference.py`  
    - IPO code: GRPO + memory integration
- **evaluation/**
  - `evaluator.py`  
    - Scripts to check correctness & formatting of outputs

---

## scripts
- `run_mcot.py`  
  - Script to run memory-based Chain-of-Thought
- `run_ipo.py`  
  - Script to run Inference Preference Optimization with memory


# Helpful Math Resources

Below are some references for math problems and step-by-step solution styles:

- **Paul’s Online Math Notes**  
  - [tutorial.math.lamar.edu](https://tutorial.math.lamar.edu/)  
  - A great source for integrals, derivatives, and step-by-step examples.

- **Khan Academy**  
  - [khanacademy.org](https://www.khanacademy.org/)  
  - Offers basic to intermediate math problem sets, useful for generating toy tasks.

- **MIT OpenCourseWare - Single Variable Calculus**  
  - [ocw.mit.edu](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/)  
  - For more advanced integrals, series expansions, or limit problems.

- **Wolfram MathWorld**  
  - [mathworld.wolfram.com](https://mathworld.wolfram.com/)  
  - An extensive resource of definitions, theorems, and sample problems.

By referencing these resources, you can craft diverse math prompts and confirm your model’s step-by-step solutions align with standard approaches.