# Introduction

This repository implements Laser, a Large Language Model (LLM)-assisted semi-automated tool designed for reproducing experiments in Low Earth Orbit (LEO) Satellite Networks (LSN). It leverages advanced natural language understanding capabilities of LLMs to interpret research papers and generate executable code, significantly reducing the manual effort required for re-implementing experiments from scratch. More details can be found in the LEO-NET 2025 paper: *How LLM Saved Me from Struggling with Experiment Reproduction: LEO Networking as A Case Study.*



## Environment and Dependencies

### 1. Python is needed

Laser is run in python 3 and the following libraries should be installed:

- faiss-cpu>=1.11.0.post1
- torch>=2.7.1
- numpy>=2.1.2
- transformers>=4.53.2
- sentence-transformers>=5.0.0
- safetensors>=0.5.3
- huggingface-hub>=0.33.4


### 2. An LSN Simulator is needed

Prepare an LSN Simulator with APIs and documents (e.g., [StarPerf](https://github.com/SpaceNetLab/StarPerf_Simulator)).



### 3. An LLM is needed

You can use any LLM that supports code generation, such as OpenAI's ChatGPT or other similar models. Make sure you have access to the LLM's API and can integrate it with the Laser tool.


## Usage & Workflow

### Preliminaries for Experiment Reproduction

Step 1: Run `preprocessor.py` to collect, filter and map the online research papers or the papers under `paper_pdf` folder. The output will be stored in `paper_txt` folder.

Step 2: Write your experiment requirement file in `requirement.txt`. This file should contain the specific requirements for the experiment you want to reproduce, such as "Reproduce the experiment presented in this LEO networking paper."

Step 3: Laser stores the embeddings map and experiment requirements in a two-level vector database.
So, run `L1_database.py` for building coarse-grained database and `L2_database.py` for building fine-grained database. The output will be stored in `coarse_index.faiss` and `fine_index.faiss`.


Step 4: Run `search.py` to map the experiment requirement and match relevant information. `search.py` will search in `coarse_index.faiss` and `fine_index.faiss` to find the relevant information. The output will be stored in `results.txt`.


### Experiment Reproduction

Step 1: Based on the APIs and documents of LSN Simulator, Laser generates a standardized manual (`Manual.pdf`). This manual serves as a template for code generation, and Laser performs few-shot learning using the manual as guidance.

Step 2: The requirements and knowledge in `results.txt`, along with the template in `Manual.pdf`, are fed into the LLM.

Step 3: LLM generates experiment code based on the simulator APIs, and then, you need manually pastes the code into the simulator and runs it. 

Step 4: The simulator provides feedback on the results to you. If you find any errors in the code, you need provide fix prompts to Laser. Laser will then continue the execution and adjust the code accordingly to generate the corrected version.


