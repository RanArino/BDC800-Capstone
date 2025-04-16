# BDC800-Capstone: RAG System Project

This repository contains the code and resources for the "BDC800 - Capstone Project" at Seneca Polytechnic, focusing on the development, evaluation, and experimentation of Retrieval-Augmented Generation (RAG) systems.

## 👥 Collaborators

- [Ran Arino](https://github.com/RanArino)
- [Nipun Ravindu Fernando Warnakulasuriya](https://github.com/Nipunfdo-git)
- [Umer Aftab](https://github.com/uaftab101-git)

## 🚀 Getting Started: Running Your Own RAG Experiments

This guide will walk you through setting up the project environment and running the RAG experiments included in this repository.

### 1. Clone Repository

Clone the repository to your local machine using your preferred tool (VS Code, Cursor, or command line):

```bash
git clone https://github.com/RanArino/BDC800-Capstone.git
cd BDC800-Capstone
```

### 2. Set Up Virtual Environment & Install Dependencies

#### Windows
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

To deactivate the virtual environment when done: `deactivate`

### 3. Set Up External Dependencies

The RAG experiments rely on external services and models.

#### a) Ollama (for Local LLMs)

Ollama is used to run open-source language models locally for generation and embedding.

1.  **Install Ollama**: Download and install Ollama from the official website: [https://ollama.com/](https://ollama.com/)
2.  **Download Models**: Open your terminal and pull the required models. The specific models used depend on the experiment configuration (`core/configs/**/*.yaml`), but common models used in this project might include:
    ```bash
    ollama pull llama3.1
    ollama run llama3.2:1b
    ollama run deepseek-r1:8b
    ollama run phi4:14b
    ```
    *Note: Check the specific `llm_model_name` and `embedding_model_name` parameters in the YAML configuration files you intend to run to confirm which models to download.*

#### b) Google AI Studio API Key (for Gemini Models)

Some configurations might use Google's Gemini models (via the `google-generativeai` library) for generation or evaluation.

1.  **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/) and create an API key.
2.  **Set Environment Variable**: Make the API key available to the application by setting an environment variable named `GOOGLE_API_KEY`.
    *   **macOS/Linux**:
        ```bash
        export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
        ```
        (Add this line to your `.bashrc`, `.zshrc`, or activate script for persistence)
    *   **Windows (Command Prompt)**:
        ```cmd
        set GOOGLE_API_KEY=YOUR_API_KEY_HERE
        ```
    *   **Windows (PowerShell)**:
        ```powershell
        $env:GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```
    *Alternatively, you can use a `.env` file and the `python-dotenv` library if preferred, but ensure the code in `core/llm/controller.py` or similar loads it.*

### 4. Configure and Run Experiments

#### a) Understanding Configuration Files

Experiments are defined by YAML configuration files located in `core/configs/`.

-   **Structure**: Configs are organized by framework type (e.g., `core/configs/simple_rag/`, `core/configs/scaler_rag/`).
-   **Files**: Each file (e.g., `rag01_sentence_chunk.yaml`) groups related experiment variations.
-   **Sections**: Inside each file, distinct configurations (e.g., `simple_rag_01_01`, `simple_rag_01_02`) define specific parameters like:
    -   `dataset_name`: Which dataset to use (from `core/datasets`).
    -   `llm_model_name`: The generator LLM (e.g., `llama3.1`, `phi4`).
    -   `embedding_model_name`: The embedding model (`huggingface-multi-qa-mpnet`).
    -   `retriever_k`: Number of documents to retrieve.
    -   Chunking parameters (`chunk_size`, `chunk_overlap`).
    -   Framework-specific settings are set on `core/frameworks/schema.py`; check this file to see which value can be set.
-   **Modification**: You can modify existing configurations or create new ones to test different hypotheses.

#### b) Running the Experiments

Experiments are launched via the `experiments/main.py` script.

1.  **Select Experiments**: Open `experiments/main.py`. In the `if __name__ == "__main__":` block at the bottom, uncomment the function calls corresponding to the experiment suites you want to run (e.g., `run_simple_rag_sentences()`).
2.  **Execute**: Run the script from the project's root directory:
    ```bash
    python experiments/main.py
    ```
    The script will iterate through the selected configurations, run the RAG pipeline, calculate metrics, and save the results.

#### c) Experiment Outputs

Results for each configuration run are saved in the `experiments/` directory, timestamped to prevent overwriting:

-   **`experiments/responses/`**: Contains detailed JSON files (`<config_name>-<timestamp>.json`) with raw inputs, outputs, and retrieved contexts for each query.
-   **`experiments/metrics/`**: Contains JSON summary files (`<config_name>-<timestamp>.json`) with aggregated evaluation metrics (retrieval, generation, relevance).
-   **`experiments/detailed_dfs/`**: Contains CSV files (`<config_name>-<timestamp>.csv`) with per-query metrics for fine-grained analysis.

## 📂 Project Structure Overview

Check more detailed structure on `docs/folder_struct.md`

```plaintext
.
├── core/                  # Core RAG components and implementations
│   ├── configs/           # YAML configuration files for experiments
│   ├── datasets/          # Dataset loading and processing modules
│   ├── evaluation/        # Evaluation metrics, summarization, and visualization
│   ├── frameworks/        # RAG framework implementations (Simple, Scaler)
│   ├── logger/            # Logging setup
│   ├── rag_core/          # Foundational components (chunking, LLM interaction)
│   └── utils/             # Utility functions (e.g., profiler)
├── experiments/           # Experiment execution scripts and results
│   ├── main.py            # Main script to run experiment batches
│   ├── base.py            # Core logic for running a single experiment
│   ├── metrics/           # Stored metrics summary JSON files
│   ├── responses/         # Stored raw response JSON files
│   └── detailed_dfs/      # Stored detailed metrics CSV files
├── notebooks/             # Jupyter notebooks for analysis, testing, visualization
├── .venv/                 # Virtual environment directory (if created here)
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## 🌿 Git Workflow for Team Members

*(This section details the branching strategy, commit process, and pull request guidelines for project contributors. Retained from the original README for collaborators.)*

### Initial Setup (One Time Only)
1. After cloning, make sure you're on the main branch: `git checkout main && git pull origin main`
2. Configure your Git identity:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

### Starting Your Work
1. Create Your Feature Branch:
   ```bash
   git checkout main && git pull origin main
   git checkout -b feature/your-feature-name
   ```
   *(Use `fix/` or `docs/` prefixes as appropriate)*

### Working on Your Branch
1. Make changes.
2. Commit regularly: `git add . && git commit -m "Brief description"`

### Keeping Your Branch Updated
1. Sync with main: `git stash && git checkout main && git pull origin main && git checkout feature/your-feature-name && git merge main && git stash pop`

### Pushing Your Changes
1. Push your branch: `git push origin feature/your-feature-name`

### Creating a Pull Request (PR)
1. Go to GitHub, create a PR from your branch to `main`.
2. Fill in the template and request reviews.

### 🚫 Common Mistakes to Avoid
1. Never commit directly to `main`.
2. Don't merge your own PR without review.
3. Keep commits focused. Write clear messages.
4. Always pull before starting new work.

---

*Remember to replace placeholders like `YOUR_API_KEY_HERE` with your actual key and consult the specific configuration files for exact model names.*

