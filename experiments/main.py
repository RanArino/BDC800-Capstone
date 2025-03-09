# experiments/main.py

from experiments.base import run_experiment

def run_simple_rag_sentences():
    config_path = "core/configs/simple_rag/rag01_sentence_chunk.yaml"
    config_names = ["simple_rag_01_01", "simple_rag_01_02", "simple_rag_01_03", "simple_rag_01_04"]
    # run the simple_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

def run_simple_rag_reasoning():
    config_path = "core/configs/simple_rag/rag02_reasoning_model.yaml"
    # "simple_rag_02_01" is the same setting as "simple_rag_01_01"
    config_names = ["simple_rag_02_02", "simple_rag_02_03", "simple_rag_02_04"]
    # run the simple_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

def run_scaler_rag_sentences():
    config_path = "core/configs/scaler_rag/scaler01_sentence_chunk.yaml"
    config_names = ["scaler_rag_01_01", "scaler_rag_01_02", "scaler_rag_01_03", "scaler_rag_01_04"]
    # run the scaler_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

def run_scaler_rag_reasoning():
    config_path = "core/configs/scaler_rag/scaler02_reasoning_model.yaml"
    # "scaler_rag_02_03" is the same setting as "scaler_rag_01_02"
    # "scaler_rag_02_04" is the same setting as "scaler_rag_01_04"
    config_names = ["scaler_rag_02_01", "scaler_rag_02_02"]
    # run the scaler_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

def run_scaler_rag_dim_reduction():
    config_path = "core/configs/scaler_rag/scaler03_dim_reduction.yaml"
    # "scaler_rag_03_03" is the same setting as "scaler_rag_01_02"
    # "scaler_rag_03_04" is the same setting as "scaler_rag_01_04"
    config_names = ["scaler_rag_03_01", "scaler_rag_03_02"]
    # run the scaler_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

def run_scaler_v1_rag_comparison():
    config_path = "core/configs/scaler_v1_rag/comparison.yaml"
    config_names = ["scaler_v1_rag_01_01", "scaler_v1_rag_01_02", "scaler_v1_rag_01_03"]
    # run the scaler_v1_rag experiment
    for config in config_names:
        print(f"\n===== Starting {config} test =====")
        run_experiment(config_path, config)
        print(f"\n===== {config} test completed =====")

if __name__ == "__main__":
    # run experiments
    run_simple_rag_sentences()
    # run_simple_rag_reasoning()
    # run_scaler_rag_sentences()
    # run_scaler_rag_reasoning()
    # run_scaler_rag_dim_reduction()
    # run_scaler_v1_rag_comparison()
