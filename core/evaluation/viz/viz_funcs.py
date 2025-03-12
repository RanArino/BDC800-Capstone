# core/evaluation/viz/viz_funcs.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Tuple

def create_generation_metrics_df(data_path: str, model_name: str):
    """
    Create a DataFrame from the metrics JSON file.
    
    Args:
        data_path (str): Path to the JSON file containing metrics data
        model_name (str): Name of the model for the data
    """
    # Load JSON data
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    
    metrics_data = []
    
    # Process ROUGE metrics
    for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
        rouge_data = data_dict['generation'][rouge_type]
        for metric_type in ['precision', 'recall', 'fmeasure']:
            metrics_data.append({
                'model': model_name,
                'metric_group': rouge_type.lower(),
                'metric_type': metric_type,
                'value': rouge_data[metric_type]['mean'],
                'q1': rouge_data[metric_type]['q1'],
                'q3': rouge_data[metric_type]['q3']
            })
    
    # Process BLEU
    bleu_data = data_dict['generation']['bleu']
    metrics_data.append({
        'model': model_name,
        'metric_group': 'bleu',
        'metric_type': 'score',
        'value': bleu_data['mean'],
        'q1': bleu_data['q1'],
        'q3': bleu_data['q3']
    })
    
    # Process Cosine Similarity
    cosine_data = data_dict['generation']['cosine_sim']
    metrics_data.append({
        'model': model_name,
        'metric_group': 'cosine_similarity',
        'metric_type': 'score',
        'value': cosine_data['mean'],
        'q1': cosine_data['q1'],
        'q3': cosine_data['q3']
    })
    
    # Process Self-Checker Accuracy
    self_checker_acc = data_dict['generation']['self_checker_accuracy']
    metrics_data.append({
        'model': model_name,
        'metric_group': 'self_checker_accuracy',
        'metric_type': 'score',
        'value': self_checker_acc,
        'q1': self_checker_acc,  # Using same value for q1/q3 since it's a single value
        'q3': self_checker_acc
    })
    
    return pd.DataFrame(metrics_data)

def visualize_generation_metrics(
        data_paths: List[str], 
        model_names: List[str], 
        title: str = 'RAG Performance Comparison',
        show_errors: bool = True,
        figsize: Tuple[int] = (24, 15),
        ):
    """
    Visualize RAG generation metrics from multiple model results.
    
    Args:
        data_pathss (list): List of dictionaries containing metrics data
        model_names (list): List of model names corresponding to each dictionary
        title (str): Title for the visualization
        figsize (tuple): Figure size (width, height)
        show_errors (bool): Whether to display error bars (default is True)
    """
    # Create DataFrames for all models
    dfs = []
    for data_path, model_name in zip(data_paths, model_names):
        df = create_generation_metrics_df(data_path, model_name)
        dfs.append(df)
    
    # Combine all dataframes
    df = pd.concat(dfs)
    
    # Create figure and subplots - 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()  # Flatten to make it easier to access
    
    # Define the metrics to plot in each position
    metric_mapping = {
        0: "rouge1",
        1: "rouge2", 
        2: "rougel",
        3: "bleu",
        4: "cosine_similarity",
        5: "self_checker_accuracy"
    }
    
    # Plot each metric in its specified position
    for i, metric_group in enumerate(metric_mapping.values()):
        ax = axes[i]
        
        # Filter data for this metric group
        group_df = df[df['metric_group'] == metric_group]
        
        # If no data for this metric, add a placeholder
        if len(group_df) == 0:
            ax.text(0.5, 0.5, f"No data for {metric_group}", 
                   horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{metric_group}', fontsize=16, pad=10)
            continue
            
        # Create bar plot
        bars = sns.barplot(
            data=group_df,
            x='metric_type',
            y='value',
            hue='model',
            ax=ax,
            width=0.7,
            capsize=0.1 if show_errors else 0,  # Control capsize based on show_errors
            err_kws={'linewidth': 2} if show_errors else None  # Control err_kws based on show_errors
        )
        
        if show_errors:
            # Add error bars for Q1 and Q3
            x_coords = np.arange(len(group_df['metric_type'].unique()))
            width = 0.7 / len(model_names)  # Adjust width based on number of models
            
            for j, model in enumerate(model_names):
                model_data = group_df[group_df['model'] == model]
                x = x_coords + (j - (len(model_names)-1)/2) * width
                
                # Plot Q1 and Q3 as error bars
                ax.vlines(x, model_data['q1'], model_data['q3'], 
                         color='black', linestyle='-', linewidth=2)
                
                # Add horizontal caps for Q1 and Q3
                for xi, q1, q3 in zip(x, model_data['q1'], model_data['q3']):
                    ax.hlines(q1, xi-0.1, xi+0.1, color='black', linewidth=2)
                    ax.hlines(q3, xi-0.1, xi+0.1, color='black', linewidth=2)
        
        ax.set_title(f'{metric_group}', fontsize=16, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='')
        
        # Add value labels on bars with mean
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()

def create_retrieval_metrics_df(data_path, model_name):
    """
    Create a DataFrame from the retrieval metrics JSON file.
    
    Args:
        data_path (str): Path to the JSON file containing retrieval metrics data
        model_name (str): Name of the model for the data
        
    Returns:
        pd.DataFrame: DataFrame containing the retrieval metrics
    """
    # Load JSON data
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    
    metrics_data = []
    
    # Process MAP (Mean Average Precision)
    if 'map' in data_dict['retrieval']:
        map_data = data_dict['retrieval']['map']
        for k, k_data in map_data.items():
            metrics_data.append({
                'model': model_name,
                'metric_group': 'map',
                'metric_type': f'k={k}',
                'value': k_data['mean'],
                'q1': k_data['q1'],
                'q3': k_data['q3']
            })
    
    # Process MRR (Mean Reciprocal Rank)
    if 'mrr' in data_dict['retrieval']:
        mrr_data = data_dict['retrieval']['mrr']
        for k, k_data in mrr_data.items():
            metrics_data.append({
                'model': model_name,
                'metric_group': 'mrr',
                'metric_type': f'k={k}',
                'value': k_data['mean'],
                'q1': k_data['q1'],
                'q3': k_data['q3']
            })
    
    # Process Hit Rate
    if 'hit' in data_dict['retrieval']:
        hit_data = data_dict['retrieval']['hit']
        for k, k_data in hit_data.items():
            metrics_data.append({
                'model': model_name,
                'metric_group': 'hit',
                'metric_type': f'k={k}',
                'value': k_data['mean'],
                'q1': k_data['q1'],
                'q3': k_data['q3']
            })
    
    return pd.DataFrame(metrics_data)

def visualize_retrieval_metrics(
        data_dicts, 
        model_names, 
        title='Retrieval Performance Comparison',
        figsize=(20, 6),
        ):
    """
    Visualize retrieval metrics from multiple model results.
    
    Args:
        data_dicts (list): List of dictionaries containing metrics data
        model_names (list): List of model names corresponding to each dictionary
        title (str): Title for the visualization
        figsize (tuple): Figure size (width, height)
    """
    # Create DataFrames for all models
    dfs = []
    for data_dict, model_name in zip(data_dicts, model_names):
        df = create_retrieval_metrics_df(data_dict, model_name)
        dfs.append(df)
    
    # Combine all dataframes
    df = pd.concat(dfs)
    
    # Get unique metric groups
    metric_groups = df['metric_group'].unique()
    
    # Create figure and subplots
    fig, axes = plt.subplots(1, len(metric_groups), figsize=figsize)
    
    # Handle case where there's only one subplot
    if len(metric_groups) == 1:
        axes = np.array([axes])
    
    # Plot each metric group
    for ax, metric_group in zip(axes, metric_groups):
        # Filter data for this metric group and create a copy
        group_df = df[df['metric_group'] == metric_group].copy()
        
        # Extract k values from metric_type using raw string for regex
        group_df.loc[:, 'k'] = group_df['metric_type'].str.extract(r'k=(\d+)').astype(int)
        
        # Create line plot with markers for each model
        for model in model_names:
            model_data = group_df[group_df['model'] == model]
            
            # Plot main line
            line = ax.plot(model_data['k'], model_data['value'], 
                          marker='o', label=model, linewidth=2, markersize=8)
            
            # Calculate label positions based on model index
            model_idx = model_names.index(model)
            is_first_model = model_idx == 0
            
            # Select key points to label (k=1, k=3, k=5, k=10)
            key_points = [(k, v) for k, v in zip(model_data['k'], model_data['value']) 
                         if k in [1, 3, 5, 10]]
            
            for k, v in key_points:
                # For first model, place labels above
                # For second model, place labels below when values are close
                if is_first_model:
                    offset_y = 15
                    va = 'bottom'
                else:
                    # Check if there's a value from first model at this k
                    other_model_data = group_df[
                        (group_df['model'] == model_names[0]) & 
                        (group_df['k'] == k)
                    ]
                    if not other_model_data.empty:
                        other_value = other_model_data.iloc[0]['value']
                        # If values are close, place label below
                        if abs(v - other_value) < 0.05:
                            offset_y = -15
                            va = 'top'
                        else:
                            offset_y = 15
                            va = 'bottom'
                    else:
                        offset_y = 15
                        va = 'bottom'
                
                ax.annotate(
                    f'{v:.3f}',
                    (k, v),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha='center',
                    va=va
                )
        
        ax.set_title(f'{metric_group} at K', fontsize=16, pad=10)
        ax.set_xlabel('K', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xticks([1, 3, 5, 10])
        ax.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 to accommodate labels
        ax.legend(title='')

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.show()
