import os
import json
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Black Box Analysis (inspired by attack_analysis.py) ---

def analyze_black_box_attacks_refined(directory):
    """
    Analyzes black box attack results.
    Success is defined as status != "BUDGET_EXHAUSTED".
    Returns a list of queries for successful attacks and the total number of attacks.
    """
    if not os.path.isdir(directory):
        print(f"警告: 黑盒目录不存在: {directory}")
        return [], 0
    
    print(f"正在分析黑盒攻击: {directory}")
    successful_queries = []
    total_attacks = 0
    
    sub_dirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for sub_dir in sub_dirs:
        summary_file = os.path.join(directory, sub_dir, 'attack_summary.json')
        if not os.path.exists(summary_file):
            continue

        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"警告: 无法解析JSON: {summary_file}")
            continue
        
        total_attacks += 1
        
        if data.get('status') != "BUDGET_EXHAUSTED":
            if 'results' in data and 'total_queries' in data['results']:
                successful_queries.append(data['results']['total_queries'])

    return successful_queries, total_attacks

# --- Grey Box Analysis (inspired by analyze_attack_results.py) ---

def analyze_grey_box_attacks_refined(model_dir):
    """
    Analyzes grey box attack results.
    Success is defined by finding a 'successful_attack*.png' and iteration count <= 100.
    Returns a list of queries for successful attacks and the total number of attacks.
    """
    if not os.path.isdir(model_dir):
        print(f"警告: 灰盒目录不存在: {model_dir}")
        return [], 0
        
    print(f"正在分析灰盒攻击: {model_dir}")
    successful_queries = []
    
    try:
        attack_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    except FileNotFoundError:
        return [], 0
    
    total_attacks = len(attack_dirs)

    for subdir in attack_dirs:
        success_files = glob.glob(os.path.join(subdir, 'successful_attack*.png'))
        if success_files:
            csv_files = glob.glob(os.path.join(subdir, '*.csv'))
            if csv_files:
                csv_path = csv_files[0]
                try:
                    # Try manual parsing first to check iteration count
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()
                    last_data_line = None
                    for line in reversed(lines):
                        line = line.strip()
                        if line and not line.startswith('#') and 'iteration,total_queries' not in line:
                            last_data_line = line
                            break
                    
                    if last_data_line:
                        parts = last_data_line.split(',')
                        last_iteration = int(parts[0])
                        if last_iteration <= 100: # Success condition from analyze_attack_results.py
                            queries = int(parts[1])
                            successful_queries.append(queries)
                except Exception:
                    print(f"警告: 无法处理CSV文件 {csv_path}")

    return successful_queries, total_attacks

# --- Plotting (inspired by attack_analysis.py) ---

def plot_cumulative_rate(ax, queries, total_attacks, label, style):
    """Helper function to plot one cumulative success rate curve using binning for smoothness."""
    if not queries or total_attacks == 0:
        print(f"Warning: No data to plot for '{label}'.")
        return

    # --- Binning Logic for smoother curves ---
    max_queries_limit = 12000
    bin_size = 200
    query_bins = np.arange(0, max_queries_limit + bin_size, bin_size)

    success_counts = np.zeros(len(query_bins))
    for i, bin_edge in enumerate(query_bins):
        successful_in_bin = sum(1 for q in queries if q <= bin_edge)
        success_counts[i] = successful_in_bin
    
    success_rate = (success_counts / total_attacks) * 100
    
    # label_with_count = f"{label} ({len(queries)}/{total_attacks} Success)"

    ax.plot(query_bins, success_rate, marker=style.get('marker'), linestyle=style.get('linestyle'), markersize=4, label=label, color=style.get('color'))

def format_ax(ax, title, show_ylabel=True):
    """Applies common formatting to a subplot Axes."""
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Number of Queries', fontsize=12)
    if show_ylabel:
        ax.set_ylabel('Cumulative Success Rate (%)', fontsize=12)
    
    ax.set_ylim(0, 101)
    yticks = np.arange(0, 101, 10)
    ax.set_yticks(yticks)
    yticklabels = [str(int(tick)) for tick in yticks]
    if yticklabels:
        yticklabels[0] = ''
    ax.set_yticklabels(yticklabels)

    ax.set_xlim(left=0, right=12000)
    ax.set_xticks(np.arange(0, 12001, 2000))
    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1)
    # ax.grid(True)

def generate_title_from_task_name(task_name):
    """Generates a formatted title 'Function Model' from a task_name string."""
    parts = task_name.split('_')
    
    models = {
        "googlenet": "GoogleNet",
        "ssrnet": "SSRNet",
        "fsanet": "FSANet",
        "ferplus": "FerPlus"
    }
    
    functions = {
        "age": "Age",
        "gender": "Gender",
        "emotion": "Emotion",
        "headpose": "Headpose"
    }
    
    model_part = None
    function_part = None
    
    for part in parts:
        if part in models:
            model_part = models[part]
        elif part in functions:
            function_part = functions[part]
            
    if not model_part or not function_part:
        # Fallback for unknown task names
        return task_name.replace("_", " ").title()

    return f"{function_part} {model_part}"

def plot_single_task(ax, task_name, results_dict, plot_styles, show_ylabel=True):
    """Plots data on an ax and formats it."""
    for name, data in results_dict.items():
        if data:
            plot_cumulative_rate(ax, data['queries'], data['total'], name, plot_styles.get(name))

    title = generate_title_from_task_name(task_name)
    format_ax(ax, title, show_ylabel)

# --- Main Execution ---

if __name__ == "__main__":
    
    plt.style.use('seaborn-v0_8-white')
    
    ALL_TASKS = {
        "emotion_ferplus": {
            "black_box_easy_dir": "black_box_outputs/emotion_ferplus_easy",
            "black_box_hard_dir": "black_box_outputs/emotion_ferplus_hard",
            "grey_box_exp1_dir": "grey_box_outputs/experiment_1/outputs/emotion_ferplus_mnn_attack_results",
            "grey_box_exp2_dir": "grey_box_outputs/experiment_2/outputs/emotion_ferplus_mnn_attack_results",
        },
        "gender_googlenet": {
            "black_box_easy_dir": "black_box_outputs/gender_googlenet_easy",
            "black_box_hard_dir": "black_box_outputs/gender_googlenet_hard",
            "grey_box_exp1_dir": "grey_box_outputs/experiment_1/outputs/gender_googlenet_mnn_attack_results",
            "grey_box_exp2_dir": "grey_box_outputs/experiment_2/outputs/gender_googlenet_mnn_attack_results",
        },
        "fsanet_headpose": {
            "black_box_easy_dir": "black_box_outputs/fsanet_headpose_easy",
            "black_box_hard_dir": "black_box_outputs/fsanet_headpose_hard",
            "grey_box_exp1_dir": "grey_box_outputs/experiment_1/outputs/fsanet_headpose_mnn_attack_results",
            "grey_box_exp2_dir": "grey_box_outputs/experiment_2/outputs/fsanet_headpose_mnn_attack_results",
        },
        "ssrnet_age": {
            "black_box_easy_dir": "black_box_outputs/ssrnet_age_easy",
            "black_box_hard_dir": "black_box_outputs/ssrnet_age_hard",
            "grey_box_exp1_dir": "grey_box_outputs/experiment_1/outputs/ssrnet_age_mnn_attack_results",
            "grey_box_exp2_dir": "grey_box_outputs/experiment_2/outputs/ssrnet_age_mnn_attack_results",
        },
        "age_googlenet": {
            "black_box_easy_dir": "black_box_outputs/age_googlenet_easy",
            "black_box_hard_dir": "black_box_outputs/age_googlenet_hard",
            "grey_box_exp1_dir": None,
            "grey_box_exp2_dir": "grey_box_outputs/experiment_2/outputs/age_googlenet_mnn_attack_results",
        }
    }
    
    PLOT_STYLES = {
        'Ttba Attack (Low Threshold)':    {'linestyle': '-', 'marker': 'o'},
        'Ttba Attack (High Threshold)':   {'linestyle': '--', 'marker': 's'},
        'NES Attack (High Threshold)':    {'linestyle': '-.', 'marker': 'x'},
        'NES Attack (Low Threshold)':     {'linestyle': ':', 'marker': 'd'},
    }

    # --- 1. Generate Combined Matrix Plot ---
    print("--- Generating combined matrix plot ---")
    tasks_for_matrix = ["emotion_ferplus", "gender_googlenet", "ssrnet_age", "fsanet_headpose"]
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()

    for i, task_name in enumerate(tasks_for_matrix):
        results = {}
        dirs = ALL_TASKS[task_name]
        q_bb_easy, n_bb_easy = analyze_black_box_attacks_refined(dirs['black_box_easy_dir'])
        results['Ttba Attack (Low Threshold)'] = {'queries': q_bb_easy, 'total': n_bb_easy}
        q_bb_hard, n_bb_hard = analyze_black_box_attacks_refined(dirs['black_box_hard_dir'])
        results['Ttba Attack (High Threshold)'] = {'queries': q_bb_hard, 'total': n_bb_hard}
        if dirs.get('grey_box_exp1_dir'):
            q_gb_exp1, n_gb_exp1 = analyze_grey_box_attacks_refined(dirs['grey_box_exp1_dir'])
            results['NES Attack (High Threshold)'] = {'queries': q_gb_exp1, 'total': n_gb_exp1}
        if dirs.get('grey_box_exp2_dir'):
            q_gb_exp2, n_gb_exp2 = analyze_grey_box_attacks_refined(dirs['grey_box_exp2_dir'])
            results['NES Attack (Low Threshold)'] = {'queries': q_gb_exp2, 'total': n_gb_exp2}
        
        # Only show y-label for plots in the first column
        plot_single_task(axes[i], task_name, results, PLOT_STYLES, show_ylabel=(i % 3 == 0))

    # Hide unused subplots
    for i in range(len(tasks_for_matrix), len(axes)):
        axes[i].axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.68, 0.25), fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend

    output_filename = 'comparison_matrix.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Combined comparison matrix saved to: {output_filename}")
    plt.close(fig)

    # --- 2. Generate Individual Plots ---
    print("\n--- Generating individual plots ---")
    for task_name, dirs in ALL_TASKS.items():
        results = {}
        q_bb_easy, n_bb_easy = analyze_black_box_attacks_refined(dirs['black_box_easy_dir'])
        results['Ttba Attack (Low Threshold)'] = {'queries': q_bb_easy, 'total': n_bb_easy}
        q_bb_hard, n_bb_hard = analyze_black_box_attacks_refined(dirs['black_box_hard_dir'])
        results['Ttba Attack (High Threshold)'] = {'queries': q_bb_hard, 'total': n_bb_hard}
        if dirs.get('grey_box_exp1_dir'):
            q_gb_exp1, n_gb_exp1 = analyze_grey_box_attacks_refined(dirs['grey_box_exp1_dir'])
            results['NES Attack (High Threshold)'] = {'queries': q_gb_exp1, 'total': n_gb_exp1}
        if dirs.get('grey_box_exp2_dir'):
            q_gb_exp2, n_gb_exp2 = analyze_grey_box_attacks_refined(dirs['grey_box_exp2_dir'])
            results['NES Attack (Low Threshold)'] = {'queries': q_gb_exp2, 'total': n_gb_exp2}

        fig_ind, ax_ind = plt.subplots(figsize=(12, 8))
        plot_single_task(ax_ind, task_name, results, PLOT_STYLES)
        ax_ind.legend(fontsize=12)
        
        output_filename = f'comparison_plot_{task_name}.png'
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"Saved individual plot: {output_filename}")
        plt.close(fig_ind)

    print("\nAll plots have been generated.")
