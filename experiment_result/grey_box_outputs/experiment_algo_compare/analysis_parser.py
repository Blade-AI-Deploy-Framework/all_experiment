import pandas as pd
import json
import io
import os
import glob
import csv
from typing import Tuple, Dict, Any

def parse_attack_csv(filepath: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Parses a CSV file with a commented JSON metadata header.

    The file is expected to have two parts:
    1. A header section with lines starting with '#' containing JSON-like metadata
       for attack parameters.
    2. A body section with standard CSV data.

    Args:
        filepath: The path to the CSV file.

    Returns:
        A tuple containing:
        - A dictionary with the parsed attack parameters.
        - A pandas DataFrame with the experiment data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")

    metadata_lines = []
    csv_start_line_num = 0
    in_metadata_block = False
    header = []
    lines_for_df = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            stripped_line = line.strip()
            if stripped_line.startswith('#'):
                # Start capturing when we find the opening brace
                if stripped_line == '# {':
                    in_metadata_block = True
                    metadata_lines.append('{')
                # Stop capturing when we find the closing brace
                elif stripped_line == '# }':
                    metadata_lines.append('}')
                    in_metadata_block = False
                # Capture content inside the block
                elif in_metadata_block:
                    # Remove the '# ' prefix and append
                    metadata_lines.append(stripped_line[2:])
            # The first non-empty, non-comment line is the CSV header
            elif stripped_line and not header:
                header = stripped_line.split(',')
                csv_start_line_num = i + 1 # Data starts on the next line
            elif stripped_line:
                lines_for_df.append(line)

    # Join and parse the JSON metadata
    metadata_str = ''.join(metadata_lines)
    metadata = {}
    if metadata_str:
        try:
            metadata = json.loads(metadata_str)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse metadata as JSON from {os.path.basename(filepath)}. Error: {e}")
            print(f"Metadata string was: \"{metadata_str}\"")
    else:
        print(f"Warning: No metadata block found in {os.path.basename(filepath)}.")

    # More robust CSV parsing for the data body
    df = pd.DataFrame()
    if header and lines_for_df:
        # The issue is often in the 'hook_details' column which contains unescaped quotes and commas.
        # We will manually parse the lines, prioritizing columns we know are well-behaved.
        
        # Find indices of columns we absolutely need
        try:
            queries_idx = header.index('total_queries')
            # We no longer need the index for time, as we'll parse from the end of the row.
        except ValueError as e:
            print(f"Header in {os.path.basename(filepath)} is missing a required column: {e}")
            return metadata, df

        parsed_data = []
        # Use the csv module for robust handling of quoted fields
        csv_reader = csv.reader(lines_for_df)
        
        for row in csv_reader:
            try:
                # The malformed 'hook_details' column can corrupt parsing by index.
                # 'total_queries' is at the start (index 1) and should be safe.
                # 'total_time_s' is the very last column. We get it with index -1 for robustness.
                parsed_row = {
                    'total_queries': int(row[queries_idx]),
                    'total_time_s': float(row[-1]) # Safely grab the last element
                }
                parsed_data.append(parsed_row)
            except (IndexError, ValueError) as e:
                # This will skip rows that are genuinely malformed (e.g., empty or not enough columns)
                # print(f"\nSkipping malformed row in {os.path.basename(filepath)}: {row}. Error: {e}")
                continue
                
        if parsed_data:
            df = pd.DataFrame(parsed_data)

    return metadata, df

def analyze_algorithm_results(root_dir: str) -> Tuple[list[int], list[float], int]:
    """
    Analyzes all experiment subdirectories within a given root directory.

    For each experiment, it checks for a success image. If found, it parses
    the corresponding CSV to find the total queries needed for the attack.

    Args:
        root_dir: The path to the algorithm's result directory (e.g., 'square_liveness').

    Returns:
        A tuple containing:
        - A list of 'total_queries' for each successful attack.
        - A list of 'total_time_s' for each successful attack.
        - The total number of experiments conducted for the algorithm.
    """
    successful_queries = []
    successful_times = []
    
    try:
        # Get all items in the directory, filter for subdirectories, and sort them to ensure consistent order
        experiment_dirs = sorted([d for d in os.scandir(root_dir) if d.is_dir()], key=lambda d: d.path)
    except FileNotFoundError:
        print(f"Error: Directory not found '{root_dir}'")
        return [], [], 0
    
    # Special handling for Bandit: only consider the last 100 experiments
    if 'bandit_liveness' in root_dir:
        print("\nNote: For Bandit attack, only considering the last 100 experiments as requested.")
        experiment_dirs = experiment_dirs[-100:]
        
    total_experiments = len(experiment_dirs)
    if total_experiments == 0:
        print(f"Warning: No experiment subdirectories found in '{root_dir}'")
        return [], [], 0

    print(f"Found {total_experiments} total experiments in '{root_dir}'.")
    
    for i, exp_dir in enumerate(experiment_dirs):
        # Simple progress indicator
        print(f"\rProcessing folder {i+1}/{total_experiments}...", end="")

        # Check for a success image (e.g., 'success_...png')
        success_files = glob.glob(os.path.join(exp_dir.path, 'success*.png'))
        if not success_files:
            continue  # This experiment was not successful

        # Find the corresponding CSV file
        csv_files = glob.glob(os.path.join(exp_dir.path, '*.csv'))
        if not csv_files:
            print(f"\nWarning: Success image found in {exp_dir.path} but no CSV file.")
            continue
        
        # Assume there's only one CSV per folder
        csv_path = csv_files[0]
        
        try:
            _, df = parse_attack_csv(csv_path)
            if not df.empty and 'total_queries' in df.columns and 'total_time_s' in df.columns:
                # Get total_queries from the last row
                final_queries = df['total_queries'].iloc[-1]
                successful_queries.append(int(final_queries))

                # Get total_time_s from the last row
                final_time = df['total_time_s'].iloc[-1]
                successful_times.append(float(final_time))
            else:
                 print(f"\nWarning: CSV file {csv_path} is empty or missing required columns.")
        except Exception as e:
            print(f"\nError parsing or processing {csv_path}: {e}")

    print("\nProcessing complete.")
    return successful_queries, successful_times, total_experiments


# This block demonstrates how to use the function
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.ticker as mticker

    # Define the directories for each algorithm to be analyzed
    base_path = '.' # Assumes script is run from the 'experiment_algo_compare' directory
    algorithm_dirs = {
        'Square': 'square_liveness',
        'Bandit': 'bandit_liveness',
        'NES': 'nes_liveness',
        'ZOSignSGD': 'zosign_liveness',
        'SPSA': 'spsa_liveness'
    }

    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    # 1. Combined plot (all algorithms on one axis)
    fig_combined, ax_combined = plt.subplots(figsize=(12, 8))

    # 2. Subplots (each algorithm in its own horizontal plot)
    num_algos = len(algorithm_dirs)
    if num_algos == 4:
        fig_subplots, axs_subplots = plt.subplots(2, 2, figsize=(15, 12), sharey=True) # Increased height for suptitle
        axs_flat = axs_subplots.flatten()
    else:
        fig_subplots, axs_subplots = plt.subplots(1, num_algos, figsize=(6 * num_algos, 6), sharey=True)
        axs_flat = [axs_subplots] if num_algos == 1 else axs_subplots

    if num_algos == 1: # Ensure axs_subplots is always iterable
        axs_subplots = [axs_subplots]

    # Define plot styles for each algorithm for better visual distinction
    plot_styles = {
        'Square': {'marker': 'o', 'linestyle': '-', 'color': 'royalblue'},
        'Bandit': {'marker': 'P', 'linestyle': '-', 'color': 'darkorange'},
        'NES':    {'marker': '^', 'linestyle': '-', 'color': 'forestgreen'},
        'ZOSignSGD': {'marker': 'D', 'linestyle': '-', 'color': 'crimson'},
        'SPSA': {'marker': 's', 'linestyle': '-', 'color': 'purple'}
    }

    # Store results for plotting after the analysis loop
    plot_data = {}

    # --- Analysis and Plotting Loop ---
    for i, (algo_name, algo_dir) in enumerate(algorithm_dirs.items()):
        full_path = os.path.join(base_path, algo_dir)
        print(f"\n{'='*10} Analyzing {algo_name} {'='*10}")
        
        successful_queries, successful_times, total_experiments = analyze_algorithm_results(full_path)
        
        if total_experiments == 0:
            print("No experiments to analyze.")
            continue
            
        num_successes = len(successful_queries)
        print(f"Result: {num_successes} successful attacks out of {total_experiments} experiments.")
        
        if num_successes > 0:
            # Store data for plotting
            plot_data[algo_name] = {
                'queries': np.insert(sorted(successful_queries), 0, 0),
                'rate': np.insert(np.arange(1, num_successes + 1) / total_experiments, 0, 0),
                'label': f'{algo_name} Attack (Success Rate: {num_successes/total_experiments:.2%})',
                'style': plot_styles.get(algo_name, {})
            }

    # --- Plotting Section ---

    # 1. --- Combined Plot (All 5 algorithms) ---
    for algo_name, data in plot_data.items():
        ax_combined.plot(data['queries'], data['rate'], **data['style'], label=data['label'], markersize=4)

    # 2. --- Individual Plots (All 5 algorithms) ---
    for algo_name, data in plot_data.items():
        fig_single, ax_single = plt.subplots(figsize=(12, 8))
        ax_single.plot(data['queries'], data['rate'], **data['style'], label=data['label'], markersize=6)
        # ... (rest of the single plot configuration)
        ax_single.set_title(f'{algo_name} Attack Success Rate vs. Number of Queries', fontsize=18)
        ax_single.set_xlabel('Number of Queries', fontsize=14)
        ax_single.set_ylabel('Cumulative Attack Success Rate', fontsize=14)
        ax_single.legend(fontsize=12)
        ax_single.set_ylim(0, 1.05)
        ax_single.set_xlim(left=0)
        ax_single.grid(False)
        ax_single.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax_single.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True))
        single_plot_filename = f'{algo_name.lower()}_attack_success_rate.pdf'
        try:
            fig_single.savefig(single_plot_filename, bbox_inches='tight')
            print(f"Individual plot saved as '{single_plot_filename}'")
        except Exception as e:
            print(f"\nError saving individual plot for {algo_name}: {e}")
        plt.close(fig_single)

    # 3. --- Subplots (Exclude Bandit) ---
    plot_data_subplots = {k: v for k, v in plot_data.items() if k != 'Bandit'}
    num_subplots = len(plot_data_subplots)

    if num_subplots > 0:
        # Create two sets of subplots for the different layouts
        # Layout 1: 2x2 grid
        fig_subplots_2x2, axs_subplots_2x2 = plt.subplots(2, 2, figsize=(15, 12), sharey=True)
        axs_flat_2x2 = axs_subplots_2x2.flatten()
        
        # Layout 2: 1x4 horizontal row
        fig_subplots_1x4, axs_subplots_1x4 = plt.subplots(1, num_subplots, figsize=(6 * num_subplots, 6), sharey=True)
        
        for i, (algo_name, data) in enumerate(plot_data_subplots.items()):
            # --- Plot on 2x2 grid ---
            ax_subplot_2x2 = axs_flat_2x2[i]
            ax_subplot_2x2.plot(data['queries'], data['rate'], **data['style'], markersize=6)
            ax_subplot_2x2.set_title(f'{algo_name} Attack', fontsize=16)
            ax_subplot_2x2.grid(False)
            ax_subplot_2x2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))

            # --- Plot on 1x4 grid ---
            ax_subplot_1x4 = axs_subplots_1x4[i]
            ax_subplot_1x4.plot(data['queries'], data['rate'], **data['style'], markersize=6)
            ax_subplot_1x4.set_title(f'{algo_name} Attack', fontsize=16)
            ax_subplot_1x4.set_xlabel('Number of Queries', fontsize=14)
            ax_subplot_1x4.grid(False)
            ax_subplot_1x4.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10, integer=True))


        # --- Configure and add legend to 2x2 subplots ---
        handles = [plt.Line2D([0], [0], **data['style']) for data in plot_data_subplots.values()]
        labels = [data['label'].split(' (')[0] for data in plot_data_subplots.values()] # Get clean name for legend
        fig_subplots_2x2.legend(handles, labels, loc='center left', fontsize=14)

        axs_subplots_2x2[0, 0].set_ylabel('Cumulative Attack Success Rate', fontsize=14)
        axs_subplots_2x2[1, 0].set_ylabel('Cumulative Attack Success Rate', fontsize=14)
        axs_subplots_2x2[0, 0].set_xlabel('')
        axs_subplots_2x2[0, 1].set_xlabel('')
        axs_subplots_2x2[1, 0].set_xlabel('Number of Queries', fontsize=14)
        axs_subplots_2x2[1, 1].set_xlabel('Number of Queries', fontsize=14)
        axs_subplots_2x2[0, 0].set_ylim(0, 1.05)
        axs_subplots_2x2[0, 0].yaxis.set_major_locator(mticker.MultipleLocator(0.1))

        fig_subplots_2x2.suptitle('Algorithm Success Rate Comparison', fontsize=22, y=1.02)
        fig_subplots_2x2.tight_layout()
        fig_subplots_2x2.subplots_adjust(left=0.2)

        # --- Configure and add legend to 1x4 subplots ---
        fig_subplots_1x4.legend(handles, labels, loc='center left', fontsize=18)
        axs_subplots_1x4[0].set_ylabel('Cumulative Attack Success Rate', fontsize=14)
        axs_subplots_1x4[0].set_ylim(0, 1.05)
        axs_subplots_1x4[0].yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        fig_subplots_1x4.suptitle('Algorithm Success Rate Comparison', fontsize=22, y=1.02)
        fig_subplots_1x4.tight_layout()
        fig_subplots_1x4.subplots_adjust(left=0.2)


    # Final configuration for the combined plot is done here
    ax_combined.set_title('Comparison of Attack Success Rate vs. Number of Queries', fontsize=18)
    ax_combined.set_xlabel('Number of Queries', fontsize=14)
    ax_combined.set_ylabel('Cumulative Attack Success Rate', fontsize=14)
    ax_combined.legend(title='Algorithms', fontsize=12)
    ax_combined.set_ylim(0, 1.05) # Y-axis from 0 to 100%
    ax_combined.set_xlim(left=0) # X-axis starts from 0
    ax_combined.grid(False)
    ax_combined.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax_combined.xaxis.set_major_locator(mticker.MaxNLocator(nbins=20, integer=True))
    
    # --- Save Plots ---
    try:
        # Save combined plot
        output_filename_combined = 'attack_success_rate_comparison.pdf'
        fig_combined.savefig(output_filename_combined, bbox_inches='tight')
        print(f"\nCombined plot successfully saved as '{output_filename_combined}'")
        
        # Save subplots if they were created
        if num_subplots > 0:
            output_filename_2x2 = 'attack_subplots_2x2_comparison.pdf'
            fig_subplots_2x2.savefig(output_filename_2x2, bbox_inches='tight')
            print(f"2x2 Subplots successfully saved as '{output_filename_2x2}'")

            output_filename_1x4 = 'attack_subplots_1x4_comparison.pdf'
            fig_subplots_1x4.savefig(output_filename_1x4, bbox_inches='tight')
            print(f"1x4 Subplots successfully saved as '{output_filename_1x4}'")
        
    except Exception as e:
        print(f"\nError saving plots: {e}")
    finally:
        plt.close(fig_combined)
        if num_subplots > 0:
            plt.close(fig_subplots_2x2)
            plt.close(fig_subplots_1x4)
