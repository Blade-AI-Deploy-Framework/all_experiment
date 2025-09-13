import os
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

def parse_csv(csv_path):
    """
    Parses a single CSV file to extract the total queries for a successful attack.
    Returns the total queries if successful, otherwise None.
    """
    try:
        # First, try to read with pandas
        df = pd.read_csv(csv_path, comment='#', engine='python')
        if not df.empty and 'total_queries' in df.columns:
            return df['total_queries'].iloc[-1]
    except Exception:
        # If pandas fails, parse manually
        try:
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
                return int(parts[1])
        except Exception:
            return None
    return None

def parse_loss_data(csv_path):
    """
    Parses a single CSV file to extract total_queries and loss data.
    """
    try:
        df = pd.read_csv(csv_path, comment='#', engine='python')
        if not df.empty and 'total_queries' in df.columns and 'loss' in df.columns:
            return df[['total_queries', 'loss']]
    except Exception as e:
        print(f"Could not parse loss data from {csv_path}: {e}")
        return None
    return None

def analyze_model_attacks(model_dir):
    """
    Analyzes all attack results for a single model.
    Returns a list of total queries for each successful attack.
    """
    print(f"Analyzing model: {os.path.basename(model_dir)}")
    successful_attack_queries = []
    successful_attack_loss_dfs = []  # To store loss dataframes
    total_attempts = 0  # Initialize total attempts counter

    # Check if model_dir exists and is a directory
    if not os.path.isdir(model_dir):
        print(f"Error: Directory '{model_dir}' does not exist or is not a directory.")
        return [], 0, []
    
    attack_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
    total_attempts = len(attack_dirs)

    for subdir in attack_dirs:
        success_files = glob.glob(os.path.join(subdir, 'successful_attack*.png'))
        if success_files:
            csv_files = glob.glob(os.path.join(subdir, '*.csv'))
            if csv_files:
                csv_path = csv_files[0]
                queries = parse_csv(csv_path)
                if queries is not None:
                    successful_attack_queries.append(queries)

                loss_df = parse_loss_data(csv_path)
                if loss_df is not None:
                    successful_attack_loss_dfs.append(loss_df)
    
    return successful_attack_queries, total_attempts, successful_attack_loss_dfs

def plot_success_rate_vs_queries(all_results):
    """
    Plots the success rate vs. number of queries for all models on a single plot.
    """
    # Configure matplotlib for Chinese font
    # plt.rcParams['font.sans-serif'] = ['Heiti TC']
    plt.rcParams['axes.unicode_minus'] = False

    # Set a fixed upper limit for queries
    max_queries_limit = 80000
    query_bins = np.arange(0, max_queries_limit + 1000, 1000)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in all_results.items():
        queries = data['queries']
        total_attacks = data['total_attempts']

        if not queries:
            print(f"Skipping model {model_name} because there is no successful attack data.")
            continue

        # If total_attacks is 0, skip to avoid division by zero
        if total_attacks == 0:
            print(f"Skipping model {model_name} because there were no attack attempts.")
            continue
            
        success_counts = np.zeros(len(query_bins))

        # For each query bin, count how many attacks succeeded within that query limit
        for i, bin_edge in enumerate(query_bins):
            successful_in_bin = sum(1 for q in queries if q <= bin_edge)
            success_counts[i] = successful_in_bin
        
        success_rate = (success_counts / total_attacks) * 100
        
        ax.plot(query_bins, success_rate, linestyle='-', label="insightface vertify pipeline")

    ax.set_title('Attack Success Rate vs. Number of Queries')
    ax.set_xlabel('Number of Queries')
    ax.set_ylabel('Cumulative Attack Success Rate (%)')
    # plt.grid(True)
    ax.legend()
    ax.set_xlim(left=0, right=max_queries_limit)
    ax.set_ylim(bottom=0, top=101)

    # Save the plot to a file
    output_filename = "attack_success_rate_face_analysis_cli.pdf"
    fig.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"Plot saved to: {output_filename}")
    plt.close(fig) # Close the figure to free memory


def plot_average_loss_curve(all_results):
    """
    Plots the average loss curve with std deviation for each model.
    """
    plt.rcParams['axes.unicode_minus'] = False
    
    max_queries_limit = 80000
    # Create a common query axis for interpolation
    interp_queries = np.arange(0, max_queries_limit + 1, 100)

    for model_name, data in all_results.items():
        loss_dfs = data.get('loss_dfs', [])

        if len(loss_dfs) < 2: # Need at least 2 runs to compute std dev
            print(f"Skipping loss plot for model {model_name}: not enough successful attacks data ({len(loss_dfs)} found).")
            continue

        # Interpolate each attack's loss data onto the common query axis
        interpolated_losses = []
        for df in loss_dfs:
            # np.interp needs increasing x-values. Sort and remove duplicates to be safe.
            df = df.sort_values('total_queries').drop_duplicates('total_queries')
            interp_loss = np.interp(interp_queries, df['total_queries'], df['loss'])
            interpolated_losses.append(interp_loss)

        if not interpolated_losses:
            continue

        # Convert to numpy array for easier stats calculation
        losses_array = np.array(interpolated_losses)
        
        # Calculate mean and standard deviation
        mean_loss = np.mean(losses_array, axis=0)
        std_loss = np.std(losses_array, axis=0)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the mean loss
        ax.plot(interp_queries, mean_loss, label='Average Loss')
        
        # Plot the shaded standard deviation area
        ax.fill_between(interp_queries, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, label='Standard Deviation')
        
        ax.set_title(f'Average Attack Loss vs. Number of Queries')
        ax.set_xlabel('Number of Queries')
        ax.set_ylabel('Loss')
        
        # Add vertical lines as requested
        ax.axvline(x=1000, color='darkgoldenrod', linestyle='--')
        ax.axvline(x=11000, color='red', linestyle='--')
        ax.axvline(x=45000, color='green', linestyle='--')

        # plt.grid(True)
        ax.legend()
        ax.set_xlim(left=0, right=max_queries_limit)
        ax.set_ylim(bottom=0)
    
        output_filename = f'average_loss_curve_{model_name}.pdf'
        fig.savefig(output_filename, format="pdf", bbox_inches="tight")
        print(f"Plot saved to: {output_filename}")
        plt.close(fig)


def plot_loss_boxplot_vs_queries(all_results):
    """
    Plots a box plot of loss distribution at different query intervals for each model.
    """
    plt.rcParams['axes.unicode_minus'] = False

    for model_name, data in all_results.items():
        loss_dfs = data.get('loss_dfs', [])

        if not loss_dfs:
            print(f"Skipping box plot for model {model_name}: no successful attacks data.")
            continue

        # Combine all dataframes into one
        combined_df = pd.concat(loss_dfs, ignore_index=True)

        # Define bins for total_queries
        max_queries = combined_df['total_queries'].max()
        num_bins = 20  # Define a fixed number of bins
        bins = np.linspace(0, max_queries, num_bins + 1)
        
        # Create a new column for query bins
        combined_df['query_bin'] = pd.cut(combined_df['total_queries'], bins=bins, right=False, include_lowest=True)

        # Prepare data for boxplot: a list of lists of loss values
        grouped = combined_df.groupby('query_bin')['loss']
        
        box_data = [group.dropna().tolist() for name, group in grouped]
        
        # Create cleaner labels from the bin midpoints
        labels = [f'{int(interval.mid)}' for interval, group in grouped]

        # Filter out empty lists from box_data and corresponding labels
        filtered_data = []
        filtered_labels = []
        for i in range(len(box_data)):
            if box_data[i]:
                filtered_data.append(box_data[i])
                filtered_labels.append(labels[i])

        if not filtered_data:
            print(f"Skipping box plot for model {model_name}: no data to plot after binning.")
            continue

        plt.figure(figsize=(15, 8))
        plt.boxplot(filtered_data)
        
        plt.title(f'Loss Distribution vs. Number of Queries')
        plt.xlabel('Number of Queries (binned)')
        plt.ylabel('Loss')
        plt.xticks(ticks=np.arange(1, len(filtered_labels) + 1), labels=filtered_labels, rotation=45, ha="right")
        # plt.grid(True, axis='y')
        plt.tight_layout()  # Adjust layout to prevent labels overlapping

        output_filename = f'loss_boxplot_{model_name}.png'
        plt.savefig(output_filename)
        print(f"Plot saved to: {output_filename}")
        plt.close()


if __name__ == "__main__":
    # Search for directories ending with '_attack_results' in the current directory
    model_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and d.endswith('_attack_results')]
    
    all_model_results = {}

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir).replace('_attack_results', '')
        queries, total_attempts, loss_dfs = analyze_model_attacks(model_dir)
        all_model_results[model_name] = {
            'queries': queries,
            'total_attempts': total_attempts,
            'loss_dfs': loss_dfs
        }
        
    plot_success_rate_vs_queries(all_model_results)
    plot_average_loss_curve(all_model_results)
    plot_loss_boxplot_vs_queries(all_model_results) 