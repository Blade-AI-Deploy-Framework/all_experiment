import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import io

def find_attack_dirs(base_dirs):
    """
    Finds all attack directories from a list of base directories, handling duplicates.
    The last directory in the list has the highest precedence.
    """
    attack_map = {}
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            print(f"Warning: Directory '{base_dir}' not found, skipping.")
            continue
        for subdir_name in os.listdir(base_dir):
            subdir_path = os.path.join(base_dir, subdir_name)
            if os.path.isdir(subdir_path):
                match = re.match(r'(.+)_attack_\d{8}-\d{6}', subdir_name)
                if match:
                    image_name = match.group(1)
                    attack_map[image_name] = subdir_path
    print(f"Found {len(attack_map)} unique attack instances to analyze.")
    return attack_map

def analyze_attack(subdir_path):
    """
    Analyzes a single attack directory to determine success and query count.
    """
    is_success = bool(glob.glob(os.path.join(subdir_path, 'success*')))
    
    csv_files = glob.glob(os.path.join(subdir_path, '*.csv'))
    if not csv_files:
        return is_success, None

    csv_path = csv_files[0]
    query_count = None
    try:
        with open(csv_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        # Find the last non-empty, non-comment line
        last_data_line = None
        for line in reversed(lines):
            line = line.strip()
            if line and not line.startswith('#'):
                last_data_line = line
                break
        
        if last_data_line:
            parts = last_data_line.split(',')
            if len(parts) > 2:
                query_count = int(parts[2])
            else:
                query_count = 0
        else:
            query_count = 0

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")

    return is_success, query_count


def parse_loss_data(csv_path):
    """
    Parses a single complex CSV file to extract total_queries and loss data.
    """
    try:
        with open(csv_path, 'r', errors='ignore') as f:
            lines = f.readlines()
        
        # Filter out comments and extract data
        data_str = ""
        for line in lines:
            if not line.strip().startswith('#'):
                data_str += line
        
        # Read the string data into pandas, assuming no header
        # Let's assume queries are in column 2 and loss is in column 3 (0-indexed)
        df = pd.read_csv(io.StringIO(data_str), header=None, usecols=[2, 3])
        df.columns = ['queries', 'loss']
        
        # Ensure data is numeric, coercing errors to NaN and dropping them
        df['queries'] = pd.to_numeric(df['queries'], errors='coerce')
        df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
        df.dropna(inplace=True)
        
        return df

    except Exception as e:
        print(f"Could not parse loss data from {csv_path}: {e}")
        return None


def plot_median_loss_curve(all_loss_dfs, model_name="face_analysis_cli"):
    """
    Plots the median loss curve with the interquartile range (IQR).
    """
    if len(all_loss_dfs) < 2:
        print(f"Skipping loss plot: not enough successful attacks data ({len(all_loss_dfs)} found).")
        return

    # plt.style.use('seaborn-v0_8-whitegrid')
    max_queries_limit = max(df['queries'].max() for df in all_loss_dfs if not df.empty)
    interp_queries = np.linspace(0, max_queries_limit, 500) # Use 500 points for a smooth curve

    interpolated_losses = []
    for df in all_loss_dfs:
        if df.empty:
            continue
        df = df.sort_values('queries').drop_duplicates('queries')
        interp_loss = np.interp(interp_queries, df['queries'], df['loss'])
        interpolated_losses.append(interp_loss)

    if not interpolated_losses:
        print("No loss data to plot after interpolation.")
        return

    losses_array = np.array(interpolated_losses)
    
    # Calculate median and quartiles
    median_loss = np.median(losses_array, axis=0)
    q1_loss = np.quantile(losses_array, 0.25, axis=0)
    q3_loss = np.quantile(losses_array, 0.75, axis=0)
    
    plt.figure(figsize=(12, 8))
    
    plt.plot(interp_queries, median_loss, label='Median Loss')
    plt.fill_between(interp_queries, q1_loss, q3_loss, alpha=0.2, label='Interquartile Range (IQR)')
    
    plt.title(f'{model_name} Median Attack Loss vs. Number of Queries', fontsize=16)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    # plt.grid(True)
    plt.legend()
    plt.xlim(left=0, right=max_queries_limit)
    plt.ylim(bottom=29)

    output_filename = f'median_loss_curve_{model_name}.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to: {output_filename}")
    plt.close()


def main():
    """
    Main function to run the analysis and plotting.
    """
    base_dirs = ['outputs_ancface', 'outputs_ancface_2']
    attack_dirs = find_attack_dirs(base_dirs)

    results = []
    all_loss_dfs = []
    for image_name, subdir_path in attack_dirs.items():
        success, queries = analyze_attack(subdir_path)
        if queries is not None:
            results.append({'image': image_name, 'success': success, 'queries': queries})

        if success:
            csv_path = glob.glob(os.path.join(subdir_path, '*.csv'))
            if csv_path:
                loss_df = parse_loss_data(csv_path[0])
                if loss_df is not None:
                    all_loss_dfs.append(loss_df)

    if not results:
        print("No results found to analyze.")
        return

    df = pd.DataFrame(results)
    total_attacks = len(df)
    print(f"Analyzed a total of {total_attacks} attacks.")

    successful_attacks = df[df['success']].sort_values('queries')
    
    if successful_attacks.empty:
        print("No successful attacks found.")
        return
        
    print(f"Number of successful attacks: {len(successful_attacks)}")

    # Save the list of successful attack image names to a file
    successful_image_names = successful_attacks['image'].tolist()
    with open('successful_attacks.txt', 'w') as f:
        for name in sorted(successful_image_names):
            f.write(f"{name}\n")
    print("List of successful attack images saved to successful_attacks.txt")

    # Calculate cumulative success rate using bins for a smoother curve
    max_queries_limit = 6000
    # Create 100 bins/points for the curve to add more detail
    query_bins = np.linspace(0, max_queries_limit, 100, dtype=int)
    success_rate_axis = []

    for q in query_bins:
        success_count = (successful_attacks['queries'] <= q).sum()
        success_rate = success_count / total_attacks
        success_rate_axis.append(success_rate)

    # Plotting
    # plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    # Plot a smooth line without markers
    plt.plot(query_bins, success_rate_axis, linestyle='-', label='Face++ Recognition Pipeline')
    
    plt.title('Attack Success Rate vs. Number of Queries', fontsize=16)
    plt.xlabel('Number of Queries', fontsize=12)
    plt.ylabel('Attack Success Rate', fontsize=12)
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.xlim(0, max_queries_limit)
    plt.ylim(0, 0.6)
    
    # Set y-axis ticks to be percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    
    plt.tight_layout()
    
    output_filename = 'attack_success_rate.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Chart saved to: {output_filename}")
    #plt.show()

    # Add the call to the new plotting function
    if all_loss_dfs:
        plot_median_loss_curve(all_loss_dfs)


if __name__ == '__main__':
    main()
