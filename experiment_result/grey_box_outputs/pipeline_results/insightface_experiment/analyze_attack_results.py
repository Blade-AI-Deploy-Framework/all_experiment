import os
import pandas as pd
import glob
import sys

def analyze_directory(directory):
    """
    Analyzes the attack results within a specific directory.

    Args:
        directory (str): The path to the directory to analyze.
    """
    print(f"Analyzing directory: {directory}")

    # Check if directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        print("-" * 40)
        return

    # Find all immediate subdirectories
    try:
        subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    except OSError as e:
        print(f"Error: Cannot read directory {directory}: {e}")
        print("-" * 40)
        return
        
    total_attempts = len(subdirs)

    if total_attempts == 0:
        print("No subdirectories found to analyze.")
        print("-" * 40)
        return

    successful_attacks = 0
    total_queries_for_successful = 0

    for subdir in subdirs:
        # Check for the existence of a successful attack image
        success_files = glob.glob(os.path.join(subdir, 'successful_attack*.png'))
        if success_files:
            successful_attacks += 1

            # Find the CSV file in the successful attack directory
            csv_files = glob.glob(os.path.join(subdir, '*.csv'))
            if csv_files:
                # Assume the first found CSV is the correct one
                csv_path = csv_files[0]
                try:
                    # Try parsing with pandas first
                    df = pd.read_csv(csv_path, comment='#', engine='python')
                    if not df.empty and 'total_queries' in df.columns:
                        last_row_queries = df['total_queries'].iloc[-1]
                        total_queries_for_successful += last_row_queries
                    else:
                        print(f"Warning: CSV file {csv_path} is empty or missing 'total_queries' column.")
                except Exception:
                    # If pandas fails, try to parse manually
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
                            last_row_queries = int(parts[1])
                            total_queries_for_successful += last_row_queries
                        else:
                            print(f"Warning: Manual parsing failed, CSV file {csv_path} is empty or missing data rows.")
                    except Exception as e2:
                        print(f"Failed to process CSV file {csv_path} (both pandas and manual parsing failed): {e2}")
            else:
                print(f"Warning: CSV file not found in successful attack directory: {subdir}")

    # Calculate metrics
    success_rate = (successful_attacks / total_attempts) * 100 if total_attempts > 0 else 0
    avg_queries = total_queries_for_successful / successful_attacks if successful_attacks > 0 else 0

    # Print results
    print(f"Total attack attempts: {total_attempts}")
    print(f"Successful attacks: {successful_attacks}")
    print(f"Attack success rate: {success_rate:.2f}%")
    if successful_attacks > 0:
        print(f"Average queries for successful attacks: {avg_queries:.2f}")
    print("-" * 40)


if __name__ == "__main__":
    # List of directories to analyze
    dirs_to_analyze = [
        "face_analysis_cli_attack_results",
    ]

    # If command line arguments are provided, use them as the directories to analyze
    if len(sys.argv) > 1:
        dirs_to_analyze = sys.argv[1:]

    for d in dirs_to_analyze:
        analyze_directory(d) 