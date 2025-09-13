import os
import json
import pandas as pd

def analyze_results(groundtruth_dir, results_dir):
    gt_files = sorted([f for f in os.listdir(groundtruth_dir) if f.endswith('.json') and f != 'analyze.py'])
    
    compilers = ['gcc', 'clang']
    optimization = 'O0'
    
    all_stats = []

    for gt_file in gt_files:
        if not gt_file.endswith('.json'):
            continue

        base_name = gt_file.replace('.json', '')
        parts = base_name.split('_')
        
        opt_info = parts[-1]
        compiler_info = parts[-2]
        framework = parts[-3]
        model = '_'.join(parts[:-3])
        
        # This loop is to process both compilers for each file pattern
        for compiler in compilers:
            # Construct corresponding hook file path
            hook_filename = f"{model}_{framework}_{compiler}_{optimization}_hook_config.json"
            hook_filepath = os.path.join(results_dir, compiler, optimization, hook_filename)
            
            groundtruth_filename = f"{model}_{framework}_{compiler}_{optimization}.json"
            groundtruth_filepath = os.path.join(groundtruth_dir, groundtruth_filename)

            if not os.path.exists(groundtruth_filepath):
                continue
            
            with open(groundtruth_filepath, 'r') as f:
                gt_data = json.load(f)
            
            gt_addresses = {item['address'] for item in gt_data}

            hook_addresses = set()
            hook_data_list = []
            if os.path.exists(hook_filepath):
                with open(hook_filepath, 'r') as f:
                    try:
                        hook_data_list = json.load(f)
                        hook_addresses = {item['address'] for item in hook_data_list}
                    except json.JSONDecodeError:
                        pass

            true_positives = len(gt_addresses.intersection(hook_addresses))
            
            # --- MODIFIED FALSE POSITIVES CALCULATION ---
            fp_addresses = hook_addresses.difference(gt_addresses)
            false_positives = 0
            for addr in fp_addresses:
                hook_item = next((item for item in hook_data_list if item['address'] == addr), None)
                if hook_item:
                    registers = hook_item.get('registers', [])
                    is_excluded = False
                    for reg_info in registers:
                        register_value = reg_info.get('register', '')
                        if 'StackDirect' in register_value or 'UniquePcode' in register_value:
                            is_excluded = True
                            break
                    if not is_excluded:
                        false_positives += 1
            # --- END OF MODIFICATION ---

            # --- MODIFIED FALSE NEGATIVES CALCULATION ---
            # Exclude 'cbz' and 'cbnz' instructions from false negatives
            missing_addresses = gt_addresses.difference(hook_addresses)
            false_negatives = 0
            for item in gt_data:
                if item['address'] in missing_addresses:
                    branch_instruction = item.get('branch_instruction', '').lower()
                    instruction = item.get('instruction', '').lower()
                    if 'cbz' not in branch_instruction and 'cbz' not in instruction and 'cbnz' not in branch_instruction and 'cbnz' not in instruction:
                        false_negatives += 1
            # --- END OF MODIFICATION ---

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Find the taint analysis result file to get total hits
            taint_result_filename = f"{model}_{framework}_{compiler}_{optimization}_taint_analysis_results.json"
            taint_result_filepath = os.path.join(results_dir, taint_result_filename)
            
            total_hits = 0
            if os.path.exists(taint_result_filepath):
                 with open(taint_result_filepath, 'r') as f:
                    try:
                        taint_data = json.load(f)
                        total_hits = len(taint_data)
                    except json.JSONDecodeError:
                        pass

            stats = {
                "Model": model,
                "Framework": framework,
                "Compiler": compiler,
                "Opt": optimization,
                "TP": true_positives,
                "FP": false_positives,
                "FN": false_negatives,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Ground Truth Count": len(gt_addresses),
                "Hook Config Count": len(hook_addresses),
                "Total Taint Hits": total_hits,
            }
            all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    
    df_display = df.copy()
    df_display["Precision"] = df_display["Precision"].map('{:.2%}'.format)
    df_display["Recall"] = df_display["Recall"].map('{:.2%}'.format)
    df_display["F1-Score"] = df_display["F1-Score"].map('{:.2%}'.format)

    display_columns = ["Model", "Framework", "Compiler", "Opt", "TP", "FP", "FN", 
                       "Precision", "Recall", "F1-Score",
                       "Ground Truth Count", "Hook Config Count", "Total Taint Hits"]
    df_display = df_display[display_columns]

    df_display = df_display.drop_duplicates()
    
    return df, df_display

if __name__ == "__main__":
    groundtruth_directory = '.'
    results_directory = '../results'
    
    analysis_df, display_df = analyze_results(groundtruth_directory, results_directory)
    
    print("Analysis Results:")
    print(display_df.to_string())

    # --- Framework Summary ---
    print("\n--- Framework Statistics ---")
    framework_summary = analysis_df.groupby('Framework').agg({
        'TP': 'sum',
        'FP': 'sum',
        'FN': 'sum'
    }).reset_index()

    def calculate_metrics(row):
        tp, fp, fn = row['TP'], row['FP'], row['FN']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return pd.Series([f"{precision:.2%}", f"{recall:.2%}", f"{f1:.2%}"], index=['Precision', 'Recall', 'F1-Score'])

    framework_metrics = framework_summary.apply(calculate_metrics, axis=1)
    framework_summary = pd.concat([framework_summary, framework_metrics], axis=1)
    
    


    # --- TFLite False Positives Details ---
    print("\n--- TFLite False Positives Details ---")
    tflite_df = analysis_df[analysis_df['Framework'] == 'tflite']
    
    all_tflite_fps = []

    for index, row in tflite_df.iterrows():
        model = row['Model']
        framework = row['Framework']
        compiler = row['Compiler']
        optimization = row['Opt']

        hook_filename = f"{model}_{framework}_{compiler}_{optimization}_hook_config.json"
        hook_filepath = os.path.join(results_directory, compiler, optimization, hook_filename)
        
        groundtruth_filename = f"{model}_{framework}_{compiler}_{optimization}.json"
        groundtruth_filepath = os.path.join(groundtruth_directory, groundtruth_filename)

        if not os.path.exists(groundtruth_filepath) or not os.path.exists(hook_filepath):
            continue

        with open(groundtruth_filepath, 'r') as f:
            gt_data = json.load(f)
        gt_addresses = {item['address'] for item in gt_data}

        with open(hook_filepath, 'r') as f:
            hook_data = json.load(f)
        
        hook_info = {item['address']: item for item in hook_data}
        hook_addresses = set(hook_info.keys())

        fp_addresses = hook_addresses.difference(gt_addresses)

        for addr in sorted(list(fp_addresses)):
            fp_details = hook_info.get(addr, {})

            # --- ADDED FP FILTERING ---
            registers = fp_details.get('registers', [])
            is_excluded = False
            for reg_info in registers:
                register_value = reg_info.get('register', '')
                if 'StackDirect' in register_value or 'UniquePcode' in register_value:
                    is_excluded = True
                    break
            if is_excluded:
                continue
            # --- END OF FILTERING ---

            all_tflite_fps.append({
                "Model": model,
                "Compiler": compiler,
                "Address": addr,
                "Instruction": fp_details.get('instruction', 'N/A'),
                "Branch Instruction": fp_details.get('original_branch_instruction', 'N/A')
            })

    if all_tflite_fps:
        fp_df = pd.DataFrame(all_tflite_fps)
        print(fp_df.to_string())
    else:
        print("No TFLite false positives found.")


    # --- NCNN False Negatives Details ---
    print("\n--- NCNN False Negatives Details ---")
    ncnn_df = analysis_df[analysis_df['Framework'] == 'ncnn']
    
    all_ncnn_fns = []

    for index, row in ncnn_df.iterrows():
        model = row['Model']
        framework = row['Framework']
        compiler = row['Compiler']
        optimization = row['Opt']

        hook_filename = f"{model}_{framework}_{compiler}_{optimization}_hook_config.json"
        hook_filepath = os.path.join(results_directory, compiler, optimization, hook_filename)
        
        groundtruth_filename = f"{model}_{framework}_{compiler}_{optimization}.json"
        groundtruth_filepath = os.path.join(groundtruth_directory, groundtruth_filename)

        if not os.path.exists(groundtruth_filepath):
            continue

        with open(groundtruth_filepath, 'r') as f:
            gt_data = json.load(f)
        gt_addresses = {item['address'] for item in gt_data}
        gt_info = {item['address']: item for item in gt_data}

        hook_addresses = set()
        if os.path.exists(hook_filepath):
            with open(hook_filepath, 'r') as f:
                try:
                    hook_data = json.load(f)
                    hook_addresses = {item['address'] for item in hook_data}
                except json.JSONDecodeError:
                    pass

        fn_addresses = gt_addresses.difference(hook_addresses)

        for addr in sorted(list(fn_addresses)):
            fn_details = gt_info.get(addr, {})
            branch_instruction = fn_details.get('branch_instruction', '').lower()
            instruction = fn_details.get('instruction', '').lower()
            
            if 'cbz' in branch_instruction or 'cbz' in instruction or 'cbnz' in branch_instruction or 'cbnz' in instruction:
                continue

            all_ncnn_fns.append({
                "Model": model,
                "Compiler": compiler,
                "Address": addr,
                "Instruction": fn_details.get('instruction', 'N/A'),
                "Branch Instruction": fn_details.get('branch_instruction', 'N/A')
            })

    if all_ncnn_fns:
        fn_df = pd.DataFrame(all_ncnn_fns)
        print(fn_df.to_string())
    else:
        print("No NCNN false negatives found (after filtering).")


    # --- ONNX Runtime False Negatives Details ---
    print("\n--- ONNX Runtime False Negatives Details ---")
    onnx_df = analysis_df[analysis_df['Framework'] == 'onnxruntime']
    
    all_onnx_fns = []

    for index, row in onnx_df.iterrows():
        model = row['Model']
        framework = row['Framework']
        compiler = row['Compiler']
        optimization = row['Opt']

        # Construct file paths
        hook_filename = f"{model}_{framework}_{compiler}_{optimization}_hook_config.json"
        hook_filepath = os.path.join(results_directory, compiler, optimization, hook_filename)
        
        groundtruth_filename = f"{model}_{framework}_{compiler}_{optimization}.json"
        groundtruth_filepath = os.path.join(groundtruth_directory, groundtruth_filename)

        if not os.path.exists(groundtruth_filepath):
            continue

        with open(groundtruth_filepath, 'r') as f:
            gt_data = json.load(f)
        gt_addresses = {item['address'] for item in gt_data}
        gt_info = {item['address']: item for item in gt_data}

        hook_addresses = set()
        if os.path.exists(hook_filepath):
            with open(hook_filepath, 'r') as f:
                try:
                    hook_data = json.load(f)
                    hook_addresses = {item['address'] for item in hook_data}
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

        fn_addresses = gt_addresses.difference(hook_addresses)

        for addr in sorted(list(fn_addresses)):
            fn_details = gt_info.get(addr, {})
            all_onnx_fns.append({
                "Model": model,
                "Compiler": compiler,
                "Address": addr,
                "Instruction": fn_details.get('instruction', 'N/A'),
                "Branch Instruction": fn_details.get('branch_instruction', 'N/A')
            })

    if all_onnx_fns:
        fn_df = pd.DataFrame(all_onnx_fns)
        print(fn_df.to_string())
    else:
        print("No ONNX Runtime false negatives found.")


    # --- Overall statistics ---
    print("\n--- Overall Statistics ---")
    total_tp = analysis_df['TP'].sum()
    total_fp = analysis_df['FP'].sum()
    total_fn = analysis_df['FN'].sum()

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"Overall Precision: {overall_precision:.2%}")
    print(f"Overall Recall: {overall_recall:.2%}")
    print(f"Overall F1-Score: {overall_f1:.2%}")

    print(f"\nTotal Ground Truth Branches: {analysis_df['Ground Truth Count'].sum()}")
    print(f"Total Hooked Branches: {analysis_df['Hook Config Count'].sum()}")
    print(f"Total Taint Analysis Hits: {analysis_df['Total Taint Hits'].sum()}")
