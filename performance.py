import pandas as pd
import os
import numpy as np




def normalize(x):
    """
    A numerically stable normalization function that scales the input values
    into the range [0, 3].

    If all input values are identical, the function returns an array of zeros.
    """
    x = np.array(x, dtype=float)
    finite_mask = np.isfinite(x)
    if not np.any(finite_mask):
        return np.zeros_like(x)

    x_valid = x[finite_mask]
    min_val, max_val = np.min(x_valid), np.max(x_valid)
    if np.isclose(min_val, max_val):
        return np.zeros_like(x)

    x_scaled = 3 * (x - min_val) / (max_val - min_val)
    x_scaled[~finite_mask] = 0
    return x_scaled


# -------------------------------
# MAE computation
# -------------------------------
def MAE_computation(folder):
    pt_files = [f[:-5] for f in os.listdir(folder) if f.endswith(".xlsx")][:-1]
    result_all = {}
    count = 0
    
    for pt_file in pt_files:
        result_all[pt_file] = {}
        model_types = ['Model_LLMSAN']
    
        # ---------------------
        # Load label information
        # ---------------------
        label_df = pd.read_excel(
            'Dataset_with_label/' + f'{pt_file}.xlsx'
        )[['id', 'label', 'time']]
        label_dict = dict(zip(label_df['id'], label_df['label']))
        time_dict = dict(zip(label_df['id'], label_df['time']))
    
        # ---------------------
        # Iterate over each model and process results separately
        # ---------------------
        print(pt_file,model_types)
        for model_type in model_types:
            df = pd.read_excel(
                f'result/{pt_file}/{model_type}.xlsx'
            )[['predicted_id', 'prediction_error']]
            df = df.dropna()
    
            # 1️⃣ Extract ids
            ids, preds = [], []
            for _, row in df.iterrows():
                pid = row['predicted_id']
                idx = int(str(pid).replace("pred_", "").strip())
                ids.append(idx)
                preds.append(row['prediction_error'])
    
            # 2️⃣ Sort by id
            sorted_pairs = sorted(zip(ids, preds), key=lambda x: x[0])
            sorted_ids = [p[0] for p in sorted_pairs]
            sorted_preds = [p[1] for p in sorted_pairs]
    
            # 3️⃣ Match labels and timestamps (based on model-specific id set)
            sorted_labels = [label_dict.get(i, np.nan) for i in sorted_ids]
            sorted_times = [time_dict.get(i, np.nan) for i in sorted_ids]
    
            # 4️⃣ Store results in a nested dictionary
            result_all[pt_file][model_type] = {
                'id': sorted_ids,
                'pred': sorted_preds,
                'label': sorted_labels,
                'time': sorted_times
            }
    
        count += 1
        
    mae_results = []  # Store MAE results for each pt_file
    
    for pt_file, models_dict in result_all.items():
        mae_entry = {'pt_file': pt_file}
    
        for model_type, data_dict in models_dict.items():
            preds = np.array(data_dict['pred'][:-1], dtype=float)
            labels = np.array(data_dict['label'][:-1], dtype=float)
    
            # Skip if predictions are empty or entirely NaN
            if len(preds) == 0 or np.all(np.isnan(preds)):
                mae_entry[model_type] = np.nan
                continue
    
            # Normalize prediction values
            preds_norm = normalize(preds)
    
            # Align labels by removing NaN values
            valid_mask = np.isfinite(preds_norm) & np.isfinite(labels)
    
            if np.sum(valid_mask) == 0:
                mae_entry[model_type] = np.nan
                continue
    
            mae = np.mean(np.abs(preds_norm[valid_mask] - labels[valid_mask]))
            mae_entry[model_type] = mae
    
        mae_results.append(mae_entry)
    
    
    # -------------------------------
    # Save results
    # -------------------------------
    df_mae = pd.DataFrame(mae_results)
    df_mae.to_excel("mae_results_normalized.xlsx", index=False)
