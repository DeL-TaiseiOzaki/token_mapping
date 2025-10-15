# compare_distances.py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_distances(mode, config_num):
    """距離データを読み込み"""
    filename = f"distances_{mode}_config{config_num}.json"
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found!")
        return None

def compare_distances():
    """config1(train)とconfig2(transform)の距離を比較"""
    
    # データ読み込み
    data1 = load_distances("train", "1")
    data2 = load_distances("transform", "2")
    
    if data1 is None or data2 is None:
        return
    
    dataset_names1 = data1["dataset_names"]
    dataset_names2 = data2["dataset_names"]
    
    print("="*100)
    print("COMPARING CONFIG1 (train) vs CONFIG2 (transform)")
    print("="*100)
    print(f"\nConfig1 datasets: {dataset_names1}")
    print(f"Config2 datasets: {dataset_names2}")
    print("\nNote: Both use the SAME model space (trained on config1)")
    print("="*100)
    
    # 各空間での比較
    for space in ["PCA_1000d", "PCA_2d", "UMAP_2d"]:
        print(f"\n{'='*100}")
        print(f"[{space}]")
        print('='*100)
        
        euclid1 = np.array(data1["distances"][space]["euclidean"])
        euclid2 = np.array(data2["distances"][space]["euclidean"])
        
        cosine1 = np.array(data1["distances"][space]["cosine"])
        cosine2 = np.array(data2["distances"][space]["cosine"])
        
        # 基準データセット（hle_text_only）からの距離を比較
        print(f"\n--- Distances from '{dataset_names1[0]}' (base dataset) ---")
        
        comparison_data = []
        for i in range(1, min(len(dataset_names1), len(dataset_names2))):
            comparison_data.append({
                'Dataset': i,
                'Config1_Name': dataset_names1[i],
                'Config1_Euclid': euclid1[0, i],
                'Config1_Cosine': cosine1[0, i],
                'Config2_Name': dataset_names2[i],
                'Config2_Euclid': euclid2[0, i],
                'Config2_Cosine': cosine2[0, i],
                'Δ_Euclid': euclid2[0, i] - euclid1[0, i],
                'Δ_Cosine': cosine2[0, i] - cosine1[0, i]
            })
        
        df_comp = pd.DataFrame(comparison_data)
        print("\nEuclidean Distance Comparison:")
        print(df_comp[['Dataset', 'Config1_Name', 'Config1_Euclid', 
                      'Config2_Name', 'Config2_Euclid', 'Δ_Euclid']].to_string(index=False))
        
        print("\nCosine Distance Comparison:")
        print(df_comp[['Dataset', 'Config1_Name', 'Config1_Cosine', 
                      'Config2_Name', 'Config2_Cosine', 'Δ_Cosine']].to_string(index=False))
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # データセット2-5の距離を棒グラフで比較
        datasets_idx = list(range(1, min(len(dataset_names1), len(dataset_names2))))
        x_labels = [f"DS{i}" for i in datasets_idx]
        
        # Euclidean距離の比較
        ax = axes[0, 0]
        x = np.arange(len(datasets_idx))
        width = 0.35
        ax.bar(x - width/2, euclid1[0, 1:len(datasets_idx)+1], width, label='Config1 (original)', alpha=0.8)
        ax.bar(x + width/2, euclid2[0, 1:len(datasets_idx)+1], width, label='Config2 (with reasoning)', alpha=0.8)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Euclidean Distance')
        ax.set_title(f'{space}: Euclidean Distance from Base')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Cosine距離の比較
        ax = axes[0, 1]
        ax.bar(x - width/2, cosine1[0, 1:len(datasets_idx)+1], width, label='Config1 (original)', alpha=0.8)
        ax.bar(x + width/2, cosine2[0, 1:len(datasets_idx)+1], width, label='Config2 (with reasoning)', alpha=0.8)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Cosine Distance')
        ax.set_title(f'{space}: Cosine Distance from Base')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 差分の棒グラフ
        ax = axes[1, 0]
        diff_euclid = euclid2[0, 1:len(datasets_idx)+1] - euclid1[0, 1:len(datasets_idx)+1]
        colors = ['green' if d < 0 else 'red' for d in diff_euclid]
        ax.bar(x, diff_euclid, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Δ Euclidean Distance')
        ax.set_title(f'{space}: Change in Euclidean Distance\n(Config2 - Config1, negative = closer)')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        diff_cosine = cosine2[0, 1:len(datasets_idx)+1] - cosine1[0, 1:len(datasets_idx)+1]
        colors = ['green' if d < 0 else 'red' for d in diff_cosine]
        ax.bar(x, diff_cosine, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Δ Cosine Distance')
        ax.set_title(f'{space}: Change in Cosine Distance\n(Config2 - Config1, negative = closer)')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{space} - Distance Comparison (Config1 vs Config2)', fontsize=16, y=0.995)
        plt.tight_layout()
        
        output_file = f"comparison_{space}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved: {output_file}")
        plt.close()

if __name__ == "__main__":
    compare_distances()