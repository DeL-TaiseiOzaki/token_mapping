# main.py
import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from umap import UMAP
import warnings
import pickle
import sys
import os
import pandas as pd
warnings.filterwarnings('ignore', category=UserWarning, module='umap')

# コマンドライン引数でモードを指定
mode = sys.argv[1] if len(sys.argv) > 1 else "train"
config_file = sys.argv[2] if len(sys.argv) > 2 else "config.json"

print(f"Mode: {mode}, Config: {config_file}")

# 設定ファイル読み込み
with open(config_file, "r") as f:
    config = json.load(f)

datasets_config = config["datasets"]
model_name = config["model_name"]
max_length = config.get("max_length", 512)

# 1. データ読み込みとテキスト作成
texts, labels = [], []
for i, ds_config in enumerate(datasets_config):
    ds = load_dataset(ds_config["name"], split=ds_config["split"])
    for row in ds:
        text = " ".join([str(row[col]) for col in ds_config["columns"]])
        texts.append(text)
        labels.append(i)

print(f"Total samples: {len(texts)}")
labels = np.array(labels)

# 2. トークナイズ
tokenizer = AutoTokenizer.from_pretrained(model_name)
embeddings = []
for text in texts:
    tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
    vec = np.bincount(tokens, minlength=tokenizer.vocab_size)
    embeddings.append(vec)

embeddings = np.array(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# PCAで事前に次元削減（UMAPの高速化）
print("Reducing dimensions with PCA...")

if mode == "train":
    # 学習モード：モデルを学習して保存
    pca_pre = PCA(n_components=1000)
    embeddings_reduced = pca_pre.fit_transform(embeddings)
    
    # 2次元変換用のモデルも学習
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(embeddings_reduced)
    
    umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    X_umap = umap_2d.fit_transform(embeddings_reduced)
    
    # モデルを保存
    with open("models.pkl", "wb") as f:
        pickle.dump({
            "pca_pre": pca_pre,
            "pca_2d": pca_2d,
            "umap_2d": umap_2d
        }, f)
    print("Models saved to models.pkl")
    
else:
    # 変換モード：保存されたモデルを読み込んで適用
    if not os.path.exists("models.pkl"):
        print("Error: models.pkl not found. Run with 'train' mode first.")
        sys.exit(1)
    
    with open("models.pkl", "rb") as f:
        models = pickle.load(f)
    
    pca_pre = models["pca_pre"]
    pca_2d = models["pca_2d"]
    umap_2d = models["umap_2d"]
    
    # 保存されたモデルで変換
    embeddings_reduced = pca_pre.transform(embeddings)
    X_pca = pca_2d.transform(embeddings_reduced)
    X_umap = umap_2d.transform(embeddings_reduced)
    print("Models loaded and applied")

print(f"Reduced embedding shape: {embeddings_reduced.shape}")

# ====== データセット間の距離計算 ======
print("\n" + "="*80)
print("Computing distances between datasets...")
print("="*80)

# データセット名の取得（短縮版）
dataset_names = []
for ds_config in datasets_config:
    name = ds_config["name"].split("/")[-1]
    # 長すぎる名前を短縮
    if len(name) > 30:
        name = name[:27] + "..."
    dataset_names.append(name)

# 各データセットの重心を計算
def compute_centroids(embeddings, labels, space_name="Original"):
    """各データセットの重心を計算"""
    n_datasets = len(np.unique(labels))
    centroids = []
    
    for i in range(n_datasets):
        mask = labels == i
        centroid = embeddings[mask].mean(axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # ペアワイズ距離を計算
    euclid_dist = euclidean_distances(centroids)
    cosine_dist = cosine_distances(centroids)
    
    print(f"\n[{space_name}] Euclidean distances:")
    df_euclid = pd.DataFrame(euclid_dist, 
                             index=dataset_names, 
                             columns=dataset_names)
    print(df_euclid.round(2))
    
    print(f"\n[{space_name}] Cosine distances:")
    df_cosine = pd.DataFrame(cosine_dist, 
                             index=dataset_names, 
                             columns=dataset_names)
    print(df_cosine.round(4))
    
    # 基準データセット（hle_text_only）からの距離を表示
    if n_datasets > 1:
        print(f"\n[{space_name}] Distances from '{dataset_names[0]}':")
        distances_from_base = []
        for i in range(1, n_datasets):
            distances_from_base.append({
                'Dataset': dataset_names[i],
                'Euclidean': euclid_dist[0, i],
                'Cosine': cosine_dist[0, i]
            })
        df_from_base = pd.DataFrame(distances_from_base)
        print(df_from_base.to_string(index=False))
    
    return {
        "centroids": centroids,
        "euclidean": euclid_dist,
        "cosine": cosine_dist,
        "df_euclidean": df_euclid,
        "df_cosine": df_cosine
    }

# 3つの空間で距離を計算
distances = {
    "PCA_1000d": compute_centroids(embeddings_reduced, labels, "PCA 1000D"),
    "PCA_2d": compute_centroids(X_pca, labels, "PCA 2D"),
    "UMAP_2d": compute_centroids(X_umap, labels, "UMAP 2D")
}

# 結果をJSONファイルに保存
output_data = {
    "config_file": config_file,
    "mode": mode,
    "dataset_names": dataset_names,
    "full_dataset_names": [ds["name"] for ds in datasets_config],
    "distances": {}
}

for space_name, dist_data in distances.items():
    output_data["distances"][space_name] = {
        "euclidean": dist_data["euclidean"].tolist(),
        "cosine": dist_data["cosine"].tolist()
    }

# ファイル名からconfig番号を抽出
config_num = config_file.replace('config', '').replace('.json', '')
output_json = f"distances_{mode}_config{config_num}.json"
with open(output_json, "w") as f:
    json.dump(output_data, f, indent=2)
print(f"\n✓ Distances saved to: {output_json}")

# 3. 可視化（重心も表示）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PCA
for i, ds_config in enumerate(datasets_config):
    mask = labels == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   label=dataset_names[i], alpha=0.6, s=20)
    # 重心を大きく表示
    centroid = distances["PCA_2d"]["centroids"][i]
    axes[0].scatter(centroid[0], centroid[1], 
                   marker='*', s=500, edgecolors='black', linewidths=2, zorder=100)

axes[0].set_title(f"PCA Visualization ({mode} mode, {config_file})")
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)

# UMAP
for i, ds_config in enumerate(datasets_config):
    mask = labels == i
    axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], 
                   label=dataset_names[i], alpha=0.6, s=20)
    # 重心を大きく表示
    centroid = distances["UMAP_2d"]["centroids"][i]
    axes[1].scatter(centroid[0], centroid[1], 
                   marker='*', s=500, edgecolors='black', linewidths=2, zorder=100)

axes[1].set_title(f"UMAP Visualization ({mode} mode, {config_file})")
axes[1].legend(loc='best', fontsize=8)
axes[1].grid(True, alpha=0.3)
<<<<<<< HEAD
axes[1].set_xlim([20, 40])
=======
>>>>>>> 86ca1544d3b14b152a908e058b01b4a6cb3dfaa3

plt.tight_layout()
output_file = f"visualization_{mode}.png"
plt.savefig(output_file, dpi=150)
print(f"✓ Saved: {output_file}")
plt.close()

print("\n" + "="*80)
print("Done!")
print("="*80)