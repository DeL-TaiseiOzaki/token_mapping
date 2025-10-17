#!/bin/bash

# config1.jsonを生成
cat > config1.json << 'EOF'
{
  "model_name": "Qwen/Qwen3-235B-A22B-Thinking-2507",
  "max_length": 16384,
  "datasets": [
    {
      "name": "DongfuJiang/hle_text_only",
      "split": "test[:2000]",
      "columns": ["question", "answer" ]
    },
    {
      "name": "llm-2025-sahara/OlymMATH-en-translatedbyQwen3-235B-instruct",
      "split": "train[:1000]",
      "columns": ["problem", "answer"]
    },
    {
      "name": "toloka/u-math",
      "split": "test[:1000]",
      "columns": ["problem_statement","golden_answer"]
    },
    {
      "name": "KbsdJames/Omni-MATH",
      "split": "test[:1000]",
      "columns": ["problem", "solution", "answer"]
    },
    {
      "name": "llm-2025-sahara/LiveMathBench-en",
      "split": "test[:1000]",
      "columns": ["question", "answer"]
    }
  ]
}
EOF

# config2.jsonを生成
cat > config2.json << 'EOF'
{
  "model_name": "Qwen/Qwen3-235B-A22B-Thinking-2507",
  "max_length": 16384,
  "datasets": [
    {
      "name": "DongfuJiang/hle_text_only",
      "split": "test[:2000]",
      "columns": ["question", "answer" ]
    },
    {
<<<<<<< HEAD
      "name": "DeL-TaiseiOzaki/hle-failed-problems-byQwen3-32b",
      "split": "train[:1500]",
=======
      "name": "llm-2025-sahara/OlymMATH-en-with-reasoning",
      "split": "train[:1000]",
      "columns": ["question","answer"]
    },
    {
      "name": "llm-2025-sahara/Omni-MATH-imo-imc-with-reasoning2",
      "split": "train[:1000]",
      "columns": ["question","answer"]
    },
    {
      "name": "llm-2025-sahara/u-math-with-reasoning",
      "split": "train[:1000]",
      "columns": ["question","answer"]
    },
    {
      "name": "llm-2025-sahara/LiveMathBench-en-with-reasoning",
      "split": "train[:1000]",
>>>>>>> 86ca1544d3b14b152a908e058b01b4a6cb3dfaa3
      "columns": ["question","answer"]
    }
  ]
}
EOF

echo "=== Config files created ==="
echo ""

# 1回目：trainモードでconfig1を実行（モデルを学習）
echo "=== Step 1: Training with config1.json ==="
python main.py train config1.json
echo ""

# 2回目：transformモードでconfig2を実行（学習済みモデルを使用）
echo "=== Step 2: Transforming with config2.json ==="
python main.py transform config2.json
echo ""

# 3回目：距離を比較
echo "=== Step 3: Comparing distances ==="
python compare.py
echo ""

echo "=== Complete! ==="
echo "Generated files:"
echo "  - visualization_train.png          (config1 with trained models)"
echo "  - visualization_transform.png      (config2 with same models)"
echo "  - distances_train_config1.json     (distance matrix for config1)"
echo "  - distances_transform_config2.json (distance matrix for config2)"
echo "  - comparison_PCA_1000d.png         (comparison charts)"
echo "  - comparison_PCA_2d.png"
echo "  - comparison_UMAP_2d.png"
echo ""
echo "The comparison shows how adding reasoning affects dataset similarity!"