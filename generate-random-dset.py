# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#TO DO: Learn different ML algorithms and research why they would work to analyse and detect anomalies in DSET A and DSET B
# Task 1: Binary Phishing Detection (Stage 1). Using Dataset A, build a model that flags a URL
# as phishing or legitimate. This model will be used in Task 3's pipeline.
# • Analyse the class balance and correlated/noisy features
# • Encode and scale features as needed; address class imbalance
# • Identify the most predictive features
# • Compare at least 3 traditional ML models and one deep-learning baseline
# • Evaluate with accuracy, precision, recall, and F1; discuss the precision/recall trade-off for
# an SOC context
# 4
# Task 2: Attack-Type Classification (Stage 2). Using Dataset B, build a model that classifies the
# attack technique behind a flagged message. This model will also be used in Task 3's pipeline.
# • Clean and preprocess the message text (tokenise, remove stopwords, lemmatise)
# • Vectorise text (TF-IDF and/or embeddings)
# • Compare traditional models and a deep-representation baseline
# • Evaluate with macro-F1

#GENERATING CUSTOM TEST DATASET TO TEST PIPELINE
# for older versions of python run python generate-random-dset.py --rows [specify row] --mode random --chaos [specifiy chaos range]
# e.g python generate-random-dset.py --rows 50 --mode random --chaos 1.8
# for python 3+ -> python3 generate-random-dset.py --rows 50 --mode random --chaos 1.8
from __future__ import annotations

import os
import json
import uuid
import random
import argparse
import numpy as np
import pandas as pd

SCHEMA_PATH = "./schema_config.json"
OUTPUT_DIR = "./"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "custom_live_test.csv")


def mutate_corpus(cat_data: dict, chaos: float) -> str:
    """
    Synthesizes a unique email body by combining a category template with
    a randomized sample of that category's top TF-IDF n-grams.
    """
    templates = cat_data.get("corpus_templates", [""])
    ngrams = cat_data.get("top_ngrams", [])

    # Pick 1 or 2 base templates and shuffle clauses
    if len(templates) >= 2 and random.random() < 0.5:
        chosen = random.sample(templates, 2)
        base_text = f"{chosen[0]} {chosen[1]}"
    else:
        base_text = random.choice(templates)

    # Inject 3 to 6 category-specific TF-IDF phrases so every row is unique
    # while staying faithful to the class's learned linguistic signature
    if ngrams:
        k = min(len(ngrams), random.randint(3, max(4, int(5 * chaos))))
        sampled_phrases = random.sample(ngrams, k)
        jitter_sentence = f" Context indicators: {', '.join(sampled_phrases)}."
        base_text = base_text + jitter_sentence

    return base_text


def sample_numeric_feature(col_stat: dict, chaos: float) -> float | int:
    """Samples a numerical feature using chaos-scaled Gaussian + Uniform boundary noise."""
    mean = col_stat["mean"]
    std = col_stat["std"]
    min_val = col_stat["min"]
    max_val = col_stat["max"]

    if std == 0 or min_val == max_val:
        val = mean
    else:
        scaled_std = std * chaos
        gaussian_val = np.random.normal(loc=mean, scale=scaled_std)

        outlier_prob = min(0.25, 0.05 * chaos)
        if random.random() < outlier_prob:
            val = np.random.uniform(low=min_val, high=max_val)
        else:
            val = gaussian_val

        val = np.clip(val, min_val, max_val)

    return int(round(val)) if col_stat["is_int"] else round(float(val), 4)


def generate_synthetic_demo(
    num_rows: int = 6,
    mode: str = "balanced",
    chaos: float = 1.0,
    seed: int | None = None,
    schema_path: str = SCHEMA_PATH,
    output_path: str = OUTPUT_PATH,
) -> pd.DataFrame:
    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Missing '{schema_path}'. Run extract_schema_local.py first and commit schema_config.json."
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        entropy = int.from_bytes(os.urandom(4), "little")
        random.seed(entropy)
        np.random.seed(entropy)

    with open(schema_path, "r") as f:
        schema = json.load(f)

    feature_cols = schema["feature_columns"]
    categories = schema["categories"]
    cat_keys = list(categories.keys())

    if mode == "balanced":
        selected_labels = [cat_keys[i % len(cat_keys)] for i in range(num_rows)]
        random.shuffle(selected_labels)
    else:
        # Keep a minimum representation per class even in random mode so metrics don't show 0 support
        weights = np.random.dirichlet(np.ones(len(cat_keys)) * 2.0)
        selected_labels = np.random.choice(cat_keys, size=num_rows, p=weights).tolist()

    synthetic_rows = []

    for label_b in selected_labels:
        cat_data = categories[label_b]
        row = {
            "ID": f"SYN-{uuid.uuid4().hex[:8].upper()}",
            "Dataset A Label": cat_data["dataset_a_label"],
            "Dataset B Label": int(label_b) if str(label_b).isdigit() else label_b,
        }

        stats = cat_data["feature_stats"]
        for col in feature_cols:
            col_stat = stats[col]
            if col_stat.get("type") == "categorical":
                row[col] = random.choice(col_stat["values"])
            else:
                row[col] = sample_numeric_feature(col_stat, chaos=chaos)

        row["Corpus"] = mutate_corpus(cat_data, chaos=chaos)
        synthetic_rows.append(row)

    ordered_cols = ["ID", "Dataset A Label", "Dataset B Label"] + feature_cols + ["Corpus"]
    demo_sample = pd.DataFrame(synthetic_rows)[ordered_cols]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    demo_sample.to_csv(output_path, index=False)

    print(
        f"Created '{output_path}' | Rows: {len(demo_sample)} | Mode: {mode} | Chaos: {chaos}x"
    )
    print(demo_sample[["ID", "Dataset A Label", "Dataset B Label", "Corpus"]].head(10))
    return demo_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SOC triage datasets.")
    parser.add_argument("-n", "--rows", type=int, default=6, help="Number of rows to generate (default: 6)")
    parser.add_argument(
        "-m", "--mode", choices=["balanced", "random"], default="balanced",
        help="'balanced' (equal representation) or 'random' (unpredictable wave)"
    )
    parser.add_argument(
        "-c", "--chaos", type=float, default=1.0,
        help="Randomness/variance multiplier (e.g., 0.5 = tight, 1.0 = normal, 2.0 = wild outliers)"
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_PATH, help="Output CSV file path")

    args = parser.parse_args()
    generate_synthetic_demo(
        num_rows=args.rows,
        mode=args.mode,
        chaos=args.chaos,
        seed=args.seed,
        output_path=args.output,
    )