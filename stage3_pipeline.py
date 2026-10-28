import os
import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Import the text preprocessor from task2
from stage2 import clean_and_lemmatise


def run_integrated_pipeline(csv_path: str):
    print(f"\n[+] Loading test dataset from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[!] Error: File not found at '{csv_path}'.")
        return

    df_c = pd.read_csv(csv_path)

    # 1. If Dataset C only has IDs and Labels, merge the URL features (from Dataset A)
    #    and the message Corpus (from Dataset B) using the 'ID' column!
    if "Corpus" not in df_c.columns:
        print("[+] Merging URL features (Dataset A) and message Corpus (Dataset B) onto Dataset C by 'ID'...")
        df_a = pd.read_csv("./dataset/Dataset A.csv").drop(columns=["label"], errors="ignore")
        df_b = pd.read_csv("./dataset/Dataset B.csv")[["ID", "Corpus"]]

        df_c = df_c.merge(df_a, on="ID", how="left")
        df_c = df_c.merge(df_b, on="ID", how="left")
        print(f"[+] Merged Dataset C shape: {df_c.shape}")

    # 2. Load saved Stage 1 and Stage 2 production artifacts
    print("[+] Loading Stage 1 and Stage 2 production artifacts...")
    stage1_pipe = joblib.load("stage1_pipeline.pkl")
    stage1_cols = joblib.load("stage1_features.pkl")
    stage2_pipe = joblib.load("stage2_pipeline.pkl")
    stage2_le = joblib.load("stage2_label_encoder.pkl")

    # --- STAGE 1: URL Binary Filtering ---
    print("[+] Executing Stage 1 (URL Phishing Detection)...")
    # Drop text and label columns before encoding URL features
    url_df = df_c.drop(columns=["ID", "Corpus", "Dataset A Label", "Dataset B Label", "label", "Labels"], errors="ignore")
    X_url = pd.get_dummies(url_df, drop_first=True)
    X_url = X_url.reindex(columns=stage1_cols, fill_value=0)

    stage1_preds = stage1_pipe.predict(X_url)             # 0 = Benign, 1 = Malicious
    stage1_probs = stage1_pipe.predict_proba(X_url)[:, 1] # Confidence of Malicious

    # --- STAGE 2: Conditional Escalation Gatekeeper ---
    # We keep verdicts in the exact original row order of Dataset C!
    final_verdicts = []
    end_to_end_stage2_preds = []

    for i, is_malicious in enumerate(stage1_preds):
        record_id = df_c["ID"].iloc[i]

        if is_malicious == 0:
            # Gatekeeper stops it here: marked Benign without running Stage 2
            final_verdicts.append({
                "ID": record_id,
                "Stage1_Flag": "Benign",
                "Stage1_Confidence": round(float(1 - stage1_probs[i]), 3),
                "Stage2_Attack_Type": "NOT-Malicious General Class"
            })
            end_to_end_stage2_preds.append("NOT-Malicious General Class")
        else:
            # Escalate to Stage 2: Clean the message text and predict the specific attack type!
            raw_text = str(df_c["Corpus"].iloc[i])
            clean_text = clean_and_lemmatise(raw_text)

            pred_idx = stage2_pipe.predict([clean_text])[0]
            attack_name = stage2_le.inverse_transform([pred_idx])[0]

            final_verdicts.append({
                "ID": record_id,
                "Stage1_Flag": "Malicious",
                "Stage1_Confidence": round(float(stage1_probs[i]), 3),
                "Stage2_Attack_Type": attack_name
            })
            end_to_end_stage2_preds.append(attack_name)

    results_df = pd.DataFrame(final_verdicts)

    print("\n--- Pipeline Verdicts Preview (First 15 Rows) ---")
    print(results_df.head(15).to_string(index=False))

    output_filename = "pipeline_verdict_report.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\n[+] Full end-to-end report saved to '{output_filename}'!")

    # =====================================================================
    # TASK 3 EVALUATION METRICS (End-to-End Detection & Identification Rates)
    # =====================================================================
    if "Dataset A Label" in df_c.columns and "Dataset B Label" in df_c.columns:
        print("\n==================================================")
        print("          TASK 3 PIPELINE PERFORMANCE METRICS     ")
        print("==================================================")

        # 1. Stage 1 Binary Ground Truth (Python 3.14 safe string check!)
        stage1_true = (df_c["Dataset A Label"].astype(str).str.lower() == "malicious").astype(int).to_numpy()

        det_acc = accuracy_score(stage1_true, stage1_preds)
        det_rec = recall_score(stage1_true, stage1_preds, zero_division=0)
        det_prec = precision_score(stage1_true, stage1_preds, zero_division=0)
        det_f1 = f1_score(stage1_true, stage1_preds, zero_division=0)

        print(f"1. End-to-End Detection Rate (Stage 1 Recall): {det_rec:.3f} ({det_rec:.1%})")
        print(f"   Stage 1 Binary Accuracy: {det_acc:.3f} | Precision: {det_prec:.3f} | F1: {det_f1:.3f}")

        # Confusion Matrix with explicit labels=[0, 1] so it is ALWAYS 2x2!
        cm = confusion_matrix(stage1_true, stage1_preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n2. Pipeline-Level False Positive / Negative Impact:")
        print(f"   - True Negatives  (Benign URLs stopped at Stage 1):       {tn}")
        print(f"   - False Positives (Benign URLs needlessly sent to Stg 2): {fp}")
        print(f"   - False Negatives (Malicious URLs missed at Stage 1!):    {fn}")
        print(f"   - True Positives  (Malicious URLs escalated to Stage 2):  {tp}")

        # 3. End-to-End Attack-Identification Rate (Stage 1 + Stage 2 Combined vs 'Dataset B Label')
        stage2_true = df_c["Dataset B Label"].astype(str).to_numpy()
        stage2_pred_arr = pd.Series(end_to_end_stage2_preds).astype(str).to_numpy()

        id_acc = accuracy_score(stage2_true, stage2_pred_arr)
        id_macro_f1 = f1_score(stage2_true, stage2_pred_arr, average="macro", zero_division=0)

        print(f"\n3. End-to-End Attack-Identification Rate (Overall Accuracy): {id_acc:.3f} ({id_acc:.1%})")
        print(f"   End-to-End Multi-Class Macro-F1:                          {id_macro_f1:.3f}")
        print("\n--- End-to-End Multi-Class Classification Report (Dataset C) ---")
        print(classification_report(stage2_true, stage2_pred_arr, digits=3, zero_division=0))

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SentinelMail Two-Stage Phishing Detection Pipeline.")
    #Update this line to /custom_live_test.csv so script doesnt crash or create your dataset folder and put in your test
    #csv files you can use the /shema_config.json to generate a new dataset for you to test the pipeline
    parser.add_argument("dataset", type=str, nargs="?", default="./custom_live_test.csv",
                        help="Path to the test CSV file (default: ./dataset/Dataset C.csv)")
    args = parser.parse_args()
    run_integrated_pipeline(args.dataset)