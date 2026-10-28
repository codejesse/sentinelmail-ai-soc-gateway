import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Model config imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, precision_score, f1_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# pip install shap lime tensorflow
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

if __name__ == "__main__":
    # Task 1: Binary Phishing Detection (Stage 1). Using Dataset A, build a model that flags a URL
    # as phishing or legitimate. This model will be used in Task 3's pipeline.
    # • Analyse the class balance and correlated/noisy features
    # • Encode and scale features as needed; address class imbalance
    # • Identify the most predictive features
    # • Compare at least 3 traditional ML models and one deep-learning baseline
    # • Evaluate with accuracy, precision, recall, and F1; discuss the precision/recall trade-off for
    # an SOC context
    #Load dataset

    #Load Datasets (A: stage 1 training data, B: End-End test data)
    dataset_a = pd.read_csv('./dataset/Dataset A.csv')
    dataset_c = pd.read_csv('./dataset/Dataset C.csv')

    print(f"Original Dataset A shape: {dataset_a.shape}")
    print(f"summary of dataset: {dataset_a.describe()}")

    # Prevent Data Leakage (check repeating entries by their ID and update leaked_rows list, and finally print out leaked rows):
    is_leaked = dataset_a['ID'].isin(dataset_c['ID'])

    leaked_rows = dataset_a[is_leaked]

    print(f"Total overlapping rows found in both Dsets A and c: {len(leaked_rows)}")
    print(leaked_rows)

    # Create a clean dataframe from dataset A with all leakages dropped
    # To avoid feature leakage we don't feed the ID or row number into the ML
    dataset_a_clean = dataset_a[is_leaked == False].copy()
    print(f"original Dataset A rows: {len(dataset_a)}")
    print(f"Leaked rows removed: {len(leaked_rows)}")
    print(f"Clean Dataset A rows: {len(dataset_a_clean)}")

    # dropping the 'ID' column from the clean dataframe
    #i did this so the model can't be tricked where exploiters populate the ID column with the actual phising url and the model
    # misses that and it's used to derail the models training
    dataset_a_clean = dataset_a_clean.drop(columns=['ID'])

    # printing all column names feature and target columns are visible
    print(dataset_a_clean.columns)

    # classification analysis/class balancing
    # counting the exact number of Benign and Malicious rows
    class_counts = dataset_a_clean['label'].value_counts()
    print("Exact counts per class:")
    print(class_counts)

    # calculating the percentages of each class
    total_rows = len(dataset_a_clean)
    class_percentages = (class_counts / total_rows) * 100
    print(f"Percentage per class: {class_percentages}")

    # Visualization of classification
    sns.countplot(data=dataset_a_clean, x='label')
    plt.title("Class Balance: Benign Vs Malicious")
    plt.show()

    # Checking for noisy/useless Features
    #  Count how many missing values are in each column
    missing_counts = dataset_a_clean.isnull().sum()
    print("Missing values in each column:")
    print(missing_counts)

    dataset_a_clean = dataset_a_clean.dropna()
    print("Remaining rows after dropping missing values:", len(dataset_a_clean))

    #drop constant columns (zero variance: unique values <= 1)
    # we drop these constants because if there's no non-linearity between values of the column the variance is 0 and our model works with variance
    # this would break our script if divided by that 0 variance
    constant_cols = [col for col in dataset_a_clean.columns if dataset_a_clean[col].nunique() <= 1]
    dataset_a_clean = dataset_a_clean.drop(columns=constant_cols)

    #checking for correlation between features
    # correlation filtering (> 0.90)
    # we want to check columns with correlation, multicorrelation so we don't overload our models training
    # we keep one and drop the other column, we measure this by a threshold, if a column goes close to +1.0 or -1.0
    # if two features have a correlation higher than 0.90 they are twins and keeping both conufses our linear algorithms.
    # in our case for dataset a we want to check numerical correlation between features
    # we drop the label column because we are doing a feature by feature comparison not a feature-to-label check
    numeric_features = dataset_a_clean.select_dtypes(include=[np.number]).drop(columns=['label'], errors='ignore')
    corr_matrix = numeric_features.corr().abs()

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > 0.90)]
    print(f"Dropped, multi-correlation columns: {to_drop_corr}")

    dataset_a_clean = dataset_a_clean.drop(columns=to_drop_corr)

    # Encoding, scaling, Splitting remaining columns in dataset for testing and training and handling class imbalance
    # Separate features (x) and target label (y) explicitly by column name
    x = dataset_a_clean.drop(columns=['label'])

    # One-hot encode any categorical text columns in x (if any exist)
    x = pd.get_dummies(x, drop_first=True)

    # Explicit Binary level encoding (1 = malicious, 0 = benign)
    # Safely handles whether Dataset A stores labels as strings ("malicious"/"attack") or integers (0/1)
    # Explicit Binary label encoding (1 = malicious, 0 = benign)
    y = (dataset_a_clean["label"].str.lower() == "malicious").astype(int).to_numpy()
    print(f"Target encoded successfully. Class balance (0 vs 1): {np.bincount(y)}")

    print(f"Target encoded successfully. Class balance (0 vs 1): {np.bincount(y)}")

    # Splitting the dataset for a 80/20 Train-Test Split (preserves exact class ratio)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=123, stratify=y
    )

    # cross-validation 5-fold k-fold classification scheme
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # spot checking models and running k-fold classification on them
    models = {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=123),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=123),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=123),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=123)
    }

    results = {}
    print("\n--- 5-Fold Cross-Validation Spot-Check (Training Set) ---")
    for name, clf in models.items():
        t0 = time.perf_counter()
        pipe = make_pipeline(StandardScaler(), clf)
        s = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="f1")
        results[name] = s
        print(f"{name:<13} CV F1 = {s.mean():.3f} +/- {s.std():.3f} ({time.perf_counter() - t0:.1f}s)")

    # Boxplot comparing CV F1 scores across folds
    plt.boxplot(list(results.values()))
    plt.xticks(range(1, len(results) + 1), results.keys())
    plt.ylabel("F1 Score"); plt.title("Stage 1 Spot-Check: 5-Fold CV"); plt.show()

    # Select the best traditional model from CV and evaluate ONCE on the held-out test set
    best_name = max(results, key=lambda k: results[k].mean())
    best_classical = make_pipeline(StandardScaler(), models[best_name]).fit(x_train, y_train)
    y_pred = best_classical.predict(x_test)

    print(f"Best Traditional Model on Test Set: {best_name}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"], digits=3))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["benign", "malicious"], cmap="Blues")
    plt.title(f"Confusion Matrix ({best_name})")
    plt.show()

    # Identifying the most predictive features (SHAP Global + LIME local)
    scaler_xai = StandardScaler()
    x_train_scaled_df = pd.DataFrame(scaler_xai.fit_transform(x_train), columns=x.columns)
    x_test_scaled_df = pd.DataFrame(scaler_xai.transform(x_test), columns=x.columns)

    # SHAP: SHapley Additive exPlanations
    # Global audit using TreeExplainer on Random Forest
    # SHAP distributes the payout (output/prediction) of the model across the features to see how much each feature is
    # affecting the global model's output/prediction
    rf_shap = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=123)
    rf_shap.fit(x_train_scaled_df, y_train)

    x_test_sample = x_test_scaled_df.iloc[:200]
    explainer = shap.TreeExplainer(rf_shap)
    shap_values = explainer(x_test_sample)
    shap_mal = shap_values[:, :, 1]

    plt.title("Global SHAP: Top 10 most predictive URL features")
    shap.plots.bar(shap_mal, max_display=10)

    plt.title("SHAP Beeswarm: Feature Impact on malicious URL prediction")
    shap.plots.beeswarm(shap_mal, max_display=10)

    # LIME: Local Interpretable Model-agnostic Explanations.
    # Doesn't explain the full impact of features in a model on predictions; instead picks one and explains it and presents it
    # in a human readable manner
    lime_explainer = LimeTabularExplainer(
        training_data=x_train.to_numpy(),
        feature_names=list(x.columns),
        class_names=["benign", "malicious"],
        mode="classification",
        random_state=123
    )

    alert_index = 0
    lime_exp = lime_explainer.explain_instance(
        data_row=x_test.iloc[alert_index].to_numpy(),
        predict_fn=best_classical.predict_proba,
        num_features=6
    )

    print(f"\nLIME Local Per-Alert Explanation (Test Row #{alert_index})")

    for rule, weight in lime_exp.as_list():
        print(f"  {rule:<32} -> contribution: {weight:+.3f}")

    fig = lime_exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation (Alert #{alert_index})")
    plt.tight_layout()
    plt.show()

    # Comparison and deep learning model baseline
    tf.random.set_seed(123)
    np.random.seed(123)

    dl_model = keras.Sequential([
        keras.Input(shape=(x_train_scaled_df.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    dl_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(name="recall")]
    )

    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = dl_model.fit(
        x_train_scaled_df, y_train,
        validation_split=0.2, epochs=20, batch_size=64, callbacks=[stop], verbose=2
    )

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Deep learning, learning curves")
    plt.legend(); plt.grid(True); plt.show()

    p_dl = dl_model.predict(x_test_scaled_df, verbose=0).ravel()
    y_pred_dl = (p_dl >= 0.5).astype(int)

    print("\nDeep learning baseline results on Test set")
    print(confusion_matrix(y_test, y_pred_dl))
    print(classification_report(y_test, y_pred_dl, target_names=["benign", "malicious"], digits=3))

    comparison_df = pd.DataFrame([
        {
            "Model": f"Best classical ({best_name})",
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred)
        },
        {
            "Model": "Deep Learning (Keras MLP)",
            "Accuracy": accuracy_score(y_test, y_pred_dl),
            "Precision": precision_score(y_test, y_pred_dl),
            "Recall": recall_score(y_test, y_pred_dl),
            "F1-Score": f1_score(y_test, y_pred_dl)
        }
    ]).round(3)

    print("\nFinal Test Set comparison")
    print(comparison_df.to_string(index=False))

    # Save winning Stage 1 pipeline and feature names for Task 3
    joblib.dump(best_classical, "stage1_pipeline.pkl")
    joblib.dump(list(x.columns), "stage1_features.pkl")
    print("\nSaved 'stage1_pipeline.pkl' and 'stage1_features.pkl' for Task 3!")