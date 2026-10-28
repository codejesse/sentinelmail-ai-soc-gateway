import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import time
import joblib

# For text cleaniing and lematizing
import nltk
from keras.src.ops import vectorize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#model encoding, training and dl
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#download gthe nltk dictionaries once
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Lemmatizing is simple taking the different variants of a word and bringing it to it's base called a lemma
# e.g running, ran, runs -> "run"
def clean_and_lemmatise(text: str) -> str:
    # lowercasing string and removing punctuations and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())

    # Tokenise split string into list of words
    tokens = text.split()

    clean_tokens = [
        lemmatizer.lemmatize(lemmatizer.lemmatize(word), pos='v')
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(clean_tokens)

# Helper function returning our TF-IDF vectoriser
# ngram_range=(1, 2) captures both single words ("account") and 2-word phrases ("account suspended")!
def tfidf_vec():
    return TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))


if __name__ == "__main__":
    # loading datasets
    dataset_b = pd.read_csv('./dataset/Dataset B.csv')
    dataset_c = pd.read_csv('./dataset/Dataset C.csv')

    print(f"Original Dataset B shape: {dataset_b.shape}")

    # Checking for data leakage betwwen dataset B and C and removing it by creating a clean copy of DSET B without leakages
    is_leaked_b = dataset_b['ID'].isin(dataset_c['ID'])
    leaked_rows_b = dataset_b[is_leaked_b]
    print(f"Overlapping rows found in both Datasets B and C: {len(leaked_rows_b)}")


    dataset_b_clean = dataset_b[is_leaked_b == False].copy()
    dataset_b_clean = dataset_b_clean.drop(columns=['ID'])

    # Removing/dropping rows where there's missing corpus/message and Labels
    dataset_b_clean = dataset_b_clean.dropna(subset=['Corpus', 'Labels']).reset_index(drop=True)
    print(f"Clean Dataset B rows: {len(dataset_b_clean)}")

    # inspect the 6 social-engineering class sizes
    class_counts_b = dataset_b_clean['Labels'].value_counts()
    class_percentages_b = (class_counts_b / len(dataset_b_clean)) * 100

    print("Class counts: \n", class_counts_b)
    print("Class percentages (%): \n", class_counts_b.round(2))

    plt.figure(figsize=(9, 5))
    sns.countplot(data=dataset_b_clean, y='Labels', order=class_counts_b.index)
    plt.title("Distribution of social engineeering Attack categories")
    plt.tight_layout()
    plt.show()

    #Apply preprocessing to creat a clean text column
    dataset_b_clean['Clean_Corpus'] = dataset_b_clean['Corpus'].apply(clean_and_lemmatise)

    # Example for 3 rows in corpus column with text preprocessed (cleaned, lemmatized, and tokenized)
    print("Sample before vs afetr Text preprocessing")
    for i in range(3):
        print(f"RAW: [{dataset_b_clean['Labels'].iloc[i]}]: {dataset_b_clean['Corpus'].iloc[i][:70]}...")
        print(f"CLEAN [{dataset_b_clean['Labels'].iloc[i]}]: [{dataset_b_clean['Clean_Corpus'].iloc[i][:70]}]...\n")

    # encode the 6 text labels which is the classification labels for training
    # Encode the 6 text labels into integers (0 to 5) - label encoding not one shot because we have a lot of labels and
    # relationship matching with one-shot won't work here label encoding would work where we assign a unique integer to a label
    label_encode = LabelEncoder()
    y_b = label_encode.fit_transform(dataset_b_clean['Labels'])
    class_names_b = list(label_encode.classes_)
    print("Label Mapping:", dict(zip(class_names_b, range(len(class_names_b)))))

    # Stratified 80.20 Train-Test Split before fitting DTM
    text_train, text_test, y_train_b, y_test_b = train_test_split(
        dataset_b_clean['Clean_Corpus'], y_b, test_size=0.20, random_state=123, stratify=y_b
    )

    cv_b = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # Building the DTM (document term matrix) to check wording frequency
    # so we vectorize and then create the DTM
    # using TF-IDF (Term frequency: how often a word appears in a speciifc document)
    # IDF (Inverse Document Frequency): this is reduces the weight of common words accross a corpus and increases the words of rare words
    # in out case want the rare words captured to be the ones that match the classifications
    vec_dtm = CountVectorizer(min_df=2)
    dtm_train = vec_dtm.fit_transform(text_train)
    vocab_b = vec_dtm.get_feature_names_out()

    density = dtm_train.nnz / (dtm_train.shape[0] * dtm_train.shape[1])
    print(f"Train DTM shape: {dtm_train.shape} | Non-zero density: {density:.2%}")

    print("Top 8 DTM keywords per social engineering category")
    for class_index, class_label in enumerate(class_names_b):
        # sum word counts accross all training messages belonging to this specific class
        class_freq = np.asarray(dtm_train[y_train_b == class_index].sum(axis=0)).ravel()

        # [::-1] ensures we take the HIGHEST counts, not alphabetical 'a' words!
        top_word_indices = np.argsort(class_freq)[::-1][:8]
        top_words = vocab_b[top_word_indices]
        print(f" {class_label:<22}: {list(top_words)}")

    reps = {
        "counts": CountVectorizer(min_df=2, ngram_range=(1, 2)),
        "binary": CountVectorizer(min_df=2, binary=True, ngram_range=(1, 2)),
        "tf-idf": TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 2))
    }

    print("Comparing DTM Representations")
    for name, v in reps.items():
        s = cross_val_score(make_pipeline(v, MultinomialNB()), text_train, y_train_b, cv=cv_b, scoring="f1_macro")
        print(f" {name:<8} Macro-F1 = {s.mean():.3f} +/- {s.std():.3f}")

    models_b = {
        "NaiveBayes": MultinomialNB(),
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=123),
        "LinearSVM": LinearSVC(class_weight="balanced", random_state=123),
        "RandomForest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=123)
    }

    results_b = {}
    print("Classical Models spot-check (5 fold cv Macro-F1")
    for name, clf in models_b.items():
        t0 = time.perf_counter()
        s = cross_val_score(make_pipeline(tfidf_vec(), clf), text_train, y_train_b, cv=cv_b, scoring="f1_macro")
        results_b[name] = s
        print(f" {name: <13} CV Macro-F1 = {s.mean():.3f} +/- {s.std():.3f} ({time.perf_counter() - t0:.1f}s)")

    plt.boxplot(list(results_b.values()))
    plt.xticks(range(1, len(results_b) + 1), results_b.keys())
    plt.ylabel("Macro-F1 Score"); plt.title("Stage 2 Spot-Check: 5-fold CV")

    # The best classical model once on Held-out Test set
    best_name_b = max(results_b, key=lambda k: results_b[k].mean())
    best_stage2 = make_pipeline(tfidf_vec(), models_b[best_name_b]).fit(text_train, y_train_b)
    y_pred_b = best_stage2.predict(text_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test_b, y_pred_b, display_labels=class_names_b, cmap="Blues", xticks_rotation=45, ax=ax
    )
    plt.title(f"Stage 2 Confusion matrix ({best_name_b})")
    plt.tight_layout()
    plt.show()

    # Inspect top predictive words per class using Logistic Regression Coefficients
    lr_explain = make_pipeline(tfidf_vec(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=122)).fit(text_train, y_train_b)
    lr_vocab = lr_explain[0].get_feature_names_out()

    print("Top 6 words Driving Each attack prediction (LogReg Coefficients")
    for index, label in enumerate(class_names_b):
        top_indices = np.argsort(lr_explain[-1].coef_[index])[-6:][::-1]
        print(f" push towards {label:<20}: {list(lr_vocab[top_indices])}")

    # Deep representation baseline comparison with tradtional models
    tf.random.set_seed(123)
    np.random.seed(123)

    MAX_TOKENS, SEQ_LEN = 10000, 150
    num_classes = len(class_names_b)

    vectorize = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
    vectorize.adapt(text_train.to_numpy().astype(str)) #Learn vocabulary from training text only

    seq_train = vectorize(text_train.to_numpy().astype(str)).numpy()
    seq_test = vectorize(text_test.to_numpy().astype(str)).numpy()

    # Compute balanced class weights for keras so minority attack types aren't ignored
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train_b), y=y_train_b)
    keras_class_weights = dict(enumerate(weights))

    dl_stage2 = keras.Sequential([
        keras.Input(shape=(SEQ_LEN,)),
        layers.Embedding(MAX_TOKENS, 64, mask_zero=True), #Learns a 64-dim vector per word
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax") # 6 class probability output
    ])

    dl_stage2.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    stop_b = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    history_b = dl_stage2.fit(
        seq_train, y_train_b,
        validation_split=0.2,
        epochs=25,
        batch_size=32,
        class_weight=keras_class_weights,
        callbacks=[stop_b],
        verbose=2
    )

    # Predict on test sequneces np.argmaz picks the class(0-5) with the highest probability
    p_dl_b = dl_stage2.predict(seq_test, verbose=0)
    y_pred_dl_b = np.argmax(p_dl_b, axis=1)

    print("Deep Embedding Network Results on Test set")
    print(confusion_matrix(y_test_b, y_pred_dl_b))
    print(classification_report(y_test_b, y_pred_dl_b, target_names=class_names_b, digits=3))

    # Final Stage 2 Comparison
    print(f"\n=== STAGE 2 FINAL TEST SET COMPARISON ===")
    print(f"  {best_name_b:<15} Test Macro-F1 = {f1_score(y_test_b, y_pred_b, average='macro'):.3f} | Acc = {accuracy_score(y_test_b, y_pred_b):.3f}")
    print(f"  {'EmbeddingNet':<15} Test Macro-F1 = {f1_score(y_test_b, y_pred_dl_b, average='macro'):.3f} | Acc = {accuracy_score(y_test_b, y_pred_dl_b):.3f}")

    # Save winning Stage 2 pipeline and label encoder for Task 3!
    joblib.dump(best_stage2, "stage2_pipeline.pkl")
    joblib.dump(label_encode, "stage2_label_encoder.pkl")
    print("\nSaved 'stage2_pipeline.pkl' and 'stage2_label_encoder.pkl'!")