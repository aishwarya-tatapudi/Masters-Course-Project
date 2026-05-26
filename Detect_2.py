import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================
# 1. LOAD DATA
# =========================

# deepset
d_train = pd.read_parquet("train-00000-of-00001-9564e8b05b4757ab.parquet")
d_test  = pd.read_parquet("test-00000-of-00001-701d16158af87368.parquet")

# JasperLS
j_train = pd.read_parquet("train-00000-of-00001-9564e8b05b4757ab (1).parquet")
j_test  = pd.read_parquet("test-00000-of-00001-701d16158af87368 (1).parquet")


# =========================
# 2. CLEAN FUNCTION
# =========================

def clean_df(df):
    # unify text column
    if "text" in df.columns:
        df["text_clean"] = df["text"]
    elif "prompt" in df.columns:
        df["text_clean"] = df["prompt"]
    else:
        raise Exception("No text column found")

    # unify label column
    if "label" in df.columns:
        df["label_clean"] = df["label"]
    elif "is_injection" in df.columns:
        df["label_clean"] = df["is_injection"]
    else:
        raise Exception("No label column found")

    # normalize labels to 0/1
    df["label_clean"] = df["label_clean"].apply(
        lambda x: 1 if str(x).lower() in ["1","true","yes","injection"] else 0
    )

    return df[["text_clean", "label_clean"]]


# Apply cleaning
d_train = clean_df(d_train)
d_test  = clean_df(d_test)
j_train = clean_df(j_train)
j_test  = clean_df(j_test)


# =========================
# 3. COMBINE DATA
# =========================

combined = pd.concat([d_train, d_test, j_train, j_test], ignore_index=True)

print("Total samples:", len(combined))
print(combined["label_clean"].value_counts())


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X = combined["text_clean"]
y = combined["label_clean"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# =========================
# 5. TF-IDF + SVM
# =========================

print("\nTraining TF-IDF + SVM...")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = SVC(kernel="rbf", class_weight="balanced")
model.fit(X_train_vec, y_train)

print("Training complete.")


# =========================
# 6. EVALUATION
# =========================

preds = model.predict(X_test_vec)

tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

acc  = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp)
rec  = tp / (tp + fn)
f1   = 2 * (prec * rec) / (prec + rec)

fpr = fp / (fp + tn)
fnr = fn / (tp + fn)

print("\n==============================")
print("TF-IDF + SVM RESULTS")
print("==============================")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall (TPR): {rec:.4f}")
print(f"FPR: {fpr:.4f}")
print(f"FNR: {fnr:.4f}")
print(f"F1 Score: {f1:.4f}")
print("==============================")