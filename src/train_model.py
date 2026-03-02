import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/products.csv"
MODEL_PATH = "model/product_category_model.pkl"


def main():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Product Title", "Category Label"])

    df["Category Label"] = df["Category Label"].astype(str).str.strip()
    df["Category Label"] = df["Category Label"].replace({
        "CPU": "CPUs",
        "Mobile Phone": "Mobile Phones",
        "fridge": "Fridges"
    })

    X = df["Product Title"].astype(str)
    y = df["Category Label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()