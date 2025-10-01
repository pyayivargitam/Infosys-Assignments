# train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Load the Iris dataset
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    feature_names = list(X.columns)
    target_names = iris.target_names.tolist()

    # 2. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build pipeline (scaler + random forest)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Train the model
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=target_names))

    # 6. Save model + metadata
    model_dict = {
        'model': pipeline,
        'feature_names': feature_names,
        'target_names': target_names
    }
    joblib.dump(model_dict, "model.joblib")
    print("\n✅ Model trained and saved as model.joblib")

if __name__ == "__main__":
    main()
