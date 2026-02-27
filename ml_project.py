import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

import mlflow
import os


# 
# Custom Transformers
# 

class HandCentering(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_array = X.values if hasattr(X, 'values') else X
        landmarks = data_array.reshape(-1, 21, 3)
        wrist = landmarks[:, 0:1, :]
        centered = landmarks - wrist
        return centered.reshape(-1, 63)


class HandNormalization(BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_array = X.values if hasattr(X, 'values') else X
        landmarks = data_array.reshape(-1, 21, 3)
        middle = landmarks[:, 12, :]
        magnitude = np.linalg.norm(middle, axis=1, keepdims=True).reshape(-1, 1, 1)
        normalized = landmarks / magnitude
        return normalized.reshape(-1, 63)


# 
# Data Loading
# 

def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")
    print(f"Label distribution:\n{data['label'].value_counts()}\n")
    return data


# 
# Visualization
# 

def visualize_label_distribution(data: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.countplot(y=data['label'], color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Labels', fontsize=16, fontweight='bold')
    plt.xlabel('Number of samples', fontsize=12)
    plt.ylabel('Label', fontsize=12)
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/label_distribution.png')
    mlflow.log_artifact('artifacts/label_distribution.png')
    plt.show()
    plt.close()


def visualize_pca(features, labels: pd.Series, title: str = 'PCA Plot'):
    pca = sk.decomposition.PCA(n_components=2)
    reduced = pca.fit_transform(features)
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, alpha=0.6)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig(f'artifacts/{title.replace(" ", "_").lower()}.png')
    mlflow.log_artifact(f'artifacts/{title.replace(" ", "_").lower()}.png')
    plt.show()
    plt.close()


def visualize_keypoints(features, labels: pd.Series, title_prefix: str = 'Keypoint'):
    data_array = features if isinstance(features, np.ndarray) else features.values
    fig, axes = plt.subplots(3, 7, figsize=(60, 45))
    axes = axes.flatten()

    for i in range(21):
        x_idx = i * 3
        y_idx = i * 3 + 1
        sns.scatterplot(
            x=data_array[:, x_idx],
            y=data_array[:, y_idx],
            hue=labels,
            ax=axes[i],
            legend=False,
            alpha=0.6
        )
        axes[i].set_title(f'{title_prefix} {i + 1}', fontsize=12)
        axes[i].set_xlabel(f'X position for point {i + 1}', fontsize=9)
        axes[i].set_ylabel(f'Y position for point {i + 1}', fontsize=9)

    handles, vis_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, vis_labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    plt.tight_layout()
    os.makedirs('artifacts', exist_ok=True)
    safe_prefix = title_prefix.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f'artifacts/{safe_prefix}_plots.png')
    mlflow.log_artifact(f'artifacts/{safe_prefix}_plots.png')
    plt.show()
    plt.close()


# 
# Preprocessing
# 

def build_preprocessor() -> Pipeline:
    return Pipeline([
        ('hand_centering', HandCentering()),
        ('hand_normalization', HandNormalization())
    ])


def analyze_variance(features: pd.DataFrame):
    print("Variance Analysis:")
    for axis, offset, label in [('Z', 2, 'z'), ('Y', 1, 'y'), ('X', 0, 'x')]:
        print(f"\n--- {axis} columns ---")
        for i in range(21):
            col_idx = i * 3 + offset
            print(f"  Variance of {label}{i} = {np.var(features.iloc[:, col_idx]):.6f}")
    print("\nNote: Z columns have near-zero variance (depth info kept for camera distance).\n")


# 
# Training
# 

MODEL_PARAM_MAP = {
    'Support Vector Classifier': {
        'model': SVC(),
        'param_grid': {
            'C': [0.1, 1, 5],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 0.01]
        },
        'label_processing': False
    },
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'param_grid': {
            'C': [0.01, 0.1, 1, 5, 10, 20, 50],
            'solver': ['lbfgs', 'saga', 'liblinear'],
            'penalty': ['l2']
        },
        'label_processing': False
    },
    'Random Forest Classifier': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [50, 100, 150, 200, 250],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'label_processing': False
    },
    'XGBoost Classifier': {
        'model': xgb.XGBClassifier(eval_metric='mlogloss'),
        'param_grid': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        },
        'label_processing': True
    }
}


def training_process(
    features: pd.DataFrame,
    labels: pd.Series,
    model_param_map: dict,
    feature_processor: Pipeline = None,
    label_processor: LabelEncoder = None,
    cv: int = 3,
    n_iter: int = 15,
    random_state: int = 42,
    verbose_mode: int = 4
):
    """
    Train and evaluate multiple models with hyperparameter tuning.

    Args:
        features: Input feature data
        labels: Target labels
        model_param_map: Dictionary mapping model names to model configs
        feature_processor: Feature preprocessing pipeline
        label_processor: Label encoder (default: LabelEncoder)
        cv: Cross-validation folds
        n_iter: Number of random search iterations
        random_state: Random seed for reproducibility
        verbose_mode: Verbosity level for RandomizedSearchCV
    """
    if feature_processor is None:
        feature_processor = build_preprocessor()
    if label_processor is None:
        label_processor = LabelEncoder()

    print(f"Dataset shape: {features.shape}")

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels
    )

    features_train_processed = feature_processor.fit_transform(features_train)
    features_test_processed = feature_processor.transform(features_test)

    labels_train_encoded = label_processor.fit_transform(labels_train)
    labels_test_encoded = label_processor.transform(labels_test)

    joblib.dump(label_processor, 'processors/label_encoder.joblib')
    joblib.dump(feature_processor, 'processors/feature_processor.joblib')

    for model_name, model_info in model_param_map.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            print(f"\n{'=' * 50}")
            print(f"Training: {model_name}")
            print(f"{'=' * 50}")

            cur_train_labels = labels_train_encoded if model_info['label_processing'] else labels_train
            cur_test_labels  = labels_test_encoded  if model_info['label_processing'] else labels_test


            random_search = RandomizedSearchCV(
                model_info['model'],
                param_distributions=model_info['param_grid'],
                n_iter=n_iter,
                cv=cv,
                verbose=verbose_mode,
                random_state=random_state
            )

            random_search.fit(features_train_processed, cur_train_labels)
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best CV score:   {random_search.best_score_:.4f}")

            best_model = random_search.best_estimator_
            predictions = best_model.predict(features_test_processed)

            signature = mlflow.models.infer_signature(features_train_processed, best_model.predict(features_train_processed))
            mlflow.sklearn.log_model(sk_model=best_model, artifact_path=f"{model_name}_model", signature=signature)
            mlflow.log_params(random_search.best_params_)
            mlflow.log_metric('best_cv_score', random_search.best_score_)
            mlflow.log_artifact('processors/label_encoder.joblib', artifact_path='processors')
            mlflow.log_artifact('processors/feature_processor.joblib', artifact_path='processors')

            accuracy  = accuracy_score(cur_test_labels, predictions)
            f1        = f1_score(cur_test_labels, predictions, average='weighted')
            recall    = recall_score(cur_test_labels, predictions, average='weighted')
            precision = precision_score(cur_test_labels, predictions, average='weighted')

            mlflow.log_metrics({
                'accuracy': accuracy,
                'f1_score': f1,
                'recall': recall,
                'precision': precision
            })

            print("--- Test Dataset Evaluation ---")
            print(f"  Accuracy  = {accuracy:.4f}")
            print(f"  F1        = {f1:.4f}")
            print(f"  Recall    = {recall:.4f}")
            print(f"  Precision = {precision:.4f}")

            joblib.dump(best_model, f'models/{model_name}_model.joblib')
            print(f"Model saved: models/{model_name}_model.joblib")


# 
# Main
# 

def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hand-gesture-classification")

    with mlflow.start_run(run_name="full_pipeline"):
        _run_pipeline()


def _run_pipeline():
    # Load
    data = load_data('data/hand_landmarks_data.csv')
    mlflow.log_input(mlflow.data.from_pandas(data), context='raw_data')

    features = data.drop('label', axis=1)
    labels = data['label']
    mlflow.log_param('dataset_rows', data.shape[0])
    mlflow.log_param('dataset_cols', data.shape[1])

    # Visualize raw data
    visualize_label_distribution(data)
    visualize_pca(features, labels, title='PCA Plot Raw Features')
    visualize_keypoints(features.values, labels, title_prefix='Keypoint Raw')

    # Variance analysis
    analyze_variance(features)

    # Preprocess and visualize
    preprocessor = build_preprocessor()
    new_features = preprocessor.fit_transform(features)

    visualize_keypoints(new_features, labels, title_prefix='Keypoint Preprocessed')
    visualize_pca(new_features, labels, title='PCA Plot After Preprocessing')

    # Train
    training_process(
        features=features,
        labels=labels,
        model_param_map=MODEL_PARAM_MAP,
        feature_processor=build_preprocessor(),
        label_processor=None,
        cv=3,
        n_iter=15,
        random_state=42,
        verbose_mode=4
    )


if __name__ == '__main__':
    main()
