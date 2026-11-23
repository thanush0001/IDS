"""
Data preprocessing pipeline module for IoT Intrusion Detection System.
Handles data cleaning, label encoding, feature selection, scaling, and train-test splitting.
Prevents data leakage by fitting transformers solely on the training partition.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from config import Config


class DataPreprocessor:
    """End-to-end preprocessing pipeline for network traffic data."""

    # Candidate label column names found in datasets like CICIoT2023, BoT-IoT, UNSW-NB15, etc.
    CANDIDATE_LABELS = [
        'label', 'Label', 'attack_type', 'Attack_type', 'Attack_Type',
        'class', 'Class', 'category', 'Category', 'target', 'Target'
    ]

    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = []
        self.feature_names = []
        self.label_column = None
        self.label_classes = []
        self.original_shape = (0, 0)
        self.cleaned_shape = (0, 0)
        self.preprocessing_log = []
        self.missing_imputed = 0
        self.duplicates_removed = 0

    def load_data(self, filepath):
        """Load CSV data into DataFrame, detect label column, and record metadata."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

        # Read CSV
        self.df = pd.read_csv(filepath)
        self.original_shape = self.df.shape
        self.preprocessing_log.append(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

        # Auto-detect label column
        self.label_column = self.detect_label_column(self.df)
        self.preprocessing_log.append(f"Detected label column: '{self.label_column}'")
        return self.df

    @classmethod
    def detect_label_column(cls, df):
        """Find the label column from candidate names or default to the last column."""
        for candidate in cls.CANDIDATE_LABELS:
            if candidate in df.columns:
                return candidate
        # Fallback to last column
        return df.columns[-1]

    def get_info(self):
        """Return comprehensive dataset profile."""
        if self.df is None:
            return {}

        missing_counts = self.df.isnull().sum()
        total_missing = int(missing_counts.sum())
        total_duplicates = int(self.df.duplicated().sum())

        col_details = []
        for col in self.df.columns:
            col_details.append({
                'name': col,
                'dtype': str(self.df[col].dtype),
                'missing': int(self.df[col].isnull().sum()),
                'unique': int(self.df[col].nunique())
            })

        class_dist = {}
        if self.label_column and self.label_column in self.df.columns:
            class_dist = self.df[self.label_column].value_counts().to_dict()

        return {
            'shape': self.df.shape,
            'num_rows': self.df.shape[0],
            'num_cols': self.df.shape[1],
            'columns': list(self.df.columns),
            'total_missing': total_missing,
            'total_duplicates': total_duplicates,
            'label_column': self.label_column,
            'class_distribution': class_dist,
            'column_details': col_details
        }

    def clean_data(self):
        """Replace infs, impute missing values, and remove constant columns."""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data first.")

        # Replace +/- inf with NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Count missing before imputation
        self.missing_imputed = int(self.df.isnull().sum().sum())

        # Drop columns with > 50% missing values
        thresh = len(self.df) * 0.5
        self.df.dropna(thresh=thresh, axis=1, inplace=True)

        # Separate feature columns and label
        feature_cols = [c for c in self.df.columns if c != self.label_column]

        # Impute missing values: median for numeric, mode for categorical
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(0.0 if pd.isna(median_val) else median_val)
            else:
                # Categorical / string feature
                mode_vals = self.df[col].mode()
                fill_val = mode_vals[0] if len(mode_vals) > 0 else 'Unknown'
                self.df[col] = self.df[col].fillna(fill_val)

        # Drop rows where label itself is NaN
        if self.label_column in self.df.columns:
            self.df.dropna(subset=[self.label_column], inplace=True)

        self.preprocessing_log.append(f"Cleaned data: handled {self.missing_imputed} missing/infinite values.")
        return self.df

    def remove_duplicates(self):
        """Remove duplicate rows from dataset."""
        if self.df is None:
            raise ValueError("No data loaded.")
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.duplicates_removed = initial_len - len(self.df)
        self.preprocessing_log.append(f"Removed {self.duplicates_removed} duplicate records.")
        return self.df

    def encode_labels(self):
        """Encode target labels to integer codes and preserve mapping."""
        if self.df is None or self.label_column not in self.df.columns:
            raise ValueError("Target label column missing or data not loaded.")

        # Ensure labels are string representation
        self.df[self.label_column] = self.df[self.label_column].astype(str)
        self.label_classes = sorted(list(self.df[self.label_column].unique()))
        self.label_encoder.fit(self.df[self.label_column])

        y_encoded = self.label_encoder.transform(self.df[self.label_column])
        self.preprocessing_log.append(
            f"Encoded {len(self.label_classes)} classes: {', '.join(map(str, self.label_classes[:5]))}..."
        )
        return y_encoded

    def select_features(self, X_df, y_arr, max_features=25):
        """
        Select top informative features:
        1. Remove near-zero variance features
        2. Rank remaining features using ANOVA F-value (f_classif)
        """
        # Ensure only numeric features are passed
        numeric_df = X_df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            # Try converting object columns to numeric where possible
            numeric_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Variance threshold filter (remove constant or nearly constant features)
        var_filter = VarianceThreshold(threshold=1e-4)
        try:
            var_filter.fit(numeric_df)
            retained_cols = numeric_df.columns[var_filter.get_support()].tolist()
            if not retained_cols:
                retained_cols = list(numeric_df.columns)
        except Exception:
            retained_cols = list(numeric_df.columns)

        X_filtered = numeric_df[retained_cols]

        # Top K feature selection using SelectKBest
        k = min(max_features, len(retained_cols))
        selector = SelectKBest(score_func=f_classif, k=k)
        try:
            selector.fit(X_filtered, y_arr)
            selected_mask = selector.get_support()
            self.selected_features = [col for col, sel in zip(retained_cols, selected_mask) if sel]
        except Exception:
            # Fallback if f_classif has zero variance or single class
            self.selected_features = retained_cols[:k]

        self.feature_names = self.selected_features
        self.preprocessing_log.append(f"Selected {len(self.selected_features)} top features for training.")
        return X_filtered[self.selected_features]

    def split_data(self, X_data, y_data, test_size=0.2, random_state=42):
        """
        Perform stratified train-test split to preserve attack ratios.
        Optional SMOTE application only on the training set to prevent data leakage.
        """
        # Determine whether stratification is possible (need >= 2 samples per class)
        unique_classes, counts = np.unique(y_data, return_counts=True)
        can_stratify = all(c >= 2 for c in counts)

        stratify_param = y_data if can_stratify else None

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_data, y_data,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )

        # Scale features: fit ONLY on X_train, transform X_test
        self.X_train = self.scaler.fit_transform(X_tr)
        self.X_test = self.scaler.transform(X_te)
        self.y_train = np.array(y_tr)
        self.y_test = np.array(y_te)

        # Try applying SMOTE to balance minority attacks in training partition only
        try:
            from imblearn.over_sampling import SMOTE
            # SMOTE requires min samples per class > k_neighbors (default 5)
            min_count = min(counts)
            if min_count > 5:
                smote = SMOTE(random_state=random_state, k_neighbors=min(5, min_count - 1))
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                self.preprocessing_log.append("Applied SMOTE balancing to training set.")
        except Exception as e:
            self.preprocessing_log.append(f"SMOTE skipped ({str(e)}). Using standard split.")

        self.preprocessing_log.append(
            f"Train partition: {len(self.X_train)} samples, Test partition: {len(self.X_test)} samples."
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def run_full_pipeline(self, filepath, test_size=0.2, max_features=25):
        """Execute complete end-to-end preprocessing pipeline on dataset."""
        self.load_data(filepath)
        self.clean_data()
        self.remove_duplicates()
        y_encoded = self.encode_labels()

        feature_cols = [c for c in self.df.columns if c != self.label_column]
        X_df = self.df[feature_cols]

        X_selected = self.select_features(X_df, y_encoded, max_features=max_features)
        self.split_data(X_selected, y_encoded, test_size=test_size, random_state=Config.RANDOM_STATE)

        self.cleaned_shape = (len(self.df), len(self.selected_features))

        return self.get_summary()

    def get_summary(self):
        """Return structured summary of preprocessing outputs."""
        return {
            'original_rows': self.original_shape[0],
            'original_cols': self.original_shape[1],
            'cleaned_rows': len(self.df) if self.df is not None else 0,
            'features_count': len(self.selected_features),
            'selected_features': self.selected_features,
            'missing_imputed': self.missing_imputed,
            'duplicates_removed': self.duplicates_removed,
            'label_column': self.label_column,
            'classes': list(map(str, self.label_classes)),
            'num_classes': len(self.label_classes),
            'train_size': len(self.X_train) if self.X_train is not None else 0,
            'test_size': len(self.X_test) if self.X_test is not None else 0,
            'scaling_method': 'StandardScaler',
            'log': self.preprocessing_log
        }

    def save_pipeline(self, filepath):
        """Serialize fitted pipeline transformers for real-time inference."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        pipeline_bundle = {
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'label_classes': self.label_classes,
            'label_column': self.label_column
        }
        joblib.dump(pipeline_bundle, filepath)
        return filepath

    @staticmethod
    def load_pipeline(filepath):
        """Load fitted pipeline components from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        return joblib.load(filepath)

    def transform_new_data(self, df_new):
        """Transform incoming batch of network traffic matching fitted feature space."""
        df_copy = df_new.copy()
        # Drop label column if present
        if self.label_column and self.label_column in df_copy.columns:
            df_copy = df_copy.drop(columns=[self.label_column])

        # Ensure all selected features exist
        for col in self.selected_features:
            if col not in df_copy.columns:
                df_copy[col] = 0.0

        # Select only the features expected by the model in exact order
        X_subset = df_copy[self.selected_features].apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Scale using pre-fitted scaler
        return self.scaler.transform(X_subset)
