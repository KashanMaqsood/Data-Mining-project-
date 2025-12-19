import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =========================
# 1. Load dataset
# =========================
print("=" * 60)
print("WATER PUMP FAILURE PREDICTION - MODEL TRAINING")
print("=" * 60)

DATA_PATH = "data/pump_sensor_data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Make sure to place the CSV at this path.")

print("\n[1/9] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   ✓ Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# =========================
# 2. Data Preprocessing
# =========================
print("\n[2/9] Preprocessing data...")
df = df.drop(columns=["Unnamed: 0", "timestamp"], errors="ignore")

# Check for missing values
missing_count = df.isnull().sum().sum()
print(f"   ✓ Missing values: {missing_count:,} ({missing_count/df.size*100:.2f}%)")

# Outlier detection using IQR method (optional - for information)
print("\n   Analyzing data quality...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
outlier_info = []
for col in numeric_cols:
    if col not in ['target']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            outlier_info.append((col, outliers))
if outlier_info:
    print(f"   ⚠️  Outliers detected in {len(outlier_info)} features (will be handled by robust imputation)")
else:
    print(f"   ✓ No significant outliers detected")

# =========================
# 3. Encode target variable
# =========================
print("\n[3/9] Encoding target variable...")
status_mapping = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
df["target"] = df["machine_status"].map(status_mapping)

# Create label encoder for inverse transformation
label_encoder = LabelEncoder()
label_encoder.fit(df["machine_status"])

# =========================
# 4. Separate features & target
# =========================
print("\n[4/9] Separating features and target...")
X = df.drop(columns=["target", "machine_status"])
y = df["target"]

print("\nClass distribution:")
class_counts = y.value_counts().sort_index()
for idx, count in class_counts.items():
    status_name = ["NORMAL", "RECOVERING", "BROKEN"][idx]
    percentage = count / len(y) * 100
    print(f"   {idx} ({status_name}): {count:,} ({percentage:.2f}%)")

# =========================
# 5. Train-test split
# =========================
print("\n[5/9] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"   ✓ Training set: {X_train.shape[0]:,} samples")
print(f"   ✓ Test set: {X_test.shape[0]:,} samples")

# =========================
# 6. Feature Selection (Optional - can be enabled)
# =========================
print("\n[6/10] Feature selection and preprocessing...")
# Use all features initially, but we'll track importance
USE_FEATURE_SELECTION = False  # Set to True to enable feature selection

if USE_FEATURE_SELECTION:
    print("   Applying feature selection (SelectKBest)...")
    selector = SelectKBest(score_func=f_classif, k='all')  # Keep all for now
    # We'll do feature selection after initial training based on importance
else:
    print("   Using all features (feature selection based on importance will be applied)")

# =========================
# 7. Create preprocessing pipeline
# =========================
print("\n[7/10] Creating preprocessing pipeline...")
preprocessing_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Robust to outliers
    ("scaler", StandardScaler())
])
print("   ✓ Pipeline: Imputation (median) → Standardization")

# =========================
# 8. Hyperparameter tuning with cross-validation
# =========================
print("\n[8/10] Training model with optimized hyperparameters...")

# Initial training with good defaults
print("   Training Random Forest Classifier...")

# Create full pipeline
pipeline = Pipeline([
    ("preprocessing", preprocessing_pipeline),
    ("classifier", RandomForestClassifier(
        n_estimators=500,  # Increased for better performance
        max_depth=20,  # Limit depth to prevent overfitting
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        bootstrap=True,
        oob_score=True,
        verbose=0
    ))
])

# Optional: Hyperparameter tuning (commented out for faster training)
# Uncomment to enable GridSearchCV for better hyperparameters
USE_GRID_SEARCH = False

if USE_GRID_SEARCH:
    print("   Performing hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'classifier__n_estimators': [300, 500, 700],
        'classifier__max_depth': [15, 20, 25],
        'classifier__min_samples_split': [5, 10, 15],
        'classifier__min_samples_leaf': [2, 4, 6]
    }
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='f1_weighted',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    pipeline = grid_search.best_estimator_
    print(f"   ✓ Best parameters: {grid_search.best_params_}")
else:
    # Train the model with default parameters
    pipeline.fit(X_train, y_train)
    print("   ✓ Model training completed")

# =========================
# 9. Feature Selection based on Importance
# =========================
print("\n[9/10] Applying feature selection based on importance...")
classifier = pipeline.named_steps["classifier"]

# Get original feature names
original_feature_names = list(X.columns)
importance_values = classifier.feature_importances_

# Check which features were dropped by the imputer (all NaN features)
print("   Checking for features dropped during preprocessing...")
dropped_features = []
for col in X.columns:
    if X[col].isna().all():
        dropped_features.append(col)

if dropped_features:
    print(f"   ⚠️  Found {len(dropped_features)} features with all NaN values: {dropped_features[:5]}{'...' if len(dropped_features) > 5 else ''}")
    print(f"      These features were excluded by the imputer and will have 0 importance")

# Get the actual features that went through the pipeline (not dropped)
# The imputer preserves order but skips all-NaN features
# So we need to map importances to the features that weren't dropped
used_features = [col for col in X.columns if col not in dropped_features]

# Verify the mapping
if len(used_features) != len(importance_values):
    print(f"   ⚠️  Warning: Feature count mismatch!")
    print(f"      Features used: {len(used_features)}")
    print(f"      Importance values: {len(importance_values)}")
    # Use the minimum to avoid errors
    min_len = min(len(used_features), len(importance_values))
    used_features = used_features[:min_len]
    importance_values = importance_values[:min_len]
    print(f"      Using first {min_len} features/importances")

# Create importance mapping for all original features
# Features that were dropped get 0 importance
feature_importance_dict = {}
importance_idx = 0

for feat in original_feature_names:
    if feat in dropped_features:
        # Dropped features get 0 importance
        feature_importance_dict[feat] = 0.0
    else:
        # Map to the corresponding importance value
        if importance_idx < len(importance_values):
            feature_importance_dict[feat] = importance_values[importance_idx]
            importance_idx += 1
        else:
            feature_importance_dict[feat] = 0.0

# Create DataFrame
feature_importance_df = pd.DataFrame({
    'feature': list(feature_importance_dict.keys()),
    'importance': list(feature_importance_dict.values())
}).sort_values('importance', ascending=False)

print(f"   ✓ Feature importance calculated for {len(feature_importance_df)} features")
if dropped_features:
    print(f"   ✓ {len(dropped_features)} dropped features assigned 0 importance")

# Select top features (keep top 80% of cumulative importance or top 40 features)
cumulative_importance = feature_importance_df['importance'].cumsum()
threshold_idx = (cumulative_importance >= 0.95).idxmax() if (cumulative_importance >= 0.95).any() else len(feature_importance_df)
top_n = min(max(threshold_idx + 1, 30), 40)  # Keep at least 30, max 40 features

selected_features = feature_importance_df.head(top_n)['feature'].tolist()
print(f"   ✓ Selected top {len(selected_features)} features (covering {cumulative_importance.iloc[top_n-1]*100:.1f}% importance)")

# Retrain with selected features if we want to optimize
# For now, we'll keep all features but save the selection info

# =========================
# 10. Model Evaluation
# =========================
print("\n[10/10] Evaluating model performance...")

# Predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Cross-validation scores
print("   Performing cross-validation...")
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\n   📊 Performance Metrics:")
print(f"   {'─' * 50}")
print(f"   Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision:       {precision:.4f}")
print(f"   Recall:          {recall:.4f}")
print(f"   F1-Score:        {f1:.4f}")
print(f"   CV Accuracy:     {cv_mean:.4f} ± {cv_std:.4f}")

# Out-of-bag score (if available)
if hasattr(pipeline.named_steps["classifier"], 'oob_score_'):
    print(f"   OOB Score:       {pipeline.named_steps['classifier'].oob_score_:.4f}")

print(f"\n   📋 Classification Report:")
print(f"   {'─' * 50}")
print(classification_report(y_test, y_pred, target_names=["NORMAL", "RECOVERING", "BROKEN"], zero_division=0))

print(f"\n   📊 Confusion Matrix:")
print(f"   {'─' * 50}")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# =========================
# 11. Feature Importance Analysis
# =========================
print("\nAnalyzing feature importance...")
# Use the feature_importance_df from section 9
feature_importance = feature_importance_df.copy()

print(f"\n   🔝 Top 10 Most Important Features:")
print(f"   {'─' * 50}")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:.4f}")

# =========================
# 10. Save Model and Artifacts
# =========================
print("\n" + "=" * 60)
print("SAVING MODEL AND ARTIFACTS")
print("=" * 60)

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save model
MODEL_PATH = "model/pump_rf_model.pkl"
joblib.dump(pipeline, MODEL_PATH)
print(f"✓ Model saved to: {MODEL_PATH}")

# Save feature names
FEATURE_PATH = "model/feature_names.pkl"
joblib.dump(list(X.columns), FEATURE_PATH)
print(f"✓ Feature names saved to: {FEATURE_PATH}")

# Save label encoder
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
print(f"✓ Label encoder saved to: {LABEL_ENCODER_PATH}")

# Save feature importance
FEATURE_IMPORTANCE_PATH = "model/feature_importance.pkl"
joblib.dump(feature_importance, FEATURE_IMPORTANCE_PATH)
print(f"✓ Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")

# Save training metrics
METRICS_PATH = "model/training_metrics.pkl"
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'cv_mean': cv_mean,
    'cv_std': cv_std,
    'confusion_matrix': cm,
    'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    'selected_features': selected_features,
    'top_n_features': top_n
}
if hasattr(pipeline.named_steps["classifier"], 'oob_score_'):
    metrics['oob_score'] = pipeline.named_steps['classifier'].oob_score_

joblib.dump(metrics, METRICS_PATH)
print(f"✓ Training metrics saved to: {METRICS_PATH}")

# =========================
# 11. Create Visualization
# =========================
print("\nCreating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=["NORMAL", "RECOVERING", "BROKEN"],
            yticklabels=["NORMAL", "RECOVERING", "BROKEN"])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# 2. Feature Importance (Top 15)
top_features = feature_importance.head(15)
axes[0, 1].barh(range(len(top_features)), top_features['importance'].values)
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['feature'].values)
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
axes[0, 1].invert_yaxis()

# 3. Class Distribution
class_counts.plot(kind='bar', ax=axes[1, 0], color=['green', 'orange', 'red'])
axes[1, 0].set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Class')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_xticklabels(["NORMAL", "RECOVERING", "BROKEN"], rotation=0)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Metrics Comparison
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]
axes[1, 1].bar(metrics_names, metrics_values, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(metrics_values):
    axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
VIZ_PATH = "model/training_visualization.png"
plt.savefig(VIZ_PATH, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {VIZ_PATH}")
plt.close()

print("\n" + "=" * 60)
print("✅ CURRENT STATUS MODEL TRAINING COMPLETED!")
print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"  • Accuracy: {accuracy*100:.2f}%")
print(f"  • Cross-Validation: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
print(f"  • F1-Score: {f1:.4f}")

# =========================
# 12. PREDICTIVE MAINTENANCE MODEL - Future Failure Prediction
# =========================
print("\n" + "=" * 60)
print("PREDICTIVE MAINTENANCE - FUTURE FAILURE PREDICTION")
print("=" * 60)

print("\n[1/6] Loading data with timestamp for time-series analysis...")
# Reload data with timestamp for time-series features
df_ts = pd.read_csv(DATA_PATH)
df_ts = df_ts.drop(columns=["Unnamed: 0"], errors="ignore")

# Convert timestamp to datetime if it's not already
if 'timestamp' in df_ts.columns:
    try:
        df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
        df_ts = df_ts.sort_values('timestamp').reset_index(drop=True)
    except:
        # If timestamp conversion fails, use index as time
        df_ts['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df_ts), freq='1min')
        df_ts = df_ts.sort_values('timestamp').reset_index(drop=True)
else:
    # Create synthetic timestamp
    df_ts['timestamp'] = pd.date_range(start='2020-01-01', periods=len(df_ts), freq='1min')
    df_ts = df_ts.sort_values('timestamp').reset_index(drop=True)

print(f"   ✓ Data loaded with timestamp: {len(df_ts):,} rows")

print("\n[2/6] Creating future failure target variable...")
# Encode current status
status_mapping = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
df_ts["current_status"] = df_ts["machine_status"].map(status_mapping)

# Create target: Will the pump break in the next N time steps?
# Look ahead window (predict if will break in next 100-500 time steps)
LOOKAHEAD_WINDOW = 200  # Predict if will break in next 200 time steps
WARNING_WINDOW = 50     # Also predict if will break in next 50 time steps (near-term warning)

# Create future failure targets
df_ts['will_break_soon'] = 0  # Will break in next LOOKAHEAD_WINDOW steps
df_ts['will_break_near'] = 0  # Will break in next WARNING_WINDOW steps

for i in range(len(df_ts)):
    # Check if pump will be broken in future windows
    future_window = df_ts.iloc[i:min(i+LOOKAHEAD_WINDOW, len(df_ts))]
    near_window = df_ts.iloc[i:min(i+WARNING_WINDOW, len(df_ts))]
    
    # Will break soon (in next LOOKAHEAD_WINDOW steps)
    if (future_window['current_status'] == 2).any():
        df_ts.loc[i, 'will_break_soon'] = 1
    
    # Will break near (in next WARNING_WINDOW steps)
    if (near_window['current_status'] == 2).any():
        df_ts.loc[i, 'will_break_near'] = 1

print(f"   ✓ Created future failure targets")
print(f"   • Will break soon ({LOOKAHEAD_WINDOW} steps): {df_ts['will_break_soon'].sum():,} samples ({df_ts['will_break_soon'].mean()*100:.2f}%)")
print(f"   • Will break near ({WARNING_WINDOW} steps): {df_ts['will_break_near'].sum():,} samples ({df_ts['will_break_near'].mean()*100:.2f}%)")

print("\n[3/6] Creating time-series features...")
# Create rolling window features for trend analysis
sensor_cols = [col for col in df_ts.columns if col.startswith('sensor_')]

# Add rolling statistics for key sensors (top 10 most important from main model)
if feature_importance is not None:
    top_sensors = feature_importance.head(10)['feature'].tolist()
else:
    top_sensors = sensor_cols[:10]

print(f"   Creating rolling features for top {len(top_sensors)} sensors...")

for sensor in top_sensors:
    if sensor in df_ts.columns:
        # Ensure sensor values are finite and within reasonable range
        sensor_values = df_ts[sensor].replace([np.inf, -np.inf], np.nan).fillna(0)
        sensor_values = np.clip(sensor_values, -1e6, 1e6)  # Clip extreme values
        
        # Rolling mean (trend)
        df_ts[f'{sensor}_rolling_mean_10'] = sensor_values.rolling(window=10, min_periods=1).mean()
        df_ts[f'{sensor}_rolling_mean_50'] = sensor_values.rolling(window=50, min_periods=1).mean()
        
        # Rolling std (volatility)
        rolling_std_10 = sensor_values.rolling(window=10, min_periods=1).std().fillna(0)
        rolling_std_50 = sensor_values.rolling(window=50, min_periods=1).std().fillna(0)
        df_ts[f'{sensor}_rolling_std_10'] = np.clip(rolling_std_10, 0, 1e6)
        df_ts[f'{sensor}_rolling_std_50'] = np.clip(rolling_std_50, 0, 1e6)
        
        # Rate of change (diff)
        diff_values = sensor_values.diff().fillna(0)
        df_ts[f'{sensor}_diff'] = np.clip(diff_values, -1e6, 1e6)
        
        # Percentage change (handle division by zero)
        pct_change_values = sensor_values.pct_change().fillna(0)
        # Replace infinity values (from division by zero) with 0
        pct_change_values = pct_change_values.replace([np.inf, -np.inf], 0)
        # Clip to reasonable range (-100% to 100% change, or larger if needed)
        df_ts[f'{sensor}_pct_change'] = np.clip(pct_change_values, -10, 10)

# Replace all infinity and very large values
print("   Cleaning infinity and extreme values...")
df_ts = df_ts.replace([np.inf, -np.inf], np.nan)
df_ts = df_ts.fillna(0)

# Clip all numeric columns to reasonable range
numeric_cols_ts = df_ts.select_dtypes(include=[np.number]).columns
for col in numeric_cols_ts:
    if col not in ['current_status', 'will_break_soon', 'will_break_near']:
        df_ts[col] = np.clip(df_ts[col], -1e6, 1e6)

# Final fill of any remaining NaN
df_ts = df_ts.bfill().fillna(0)

# Verify no infinity or extreme values remain
if np.isinf(df_ts.select_dtypes(include=[np.number]).values).any():
    print("   ⚠️  Warning: Some infinity values still present, replacing with 0")
    df_ts = df_ts.replace([np.inf, -np.inf], 0)

print(f"   ✓ Created {len([c for c in df_ts.columns if 'rolling' in c or '_diff' in c or '_pct' in c])} time-series features")

print("\n[4/6] Preparing features for predictive maintenance model...")
# Select features: original sensors + time-series features
pm_features = sensor_cols + [c for c in df_ts.columns if 'rolling' in c or '_diff' in c or '_pct' in c]
pm_features = [f for f in pm_features if f in df_ts.columns]

X_pm = df_ts[pm_features].copy()
y_pm_soon = df_ts['will_break_soon'].copy()
y_pm_near = df_ts['will_break_near'].copy()

# Remove last LOOKAHEAD_WINDOW rows (can't predict future for these)
X_pm = X_pm.iloc[:-LOOKAHEAD_WINDOW].copy()
y_pm_soon = y_pm_soon.iloc[:-LOOKAHEAD_WINDOW].copy()
y_pm_near = y_pm_near.iloc[:-LOOKAHEAD_WINDOW].copy()

# Clean data: replace infinity and clip extreme values
print("   Cleaning PM feature data...")
X_pm = X_pm.replace([np.inf, -np.inf], np.nan)
X_pm = X_pm.fillna(0)

# Clip all values to reasonable range to prevent overflow
for col in X_pm.columns:
    X_pm[col] = np.clip(X_pm[col].astype(float), -1e6, 1e6)

# Final check and clean
if np.isinf(X_pm.values).any() or np.isnan(X_pm.values).any():
    X_pm = X_pm.replace([np.inf, -np.inf, np.nan], 0)

print(f"   ✓ Features prepared: {X_pm.shape[0]:,} samples, {X_pm.shape[1]} features")
print(f"   ✓ Data range: [{X_pm.min().min():.2f}, {X_pm.max().max():.2f}]")

print("\n[5/6] Training predictive maintenance models...")
# Final data validation and cleaning
print("   Validating and cleaning PM features...")
X_pm = X_pm.replace([np.inf, -np.inf], np.nan)
X_pm = X_pm.fillna(0)

# Clip all values to reasonable range
for col in X_pm.columns:
    X_pm[col] = np.clip(X_pm[col].astype(float), -1e6, 1e6)

# Check for any remaining issues
if np.isinf(X_pm.values).any() or np.isnan(X_pm.values).any():
    print("   ⚠️  Warning: Found infinity or NaN values, replacing with 0")
    X_pm = X_pm.replace([np.inf, -np.inf, np.nan], 0)

print(f"   ✓ Data cleaned: {X_pm.shape[0]:,} samples, {X_pm.shape[1]} features")
print(f"   ✓ Data range: [{X_pm.min().min():.2f}, {X_pm.max().max():.2f}]")

# Train-test split (time-aware: don't shuffle)
split_idx = int(len(X_pm) * 0.8)
X_pm_train, X_pm_test = X_pm.iloc[:split_idx].copy(), X_pm.iloc[split_idx:].copy()
y_pm_soon_train, y_pm_soon_test = y_pm_soon.iloc[:split_idx], y_pm_soon.iloc[split_idx:]
y_pm_near_train, y_pm_near_test = y_pm_near.iloc[:split_idx], y_pm_near.iloc[split_idx:]

print(f"   Training set: {len(X_pm_train):,} samples")
print(f"   Test set: {len(X_pm_test):,} samples")

# Create preprocessing pipeline for PM model
pm_preprocessing = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Model 1: Will break soon (LOOKAHEAD_WINDOW steps)
print("\n   Training 'Will Break Soon' model...")
pm_pipeline_soon = Pipeline([
    ("preprocessing", pm_preprocessing),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0
    ))
])

pm_pipeline_soon.fit(X_pm_train, y_pm_soon_train)
y_pm_soon_pred = pm_pipeline_soon.predict(X_pm_test)
y_pm_soon_proba = pm_pipeline_soon.predict_proba(X_pm_test)[:, 1]

pm_soon_accuracy = accuracy_score(y_pm_soon_test, y_pm_soon_pred)
pm_soon_precision = precision_score(y_pm_soon_test, y_pm_soon_pred, zero_division=0)
pm_soon_recall = recall_score(y_pm_soon_test, y_pm_soon_pred, zero_division=0)
pm_soon_f1 = f1_score(y_pm_soon_test, y_pm_soon_pred, zero_division=0)

print(f"   ✓ 'Will Break Soon' Model Performance:")
print(f"     Accuracy: {pm_soon_accuracy:.4f}, Precision: {pm_soon_precision:.4f}, Recall: {pm_soon_recall:.4f}, F1: {pm_soon_f1:.4f}")

# Model 2: Will break near (WARNING_WINDOW steps)
print("\n   Training 'Will Break Near' model...")
pm_pipeline_near = Pipeline([
    ("preprocessing", pm_preprocessing),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
        verbose=0
    ))
])

pm_pipeline_near.fit(X_pm_train, y_pm_near_train)
y_pm_near_pred = pm_pipeline_near.predict(X_pm_test)
y_pm_near_proba = pm_pipeline_near.predict_proba(X_pm_test)[:, 1]

pm_near_accuracy = accuracy_score(y_pm_near_test, y_pm_near_pred)
pm_near_precision = precision_score(y_pm_near_test, y_pm_near_pred, zero_division=0)
pm_near_recall = recall_score(y_pm_near_test, y_pm_near_pred, zero_division=0)
pm_near_f1 = f1_score(y_pm_near_test, y_pm_near_pred, zero_division=0)

print(f"   ✓ 'Will Break Near' Model Performance:")
print(f"     Accuracy: {pm_near_accuracy:.4f}, Precision: {pm_near_precision:.4f}, Recall: {pm_near_recall:.4f}, F1: {pm_near_f1:.4f}")

print("\n[6/6] Saving predictive maintenance models...")
# Save PM models
PM_MODEL_SOON_PATH = "model/pump_pm_soon_model.pkl"
PM_MODEL_NEAR_PATH = "model/pump_pm_near_model.pkl"
PM_FEATURES_PATH = "model/pm_feature_names.pkl"
PM_METRICS_PATH = "model/pm_training_metrics.pkl"

joblib.dump(pm_pipeline_soon, PM_MODEL_SOON_PATH)
print(f"✓ PM 'Soon' model saved to: {PM_MODEL_SOON_PATH}")

joblib.dump(pm_pipeline_near, PM_MODEL_NEAR_PATH)
print(f"✓ PM 'Near' model saved to: {PM_MODEL_NEAR_PATH}")

joblib.dump(list(X_pm.columns), PM_FEATURES_PATH)
print(f"✓ PM feature names saved to: {PM_FEATURES_PATH}")

pm_metrics = {
    'soon_model': {
        'accuracy': pm_soon_accuracy,
        'precision': pm_soon_precision,
        'recall': pm_soon_recall,
        'f1_score': pm_soon_f1,
        'lookahead_window': LOOKAHEAD_WINDOW
    },
    'near_model': {
        'accuracy': pm_near_accuracy,
        'precision': pm_near_precision,
        'recall': pm_near_recall,
        'f1_score': pm_near_f1,
        'warning_window': WARNING_WINDOW
    }
}
joblib.dump(pm_metrics, PM_METRICS_PATH)
print(f"✓ PM metrics saved to: {PM_METRICS_PATH}")

print("\n" + "=" * 60)
print("✅ ALL MODELS TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"  Current Status Model:")
print(f"    • Accuracy: {accuracy*100:.2f}%")
print(f"    • Cross-Validation: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
print(f"  Predictive Maintenance Models:")
print(f"    • 'Will Break Soon' ({LOOKAHEAD_WINDOW} steps): Accuracy {pm_soon_accuracy*100:.2f}%, F1 {pm_soon_f1:.4f}")
print(f"    • 'Will Break Near' ({WARNING_WINDOW} steps): Accuracy {pm_near_accuracy*100:.2f}%, F1 {pm_near_f1:.4f}")
print(f"\nYou can now run the Streamlit app with: streamlit run app.py")
