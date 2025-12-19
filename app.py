import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Water Pump Failure Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .status-normal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .status-recovering {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .status-broken {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load Model & Artifacts
# -------------------------------------------------
MODEL_PATH = "model/pump_rf_model.pkl"
FEATURE_PATH = "model/feature_names.pkl"
FEATURE_IMPORTANCE_PATH = "model/feature_importance.pkl"
METRICS_PATH = "model/training_metrics.pkl"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"

# Predictive Maintenance Model Paths
PM_MODEL_SOON_PATH = "model/pump_pm_soon_model.pkl"
PM_MODEL_NEAR_PATH = "model/pump_pm_near_model.pkl"
PM_FEATURES_PATH = "model/pm_feature_names.pkl"
PM_METRICS_PATH = "model/pm_training_metrics.pkl"

@st.cache_data
def load_model_artifacts():
    """Load all model artifacts with caching"""
    artifacts = {}
    try:
        artifacts['model'] = joblib.load(MODEL_PATH)
        artifacts['feature_names'] = joblib.load(FEATURE_PATH)
        
        if os.path.exists(FEATURE_IMPORTANCE_PATH):
            artifacts['feature_importance'] = joblib.load(FEATURE_IMPORTANCE_PATH)
        
        if os.path.exists(METRICS_PATH):
            artifacts['metrics'] = joblib.load(METRICS_PATH)
        
        if os.path.exists(LABEL_ENCODER_PATH):
            artifacts['label_encoder'] = joblib.load(LABEL_ENCODER_PATH)
        
        # Load Predictive Maintenance models (optional)
        if os.path.exists(PM_MODEL_SOON_PATH) and os.path.exists(PM_MODEL_NEAR_PATH):
            artifacts['pm_model_soon'] = joblib.load(PM_MODEL_SOON_PATH)
            artifacts['pm_model_near'] = joblib.load(PM_MODEL_NEAR_PATH)
            artifacts['pm_feature_names'] = joblib.load(PM_FEATURES_PATH)
            if os.path.exists(PM_METRICS_PATH):
                artifacts['pm_metrics'] = joblib.load(PM_METRICS_PATH)
        
        return artifacts, None
    except Exception as e:
        return None, str(e)

artifacts, error = load_model_artifacts()

if error or artifacts is None:
    st.error(
        f"❌ Error loading model files: {error if error else 'Files not found'}\n\n"
        "Please run `python train_model.py` first to train the model."
    )
    st.stop()

model = artifacts['model']
feature_names = artifacts['feature_names']
feature_importance = artifacts.get('feature_importance')
metrics = artifacts.get('metrics', {})
label_encoder = artifacts.get('label_encoder')

# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<h1 class="main-header">🚰 Water Pump Failure Prediction System</h1>', unsafe_allow_html=True)

# Create tabs for different sections
tabs_list = ["🔮 Prediction", "📊 Model Performance", "📈 Feature Analysis", "📤 Batch Prediction"]
if 'pm_model_soon' in artifacts:
    tabs_list.append("🔮 Predictive Maintenance")

if len(tabs_list) == 5:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs_list)
else:
    tab1, tab2, tab3, tab4 = st.tabs(tabs_list)
    tab5 = None

# -------------------------------------------------
# TAB 1: Prediction
# -------------------------------------------------
with tab1:
    st.write(
        "Enter sensor readings below to predict the operational status of the water pump. "
        "The model will analyze the sensor data and provide a prediction with confidence scores."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔧 Sensor Inputs")
        
        # Initialize all features with 0
        input_data = dict.fromkeys(feature_names, 0.0)
        
        # Allow user to control first 15 sensors (increased from 10)
        NUM_VISIBLE_SENSORS = 15
        
        # Create two columns for sensor inputs
        cols = st.columns(2)
        for idx, sensor in enumerate(feature_names[:NUM_VISIBLE_SENSORS]):
            col_idx = idx % 2
            with cols[col_idx]:
                # Get feature importance for this sensor if available
                importance_info = ""
                if feature_importance is not None:
                    sensor_importance = feature_importance[
                        feature_importance['feature'] == sensor
                    ]['importance'].values
                    if len(sensor_importance) > 0:
                        importance_info = f" (Importance: {sensor_importance[0]:.3f})"
                
                input_data[sensor] = st.slider(
                    label=f"{sensor}{importance_info}",
                    min_value=-10.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    help=f"Adjust {sensor} value. Range: -10.0 to 10.0"
                )
        
        st.info(f"💡 {len(feature_names) - NUM_VISIBLE_SENSORS} additional sensors are set to default values (0.0). "
                "You can modify them via batch prediction or CSV upload.")
    
    with col2:
        st.subheader("📋 Quick Actions")
        
        if st.button("🔄 Reset All to Default", use_container_width=True):
            st.rerun()
        
        if st.button("📊 View Feature Importance", use_container_width=True):
            st.session_state.show_importance = True
        
        if st.button("ℹ️ Model Info", use_container_width=True):
            st.session_state.show_model_info = True
    
    # Prediction Section
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    if st.button("🔮 Predict Pump Status", type="primary", use_container_width=True):
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        max_prob = probabilities.max()
        
        status_map = {
            0: ("✅ NORMAL", "status-normal"),
            1: ("⚠️ RECOVERING", "status-recovering"),
            2: ("❌ BROKEN", "status-broken")
        }
        
        status_text, status_class = status_map[prediction]
        
        # Display prediction with styling
        st.markdown(f'<div class="prediction-box {status_class}">', unsafe_allow_html=True)
        st.markdown(f"### {status_text}")
        st.markdown(f"**Confidence:** {max_prob*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Probability visualization
        col_prob1, col_prob2 = st.columns([2, 1])
        
        with col_prob1:
            prob_df = pd.DataFrame({
                "Status": ["NORMAL", "RECOVERING", "BROKEN"],
                "Probability": probabilities
            })
            st.bar_chart(prob_df.set_index("Status"), height=300)
        
        with col_prob2:
            st.write("**Probability Distribution:**")
            for i, (status, prob) in enumerate(zip(["NORMAL", "RECOVERING", "BROKEN"], probabilities)):
                color = ["#28a745", "#ffc107", "#dc3545"][i]
                st.markdown(
                    f'<div style="background-color: {color}20; padding: 0.5rem; margin: 0.5rem 0; border-radius: 0.25rem;">'
                    f'<strong>{status}:</strong> {prob*100:.2f}%</div>',
                    unsafe_allow_html=True
                )
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 0:
            st.success("✅ **Pump is operating normally.** Continue regular monitoring.")
        elif prediction == 1:
            st.warning("⚠️ **Pump is recovering.** Monitor closely and check for any anomalies. Consider preventive maintenance.")
        else:
            st.error("❌ **Pump is broken!** Immediate action required. Shut down the pump and perform emergency maintenance.")
        
        # SHAP Explainability
        st.markdown("---")
        st.subheader("🔍 Model Explainability (SHAP)")
        
        try:
            # Get the classifier from the pipeline
            classifier = model.named_steps["classifier"]
            
            # Transform input the same way as training
            transformed_input = model.named_steps["preprocessing"].transform(input_df)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_input)
            
            # Handle multi-class SHAP values
            if isinstance(shap_values, list):
                shap_values_for_pred = shap_values[prediction]
                expected_value = explainer.expected_value[prediction] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_values_for_pred = shap_values
                expected_value = explainer.expected_value
            
            # Create visualizations
            shap_col1, shap_col2 = st.columns(2)
            
            with shap_col1:
                st.write("**Waterfall Plot**")
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values_for_pred[0],
                        base_values=expected_value,
                        feature_names=feature_names
                    ),
                    show=False,
                    max_display=15
                )
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)
            
            with shap_col2:
                st.write("**Top Contributing Features**")
                # Get top contributing features
                contrib_df = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values_for_pred[0]
                }).sort_values('SHAP Value', key=abs, ascending=False).head(10)
                
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                colors = ['red' if x < 0 else 'green' for x in contrib_df['SHAP Value']]
                ax2.barh(range(len(contrib_df)), contrib_df['SHAP Value'], color=colors)
                ax2.set_yticks(range(len(contrib_df)))
                ax2.set_yticklabels(contrib_df['Feature'])
                ax2.set_xlabel('SHAP Value')
                ax2.set_title('Top 10 Feature Contributions')
                ax2.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
        
        except Exception as e:
            st.warning(
                f"SHAP explanation could not be generated: {str(e)}. "
                "This does not affect prediction accuracy."
            )

# -------------------------------------------------
# TAB 2: Model Performance
# -------------------------------------------------
with tab2:
    st.subheader("📊 Model Performance Metrics")
    
    if metrics:
        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
        
        with col_met1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
        with col_met2:
            st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        with col_met3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col_met4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        
        if 'cv_mean' in metrics:
            st.metric("Cross-Validation Accuracy", 
                     f"{metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
        
        if 'oob_score' in metrics:
            st.metric("Out-of-Bag Score", f"{metrics['oob_score']*100:.2f}%")
        
        # Confusion Matrix
        if 'confusion_matrix' in metrics:
            st.subheader("Confusion Matrix")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=["NORMAL", "RECOVERING", "BROKEN"],
                       yticklabels=["NORMAL", "RECOVERING", "BROKEN"])
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
            plt.close(fig)
        
        # Classification Report
        if 'classification_report' in metrics:
            st.subheader("Detailed Classification Report")
            report = metrics['classification_report']
            if isinstance(report, dict):
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Model metrics not available. Please retrain the model to generate metrics.")

# -------------------------------------------------
# TAB 3: Feature Analysis
# -------------------------------------------------
with tab3:
    st.subheader("📈 Feature Importance Analysis")
    
    if feature_importance is not None:
        # Top features visualization
        num_features = st.slider("Number of top features to display", 10, 50, 20)
        
        top_features = feature_importance.head(num_features)
        
        fig, ax = plt.subplots(figsize=(12, max(8, num_features * 0.3)))
        ax.barh(range(len(top_features)), top_features['importance'].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {num_features} Most Important Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Feature importance table
        st.subheader("Feature Importance Table")
        st.dataframe(top_features, use_container_width=True)
        
        # Statistics
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Features", len(feature_importance))
        with col_stat2:
            st.metric("Mean Importance", f"{feature_importance['importance'].mean():.4f}")
        with col_stat3:
            st.metric("Max Importance", f"{feature_importance['importance'].max():.4f}")
    else:
        st.info("Feature importance data not available. Please retrain the model.")

# -------------------------------------------------
# TAB 4: Batch Prediction
# -------------------------------------------------
with tab4:
    st.subheader("📤 Batch Prediction")
    st.write("Upload a CSV file with sensor data to make predictions for multiple samples.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"**Uploaded file:** {uploaded_file.name}")
            st.write(f"**Rows:** {len(df_upload)}")
            
            # Check if required columns exist
            missing_cols = set(feature_names) - set(df_upload.columns)
            if missing_cols:
                st.warning(f"Missing columns: {', '.join(list(missing_cols)[:10])}...")
                st.info("Filling missing columns with default value 0.0")
                for col in missing_cols:
                    df_upload[col] = 0.0
            
            # Select only required columns
            df_input = df_upload[feature_names].copy()
            
            if st.button("🔮 Predict All", type="primary"):
                # Make predictions
                predictions = model.predict(df_input)
                probabilities = model.predict_proba(df_input)
                
                # Create results dataframe
                results_df = df_upload.copy()
                results_df['Predicted_Status'] = predictions
                results_df['Status_Label'] = results_df['Predicted_Status'].map({
                    0: "NORMAL", 1: "RECOVERING", 2: "BROKEN"
                })
                results_df['Confidence'] = probabilities.max(axis=1)
                results_df['Prob_NORMAL'] = probabilities[:, 0]
                results_df['Prob_RECOVERING'] = probabilities[:, 1]
                results_df['Prob_BROKEN'] = probabilities[:, 2]
                
                st.success(f"✅ Predictions completed for {len(results_df)} samples!")
                
                # Display summary
                st.subheader("📊 Prediction Summary")
                summary = results_df['Status_Label'].value_counts()
                st.bar_chart(summary)
                
                # Display results table
                st.subheader("📋 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# -------------------------------------------------
# TAB 5: Predictive Maintenance (Future Failure Prediction)
# -------------------------------------------------
if 'pm_model_soon' in artifacts and tab5 is not None:
    with tab5:
        st.subheader("🔮 Predictive Maintenance - Future Failure Prediction")
        st.write(
            "This feature predicts if the pump **will break in the future** based on current sensor readings. "
            "It uses time-series analysis to forecast potential failures before they occur."
        )
        
        # Display PM model info
        if 'pm_metrics' in artifacts:
            pm_metrics = artifacts['pm_metrics']
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"**'Will Break Soon' Model:** Predicts failure in next {pm_metrics['soon_model'].get('lookahead_window', 200)} time steps")
                st.metric("Accuracy", f"{pm_metrics['soon_model']['accuracy']*100:.2f}%")
            with col_info2:
                st.info(f"**'Will Break Near' Model:** Predicts failure in next {pm_metrics['near_model'].get('warning_window', 50)} time steps")
                st.metric("Accuracy", f"{pm_metrics['near_model']['accuracy']*100:.2f}%")
        
        st.markdown("---")
        
        # Input section
        st.subheader("📊 Current Sensor Readings")
        st.write("Enter current sensor values to predict future failure risk:")
        
        # Get PM feature names
        pm_feature_names = artifacts.get('pm_feature_names', feature_names)
        
        # Initialize input data for PM model
        # PM model needs original sensors + rolling features
        # For simplicity, we'll use current sensor values and estimate rolling features
        pm_input_data = {}
        
        # Get original sensor values (first 15 visible)
        NUM_VISIBLE_SENSORS = 15
        sensor_inputs = {}
        
        cols = st.columns(3)
        for idx, sensor in enumerate(feature_names[:NUM_VISIBLE_SENSORS]):
            col_idx = idx % 3
            with cols[col_idx]:
                sensor_inputs[sensor] = st.number_input(
                    label=sensor,
                    min_value=-10.0,
                    max_value=10.0,
                    value=0.0,
                    step=0.1,
                    key=f"pm_{sensor}"
                )
        
        # Set all sensors (visible + default for others)
        for sensor in feature_names:
            if sensor in sensor_inputs:
                pm_input_data[sensor] = sensor_inputs[sensor]
            else:
                pm_input_data[sensor] = 0.0
        
        # Estimate rolling features (simplified - using current values as proxy)
        # In real scenario, these would come from historical data
        for sensor in feature_names[:10]:  # Top 10 sensors
            current_val = pm_input_data[sensor]
            pm_input_data[f'{sensor}_rolling_mean_10'] = current_val
            pm_input_data[f'{sensor}_rolling_mean_50'] = current_val
            pm_input_data[f'{sensor}_rolling_std_10'] = abs(current_val) * 0.1  # Estimate std
            pm_input_data[f'{sensor}_rolling_std_50'] = abs(current_val) * 0.15
            pm_input_data[f'{sensor}_diff'] = 0.0  # No change if single reading
            pm_input_data[f'{sensor}_pct_change'] = 0.0
        
        # Ensure all PM features are present
        for feat in pm_feature_names:
            if feat not in pm_input_data:
                pm_input_data[feat] = 0.0
        
        # Create DataFrame with correct feature order
        pm_input_df = pd.DataFrame([pm_input_data])[pm_feature_names]
        
        st.markdown("---")
        
        if st.button("🔮 Predict Future Failure Risk", type="primary", use_container_width=True):
            pm_model_soon = artifacts['pm_model_soon']
            pm_model_near = artifacts['pm_model_near']
            
            # Make predictions
            will_break_soon_prob = pm_model_soon.predict_proba(pm_input_df)[0, 1]
            will_break_near_prob = pm_model_near.predict_proba(pm_input_df)[0, 1]
            
            will_break_soon = pm_model_soon.predict(pm_input_df)[0]
            will_break_near = pm_model_near.predict(pm_input_df)[0]
            
            # Display results
            st.subheader("📊 Future Failure Prediction Results")
            
            col_soon, col_near = st.columns(2)
            
            with col_soon:
                st.markdown("### ⏰ Will Break Soon Prediction")
                if will_break_soon == 1:
                    st.error(f"⚠️ **HIGH RISK** - Pump likely to break soon!")
                    st.progress(will_break_soon_prob)
                    st.metric("Risk Probability", f"{will_break_soon_prob*100:.2f}%")
                else:
                    st.success(f"✅ **LOW RISK** - Pump should remain operational")
                    st.progress(will_break_soon_prob)
                    st.metric("Risk Probability", f"{will_break_soon_prob*100:.2f}%")
            
            with col_near:
                st.markdown("### 🚨 Will Break Near Prediction")
                if will_break_near == 1:
                    st.error(f"🚨 **CRITICAL RISK** - Pump likely to break very soon!")
                    st.progress(will_break_near_prob)
                    st.metric("Risk Probability", f"{will_break_near_prob*100:.2f}%")
                else:
                    st.success(f"✅ **LOW RISK** - No immediate threat detected")
                    st.progress(will_break_near_prob)
                    st.metric("Risk Probability", f"{will_break_near_prob*100:.2f}%")
            
            # Risk assessment
            st.markdown("---")
            st.subheader("💡 Risk Assessment & Recommendations")
            
            if will_break_near == 1:
                st.error("""
                **🚨 CRITICAL ALERT - IMMEDIATE ACTION REQUIRED**
                
                The predictive maintenance model indicates a **high probability of failure in the near term**. 
                Recommended actions:
                - **Immediately schedule emergency maintenance**
                - **Monitor pump continuously**
                - **Prepare backup systems**
                - **Notify maintenance team immediately**
                """)
            elif will_break_soon == 1:
                st.warning("""
                **⚠️ WARNING - PREVENTIVE ACTION RECOMMENDED**
                
                The model predicts potential failure in the medium term. Recommended actions:
                - **Schedule preventive maintenance within 24-48 hours**
                - **Increase monitoring frequency**
                - **Review sensor readings for anomalies**
                - **Prepare maintenance resources**
                """)
            else:
                st.success("""
                **✅ NORMAL OPERATION - CONTINUE MONITORING**
                
                The predictive maintenance model indicates low risk of failure. 
                - Continue regular monitoring
                - Maintain scheduled maintenance intervals
                - Monitor sensor trends for early warning signs
                """)
            
            # Visualization
            st.markdown("---")
            st.subheader("📈 Risk Probability Visualization")
            
            risk_df = pd.DataFrame({
                'Timeframe': ['Near-term Risk', 'Medium-term Risk'],
                'Probability': [will_break_near_prob, will_break_soon_prob],
                'Status': ['Critical' if will_break_near == 1 else 'Low', 
                          'Warning' if will_break_soon == 1 else 'Low']
            })
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#dc3545' if s == 'Critical' else '#ffc107' if s == 'Warning' else '#28a745' 
                     for s in risk_df['Status']]
            ax.bar(risk_df['Timeframe'], risk_df['Probability'], color=colors, alpha=0.7)
            ax.set_ylabel('Failure Probability', fontsize=12)
            ax.set_title('Future Failure Risk Assessment', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1])
            for i, (idx, row) in enumerate(risk_df.iterrows()):
                ax.text(i, row['Probability'] + 0.02, f"{row['Probability']*100:.1f}%", 
                       ha='center', fontweight='bold')
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Note about limitations
            st.info("""
            **ℹ️ Note:** This prediction is based on current sensor readings. For more accurate predictions, 
            provide historical sensor data (last 50+ readings) to calculate proper rolling statistics. 
            The model works best with time-series data showing trends and patterns.
            """)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "Developed as a Data Mining & Predictive Maintenance project "
    "using the Water Pump Sensor Dataset. | "
    f"Model trained on {len(feature_names)} sensor features."
)
