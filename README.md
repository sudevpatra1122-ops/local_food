# local_food
Food Waste management system
"""
Streamlit ML Application - Complete EDA & Prediction Platform
Author: Your Name
GitHub: https://github.com/yourusername/streamlit-ml-app
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import custom modules
from src.eda import show_eda
from src.model import load_model, make_prediction, train_model
from src.utils import load_data, get_available_models
from src.visualize import plot_feature_importance

# Page configuration
st.set_page_config(
    page_title="ML Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.svg", width=100)
    st.title("🤖 ML Analytics")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "📌 Navigation",
        ["🏠 Home", "📊 Exploratory Data Analysis", "🎯 Model Prediction", "⚙️ Model Training"]
    )
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit")
    st.caption(f"Version: 1.0.0")

# Main content
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🚀 ML Analytics Platform</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 EDA Tools", "5+", "Visualizations")
    with col2:
        st.metric("🤖 Models", "3", "Ready to use")
    with col3:
        st.metric("⚡ Speed", "Fast", "Real-time predictions")
    
    st.markdown("---")
    
    # File uploader
    st.subheader("📁 Upload Your Dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with your dataset"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Quick stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📌 **Rows:** {df.shape[0]}")
            with col2:
                st.info(f"📌 **Columns:** {df.shape[1]}")
            with col3:
                missing = df.isnull().sum().sum()
                st.info(f"📌 **Missing Values:** {missing}")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        # Use sample data
        if st.button("📂 Load Sample Dataset"):
            sample_data = Path("data/sample_data.csv")
            if sample_data.exists():
                df = pd.read_csv(sample_data)
                st.session_state.df = df
                st.success("✅ Sample dataset loaded!")
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.warning("⚠️ Sample data not found. Please upload your own CSV file.")
        
        st.markdown("""
        <div class="info-box">
        <b>📝 Instructions:</b><br>
        1. Upload a CSV file using the uploader above<br>
        2. Explore your data with EDA tools<br>
        3. Train models and make predictions
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Exploratory Data Analysis":
    if st.session_state.df is not None:
        show_eda(st.session_state.df)
    else:
        st.warning("⚠️ Please upload a dataset first (go to Home page)")

elif page == "🎯 Model Prediction":
    if st.session_state.df is not None:
        st.subheader("🎯 Make Predictions")
        
        # Load or train model
        if st.session_state.model is None:
            model_path = Path("models/model.pkl")
            if model_path.exists():
                st.session_state.model = load_model(str(model_path))
                st.success("✅ Model loaded successfully!")
            else:
                st.warning("⚠️ No trained model found. Please train a model first.")
        
        if st.session_state.model is not None:
            # Get feature columns (exclude target if exists)
            df = st.session_state.df
            feature_cols = [col for col in df.columns if col != 'target']
            
            st.subheader("🔧 Input Features")
            
            # Create input fields dynamically
            input_data = {}
            cols = st.columns(3)
            for idx, col in enumerate(feature_cols[:6]):  # Limit to 6 features
                with cols[idx % 3]:
                    input_data[col] = st.number_input(
                        f"{col}",
                        value=float(df[col].mean()),
                        step=0.1,
                        key=f"input_{col}"
                    )
            
            if st.button("🚀 Predict", type="primary"):
                features = [input_data[col] for col in feature_cols[:6]]
                prediction, probability = make_prediction(
                    st.session_state.model, 
                    [features]
                )
                
                if prediction is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"**Prediction:** {prediction[0]}")
                    with col2:
                        if probability is not None:
                            st.metric("Confidence", f"{probability.max():.2%}")
                
                # Store predictions
                st.session_state.predictions = prediction
    else:
        st.warning("⚠️ Please upload a dataset first (go to Home page)")

elif page == "⚙️ Model Training":
    st.subheader("⚙️ Train New Model")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Random Forest", "Logistic Regression", "Decision Tree", "XGBoost"]
        )
        
        # Target column selection
        target_col = st.selectbox(
            "Select Target Column",
            df.columns.tolist()
        )
        
        # Training parameters
        st.subheader("📊 Training Parameters")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, step=1)
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                try:
                    model, metrics = train_model(
                        df,
                        target_col,
                        model_type,
                        test_size,
                        random_state
                    )
                    
                    # Save model
                    model_path = Path("models/model.pkl")
                    model_path.parent.mkdir(exist_ok=True)
                    import joblib
                    joblib.dump(model, model_path)
                    st.session_state.model = model
                    
                    st.success("✅ Model trained and saved successfully!")
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    with col2:
                        st.metric("Precision", f"{metrics['precision']:.2%}")
                    with col3:
                        st.metric("Recall", f"{metrics['recall']:.2%}")
                    
                    # Feature importance
                    st.subheader("📊 Feature Importance")
                    if hasattr(model, 'feature_importances_'):
                        fig = plot_feature_importance(model, df.columns[:-1])
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
    else:
        st.warning("⚠️ Please upload a dataset first (go to Home page)")
