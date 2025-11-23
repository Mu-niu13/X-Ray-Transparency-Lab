"""
X-Ray Transparency Lab - Streamlit Web Application
Interactive interface for explainable AI pneumonia diagnosis
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys

# Add src to path
sys.path.append('src')

from explanations import get_all_explanations
from report_generator import generate_report, generate_summary

# Page configuration
st.set_page_config(
    page_title="X-Ray Transparency Lab",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .pneumonia {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .normal {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def check_setup():
    """Check if required files exist"""
    required = {
        'Model': 'models/pneumonia_classifier.pth',
        'Embeddings': 'embeddings/embeddings.npy',
        'FAISS Index': 'embeddings/similarity_index.faiss',
        'Labels': 'embeddings/labels.npy',
        'Paths': 'embeddings/paths.pkl'
    }
    
    missing = []
    for name, path in required.items():
        if not os.path.exists(path):
            missing.append(f"- {name}: `{path}`")
    
    return missing


def load_sample_images():
    """Load sample images from test directory"""
    sample_dir = 'data/chest_xray/test'
    samples = {}
    
    # Try to load a few samples from each class
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(sample_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
            if images:
                samples[class_name] = [os.path.join(class_dir, img) for img in images[:5]]
    
    return samples


def display_similar_cases(similar_cases):
    """Display similar training cases in a grid"""
    cols = st.columns(3)
    
    for idx, case in enumerate(similar_cases[:6]):
        col_idx = idx % 3
        
        with cols[col_idx]:
            try:
                # Load and display image
                img = Image.open(case['path'])
                st.image(img, use_column_width=True)
                
                # Display metadata
                st.markdown(f"""
                **Label:** {case['label_name']}  
                **Similarity:** {case['similarity']:.3f}
                """)
            except Exception as e:
                st.error(f"Could not load image: {e}")


def main():
    # Header
    st.markdown('<p class="main-header">🫁 X-Ray Transparency Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Multi-View Explanations for AI Pneumonia Diagnosis</p>', 
                unsafe_allow_html=True)
    
    # Check setup
    missing = check_setup()
    if missing:
        st.error("⚠️ **Setup Incomplete**")
        st.markdown("The following required files are missing:")
        for item in missing:
            st.markdown(item)
        st.markdown("""
        **Setup Instructions:**
        1. Train the model: `python src/train_model.py`
        2. Generate embeddings: `python src/generate_embeddings.py`
        
        See README.md for detailed instructions.
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. **Upload** a chest X-ray image or select a sample
        2. Click **"Run AI Analysis"**
        3. **Explore** multiple explanation views
        4. **Read** the generated report
        
        ---
        """)
        
        st.header("⚙️ Settings")
        device = st.selectbox(
            "Compute Device",
            options=['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'],
            help="Use GPU if available for faster processing"
        )
        
        show_probabilities = st.checkbox("Show probability details", value=True)
        
        st.markdown("---")
        st.markdown("""
        **About This Tool**
        
        This system uses three complementary XAI methods:
        - 🎯 **Grad-CAM**: Visual attention
        - 🔍 **Occlusion**: Importance testing
        - 📊 **Similar Cases**: Example-based reasoning
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔬 Demo", "📖 About & Methods", "🗂️ Case Gallery"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input")
            
            # File upload
            uploaded_file = st.file_uploader(
                "Upload Chest X-Ray Image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a PA-view chest X-ray in JPEG or PNG format"
            )
            
            # Sample selection
            use_sample = st.checkbox("Use sample image instead")
            
            selected_sample = None
            if use_sample:
                samples = load_sample_images()
                if samples:
                    sample_options = []
                    sample_map = {}
                    
                    for class_name, paths in samples.items():
                        for i, path in enumerate(paths):
                            label = f"{class_name} - Sample {i+1}"
                            sample_options.append(label)
                            sample_map[label] = path
                    
                    selected_label = st.selectbox("Select sample", sample_options)
                    selected_sample = sample_map[selected_label]
                else:
                    st.warning("No sample images found in test directory")
            
            # Determine which image to use
            image_path = None
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = "temp_upload.jpg"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                image_path = temp_path
            elif selected_sample:
                image_path = selected_sample
            
            # Display input image
            if image_path:
                st.image(image_path, caption="Input X-Ray", use_column_width=True)
                
                # Run analysis button
                if st.button("🚀 Run AI Analysis", type="primary", use_container_width=True):
                    with st.spinner("Analyzing X-ray... This may take a minute..."):
                        try:
                            # Get all explanations
                            results = get_all_explanations(
                                image_path,
                                model_path='models/pneumonia_classifier.pth',
                                embeddings_dir='embeddings',
                                device=device
                            )
                            
                            # Store in session state
                            st.session_state['results'] = results
                            st.success("✅ Analysis complete!")
                            
                        except Exception as e:
                            st.error(f"❌ Error during analysis: {str(e)}")
                            st.exception(e)
        
        with col2:
            st.subheader("Results")
            
            if 'results' in st.session_state:
                results = st.session_state['results']
                
                # Prediction box
                prediction = results['prediction']
                probability = results['probability']
                
                box_class = 'pneumonia' if prediction == 'PNEUMONIA' else 'normal'
                
                st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2>{prediction}</h2>
                    <h3>{probability*100:.1f}% Confidence</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Summary
                st.markdown("**Quick Summary:**")
                st.info(generate_summary(results))
                
                # Detailed report
                with st.expander("📄 View Detailed Report", expanded=False):
                    report = generate_report(results)
                    st.markdown(report)
                
                # Explanation views
                st.markdown("---")
                st.markdown("### 🔍 Multi-View Explanations")
                
                view_tabs = st.tabs([
                    "Original", 
                    "Grad-CAM", 
                    "Occlusion", 
                    "Similar Cases"
                ])
                
                with view_tabs[0]:
                    st.image(results['original_image'], 
                            caption="Original X-Ray",
                            use_column_width=True)
                
                with view_tabs[1]:
                    st.image(results['gradcam_overlay'], 
                            caption="Grad-CAM: Model Attention Heatmap",
                            use_column_width=True)
                    st.caption("🎯 Red regions indicate areas the model focused on most")
                
                with view_tabs[2]:
                    st.image(results['occlusion_overlay'], 
                            caption="Occlusion Sensitivity: Feature Importance",
                            use_column_width=True)
                    st.caption("🔍 Red regions have the strongest impact on the prediction")
                
                with view_tabs[3]:
                    st.markdown("**Most Similar Training Cases:**")
                    display_similar_cases(results['similar_cases'])
                    st.caption("📊 Images from training set most similar to the input")
            
            else:
                st.info("👆 Upload an image or select a sample, then click 'Run AI Analysis'")
    
    with tab2:
        st.header("About This System")
        
        st.markdown("""
        ### 🎯 Purpose
        
        This tool demonstrates **explainable AI (XAI)** techniques for medical image analysis.
        It helps understand *how* and *why* the AI model makes its pneumonia predictions.
        
        ### 🔬 Methods
        
        #### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)
        - Visualizes which regions of the image the neural network pays attention to
        - Uses gradients flowing back to the last convolutional layer
        - Highlights features that activate strongly for the predicted class
        
        #### 2. Occlusion Sensitivity
        - Tests importance by systematically masking parts of the image
        - Measures how much the prediction changes when each region is hidden
        - Confirms which areas are truly critical for the decision
        
        #### 3. Example-Based Explanation
        - Finds similar cases from the training data
        - Uses deep feature embeddings and nearest neighbor search
        - Provides concrete visual comparisons
        
        ### ⚠️ Limitations
        
        - This is a **research/educational tool**, not for clinical use
        - The model is trained on limited data and may make errors
        - AI explanations are approximations of model reasoning
        - Always require expert medical interpretation for real diagnosis
        
        ### 📊 Model Details
        
        - **Architecture:** ResNet-18 with transfer learning
        - **Training Data:** Kaggle Chest X-Ray Pneumonia Dataset
        - **Classes:** NORMAL vs PNEUMONIA (binary classification)
        - **Input:** 224×224 RGB images
        
        ### 📚 References
        
        - Selvaraju et al. (2017) - Grad-CAM
        - Zeiler & Fergus (2014) - Occlusion Sensitivity
        - Dataset: Kermany et al. (2018)
        """)
    
    with tab3:
        st.header("Case Gallery")
        st.info("Coming soon: Pre-curated cases showcasing model performance on various scenarios")
        
        # Could add pre-analyzed interesting cases here


if __name__ == '__main__':
    main()