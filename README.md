# X-Ray Transparency Lab

Multi-view explainable AI system for pneumonia detection in chest X-rays.

## Overview

This project provides an interactive web interface for understanding AI-based pneumonia diagnosis through multiple explanation methods:

- **Grad-CAM**: Visual attention heatmaps
- **Occlusion Sensitivity**: Perturbation-based importance
- **Example-Based**: Similar training cases
- **Descriptive Reports**: Natural language explanations

## Setup

**📖 For detailed step-by-step instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/x-ray-transparency-lab.git
cd x-ray-transparency-lab
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup credentials

```bash
cp .env.example .env
# Edit .env with your Kaggle and Google Drive credentials
export $(cat .env | xargs)  # Load environment variables
```

**Get credentials:**
- **Kaggle**: https://www.kaggle.com/settings/account → "Create New Token"
- **Google Drive File ID**: Contact maintainer or see COLAB_TRAINING_GUIDE.md

### 4. Download dataset (~5.3 GB)

```bash
python download_data.py
```

### 5. Download pre-trained model (~142 MB)

```bash
python download_trained_model.py
```

Or use command line argument:
```bash
python download_trained_model.py --file_id YOUR_GOOGLE_DRIVE_FILE_ID
```

### 6. Fix embedding paths (REQUIRED)

```bash
python fix_embedding_paths.py
```

### 7. Launch the app

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## Usage

### Starting the Application

```bash
# Navigate to project directory
cd x-ray-transparency-lab

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set environment variable (Mac users with conda)
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the app
python -m streamlit run app.py

# Or use the convenience script
./run_app.sh
```

The app will open in your browser at `http://localhost:8501`

### Using the Demo

1. **Upload an X-ray image** or check "Use sample image instead"
   - Supported formats: JPG, JPEG, PNG
   - Recommended: PA-view chest X-rays
   - Sample images are available in the dropdown

2. **Click "Run AI Analysis"**
   - Processing takes 30-60 seconds
   - The model will analyze the image using multiple methods

3. **View the Results:**

   **Prediction Summary:**
   - Shows classification (NORMAL or PNEUMONIA)
   - Displays confidence percentage
   - Provides quick interpretation

   **Detailed Report:**
   - Click "View Detailed Report" to expand
   - Explains the model's reasoning
   - Describes visual patterns detected
   - Compares with similar training cases

4. **Explore Multi-View Explanations:**

   **Original Tab:**
   - Your uploaded X-ray image

   **Grad-CAM Tab:**
   - Red highlights show where the model focused
   - Darker red = stronger attention
   - Helps identify key diagnostic regions

   **Occlusion Tab:**
   - Shows which areas are most important
   - Red = high impact on prediction
   - Validates Grad-CAM findings

   **Similar Cases Tab:**
   - 6 most similar training images
   - Shows their labels and similarity scores
   - Helps understand model reasoning through examples

### Tips for Best Results

- **Use clear, high-quality X-rays**: Blurry or low-resolution images may not work well
- **PA (posterior-anterior) view recommended**: The model was trained on PA chest X-rays
- **Try multiple images**: Compare how the model performs on different cases
- **Read the detailed report**: Provides context beyond just the prediction
- **Check similar cases**: See what training images the model compared your X-ray to

### Important Notes

⚠️ **This is a demonstration tool for educational purposes only**
- NOT for clinical diagnosis
- NOT FDA approved or clinically validated
- Always consult qualified healthcare professionals
- The model may make errors - use critically

### Troubleshooting

**Connection Error:**
```bash
# Make sure to set the OpenMP environment variable
export KMP_DUPLICATE_LIB_OK=TRUE
python -m streamlit run app.py
```

**Slow Performance:**
- First analysis is slower (model loading)
- Subsequent analyses are faster
- Consider using GPU if available

**"Missing files" error:**
```bash
python diagnose_setup.py  # Check what's missing
python fix_embedding_paths.py  # Fix path issues
```

## Project Structure

- `src/train_model.py` - Model training script
- `src/generate_embeddings.py` - Creates feature embeddings for training set
- `src/explanations.py` - Grad-CAM, occlusion, similarity search
- `src/report_generator.py` - Natural language report generation
- `app.py` - Streamlit web interface

## Requirements

- Python 3.8+
- PyTorch
- Streamlit
- See `requirements.txt` for full list

## Citation

Dataset: Kermany, D., Zhang, K., & Goldbaum, M. (2018). Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. Mendeley Data, v2.

## License

MIT License