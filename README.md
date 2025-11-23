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

1. **Upload an X-ray image** or select from sample cases
2. Click "**Run AI Analysis**"
3. View the prediction and explore multiple explanation views
4. Read the generated report

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