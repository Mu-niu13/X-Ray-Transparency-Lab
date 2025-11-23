# X-Ray Transparency Lab

Multi-view explainable AI system for pneumonia detection in chest X-rays.

## Overview

This project provides an interactive web interface for understanding AI-based pneumonia diagnosis through multiple explanation methods:

- **Grad-CAM**: Visual attention heatmaps
- **Occlusion Sensitivity**: Perturbation-based importance
- **Example-Based**: Similar training cases
- **Descriptive Reports**: Natural language explanations

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/x-ray-transparency-lab.git
cd x-ray-transparency-lab
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the Chest X-Ray Pneumonia dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Extract it to the `data/` folder so you have:
```
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 4. Train the model

```bash
python src/train_model.py --epochs 10 --batch_size 32
```

This will save the trained model to `models/pneumonia_classifier.pth`

### 5. Generate embeddings for similarity search

```bash
python src/generate_embeddings.py
```

This creates the embedding index in `embeddings/`

### 6. Run the web application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

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