# X-Ray Transparency Lab

Multi-view explainable AI system for pneumonia detection in chest X-rays.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/)

** Live Demo:** https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/

## Overview

This project provides an interactive web interface for understanding AI-based pneumonia diagnosis through multiple explanation methods:

- **Grad-CAM**: Visual attention heatmaps showing where the model focuses
- **Occlusion Sensitivity**: Perturbation-based importance revealing critical regions
- **Descriptive Reports**: Natural language explanations of the model's reasoning


## Features

**Diagnosis Explainer**: Two complementary XAI methods working together  
**Interactive Interface**: Upload images and get instant AI analysis  
**Natural Language Reports**: Human-readable explanations of predictions  
**Educational Tool**: Learn how AI makes medical diagnoses  

## Quick Start

### Try the Live App

Visit the deployed app: https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/

No installation required! Just upload a chest X-ray and explore the explanations.

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Mu-niu13/X-Ray-Transparency-Lab.git
cd X-Ray-Transparency-Lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Note for Mac users:** If you get an OpenMP error:
```bash
chmod +x run_app.sh
./run_app.sh
```

## Project Structure

```
X-Ray-Transparency-Lab/
├── app.py                      # Main Streamlit application
├── src/
│   ├── explanations.py         # Grad-CAM, Occlusion, Similarity Search
│   ├── report_generator.py    # Natural language report generation
│   ├── train_model.py          # Model training script
│   └── generate_embeddings.py # Embedding generation for similarity
├── requirements.txt            # Python dependencies
├── upload_to_huggingface.py   # Script to upload model to HF Hub
├── download_data.py           # Script to download data
├── download_trained_model.py  # Script to download trained model
├── fix_embedding_paths.py     # fix embedding locally
├── model_train.ipynb          # model training
├── DEPLOYMENT.md              # Deployment guide
├── README.md                  # Intro

Model files are hosted on Hugging Face Hub and downloaded automatically.
```

## How It Works

### 1. Model Architecture
- **Base Model**: ResNet-18 with transfer learning
- **Training Data**: 5,216 chest X-ray images from Kaggle dataset
- **Classes**: Binary classification (Normal vs Pneumonia)
- **Performance**: ~99% validation accuracy

### 2. Explanation Methods

**Grad-CAM (Gradient-weighted Class Activation Mapping)**
- Visualizes neural network attention
- Shows which image regions influenced the prediction
- Uses gradient information from the last convolutional layer

**Occlusion Sensitivity**
- Tests importance by systematically masking image regions
- Measures prediction change when areas are hidden
- Confirms which regions are truly critical

### 3. Technical Stack
- **Framework**: PyTorch, Streamlit
- **XAI Methods**: Grad-CAM, Occlusion Analysis
- **Deployment**: Streamlit Cloud, Hugging Face Hub
- **Model Hosting**: Hugging Face Model Hub

## Usage

### Web Interface

1. **Upload Image**: Choose a chest X-ray (JPEG/PNG) or select a sample
2. **Run Analysis**: Click "Run AI Analysis" button
3. **View Results**: 
   - Prediction with confidence score
   - Quick summary
   - Detailed report
4. **Explore Explanations**:
   - **Original**: Your input image
   - **Grad-CAM**: Attention heatmap overlay
   - **Occlusion**: Importance map

### Understanding the Results

**High Confidence (>90%)**
- Multiple methods agree
- Clear visual patterns
- Similar training cases match prediction

**Moderate Confidence (60-90%)**
- Some uncertainty in model
- May need clinical review
- Check similar cases for context

**Low Confidence (<60%)**
- Model unsure
- Consult medical professional
- Visual patterns may be ambiguous

## Development

### Training Your Own Model

If you want to train the model yourself:

1. **Setup Kaggle API**:
   ```bash
   # Get API credentials from kaggle.com/settings
   pip install kaggle
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   ```

2. **Train Model & Generate Embeddings**:
   ```bash
   # run model_train.ipynb
   ```

3. **Upload to Hugging Face**:
   ```bash
   huggingface-cli login
   python upload_to_huggingface.py
   ```

### Deployment

Want to deploy your own instance?

1. Fork this repository
2. Follow [DEPLOYMENT.md](DEPLOYMENT.md) for step-by-step instructions
3. Deploy to Streamlit Cloud

## Model Access

Pre-trained model is available on Hugging Face:
- **Repository**: [Mu-niu13/xray-pneumonia-model](https://huggingface.co/Mu-niu13/xray-pneumonia-model)
- **Files**: Model weights, embeddings, FAISS index
- **Size**: ~140 MB total

## Dataset

**Chest X-Ray Images (Pneumonia)**
- **Source**: Kermany et al. (2018)
- **Link**: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: 5,856 images
- **Classes**: Normal (1,583), Pneumonia (4,273)
- **Institution**: Guangzhou Women and Children's Medical Center

**Citation**:
```
Kermany, D., Zhang, K., & Goldbaum, M. (2018). 
Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification. 
Mendeley Data, v2.
```

## Important Disclaimers

**NOT FOR CLINICAL USE**

This is a **research and educational tool** only:
- NOT FDA approved or clinically validated
- NOT intended for medical diagnosis
- May produce incorrect results
- Always consult qualified healthcare professionals
- Educational demonstration of XAI techniques

The model may make errors. Medical decisions should ONLY be made by licensed medical professionals with appropriate training and access to complete patient information.

## Limitations

- **Training Data**: Limited to specific dataset, may not generalize
- **Image Quality**: Performance depends on X-ray quality and positioning
- **Explanation Accuracy**: XAI methods approximate model reasoning
- **Computational Cost**: Occlusion analysis is slower for large images
- **Binary Classification**: Only detects pneumonia presence, not type/severity

## Troubleshooting

### Common Issues

**App won't start locally:**
```bash
# Mac users - OpenMP conflict
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py
```

**Model download fails:**
- Check internet connection
- Verify Hugging Face repository is accessible
- Clear cache: `rm -rf .cache/`

**Out of memory:**
- Use smaller batch size
- Process images sequentially
- Close other applications

See [DEPLOYMENT.md](DEPLOYMENT.md) for more troubleshooting tips.

## Architecture

```
┌─────────────────┐
│  Input Image    │
└───────┬─────────┘
        │
  ┌─────▼──────┐
  │  ResNet-18 │
  └─────┬──────┘
        │
 ┌──────▼────────────┐
 │ Grad-CAM Heatmap  │
 └──────┬────────────┘
        │
 ┌──────▼────────────┐
 │ Occlusion Map     │
 └──────┬────────────┘
        │
 ┌──────▼────────────┐
 │ Report Generator  │
 └───────────────────┘

```

## Resources

- **Live Demo**: https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/
- **Model**: https://huggingface.co/Mu-niu13/xray-pneumonia-model
- **Dataset**: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## References

**XAI Methods:**
- Selvaraju et al. (2017) - "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- Zeiler & Fergus (2014) - "Visualizing and Understanding Convolutional Networks"

**Dataset:**
- Kermany et al. (2018) - "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset**: Kermany et al., Kaggle community
- **Framework**: PyTorch, Streamlit teams
- **Deployment**: Hugging Face Hub, Streamlit Cloud
- **Inspiration**: Explainable AI research community

## Contact

- **Author**: Mu Niu
- **GitHub**: [@Mu-niu13](https://github.com/Mu-niu13)
- **Project**: [X-Ray-Transparency-Lab](https://github.com/Mu-niu13/X-Ray-Transparency-Lab)

---