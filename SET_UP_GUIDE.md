# Local Setup Guide

Complete guide to run X-Ray Transparency Lab on your local machine.

**Note**: If you just want to use the app, visit the deployed version:
**https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/**

---

## Prerequisites

- Python 3.10 or 3.11
- 4GB RAM minimum
- 2GB free disk space
- Internet connection (for model download)

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Mu-niu13/X-Ray-Transparency-Lab.git
cd X-Ray-Transparency-Lab

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
# Mac users with OpenMP issues: ./run_app.sh
```

The app will automatically download model files (~140MB) on first run.

---

## Detailed Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/Mu-niu13/X-Ray-Transparency-Lab.git
cd X-Ray-Transparency-Lab
```

### Step 2: Python Environment

**Check Python version**:
```bash
python --version
# Should be 3.10 or 3.11
```

**Create virtual environment**:
```bash
# Create venv
python -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies installed**:
- PyTorch & TorchVision (deep learning)
- Streamlit (web interface)
- FAISS (similarity search)
- Hugging Face Hub (model download)
- OpenCV, NumPy, Pillow (image processing)

### Step 4: Configure Model Download

The app downloads model files automatically from Hugging Face on first run.

**Model Repository**: https://huggingface.co/Mu-niu13/xray-pneumonia-model

**Files downloaded** (~140MB total):
- `models/pneumonia_classifier.pth` (87MB)
- `embeddings/embeddings.npy` (10MB)
- `embeddings/labels.npy` (20KB)
- `embeddings/paths.pkl` (200KB)
- `embeddings/similarity_index.faiss` (10MB)

### Step 5: Run the App

**Standard method**:
```bash
streamlit run app.py
```

**Mac users** (if you get OpenMP error):
```bash
chmod +x run_app.sh
./run_app.sh
```

**Windows users**:
```cmd
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## First Run

### What Happens

1. **Model Download** (first time only, 2-3 minutes)
   ```
   📥 Downloading model files from Hugging Face...
   ⬇️ models/pneumonia_classifier.pth
   ⬇️ embeddings/embeddings.npy
   ⬇️ embeddings/labels.npy
   ⬇️ embeddings/paths.pkl
   ⬇️ embeddings/similarity_index.faiss
   ✅ Model files ready!
   ```

2. **App Launches**
   - Interface loads
   - Ready to analyze images

### Verify Setup

1. Check "Use sample image instead"
2. Select a sample
3. Click "Run AI Analysis"
4. Verify all features work:
   - ✅ Prediction
   - ✅ Confidence score
   - ✅ Grad-CAM heatmap
   - ✅ Occlusion map
   - ✅ Similar cases
   - ✅ Report

---

## Troubleshooting

### Python Version Issues

**Problem**: Wrong Python version
```bash
python --version
# If not 3.10 or 3.11
```

**Solution**:
```bash
# Install Python 3.10
# macOS with Homebrew:
brew install python@3.10

# Ubuntu/Debian:
sudo apt install python3.10

# Windows: Download from python.org

# Then create venv with specific version
python3.10 -m venv venv
```

### OpenMP Library Error (Mac)

**Error message**:
```
OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized.
```

**Solution 1** (Quick fix):
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py
```

**Solution 2** (Use provided script):
```bash
chmod +x run_app.sh
./run_app.sh
```

**Solution 3** (Permanent fix):
```bash
# Add to ~/.zshrc or ~/.bash_profile
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc
source ~/.zshrc
```

### Model Download Fails

**Problem**: Cannot download from Hugging Face

**Check**:
1. Internet connection
2. Hugging Face is accessible: https://huggingface.co/Mu-niu13/xray-pneumonia-model

**Manual download** (if automatic fails):
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
import os

os.makedirs('models', exist_ok=True)
os.makedirs('embeddings', exist_ok=True)

files = [
    'models/pneumonia_classifier.pth',
    'embeddings/embeddings.npy',
    'embeddings/labels.npy',
    'embeddings/paths.pkl',
    'embeddings/similarity_index.faiss'
]

for file in files:
    print(f'Downloading {file}...')
    hf_hub_download(
        repo_id='Mu-niu13/xray-pneumonia-model',
        filename=file,
        local_dir='.',
        local_dir_use_symlinks=False
    )
"
```

### Port Already in Use

**Error**: `Port 8501 is already in use`

**Solution**:
```bash
# Use different port
streamlit run app.py --server.port 8502

# Or kill existing process
# macOS/Linux:
lsof -ti:8501 | xargs kill -9

# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Out of Memory

**Error**: `RuntimeError: CUDA out of memory` or system freezes

**Solution**:
1. Close other applications
2. Process smaller images
3. Use CPU instead of GPU (already default)

### Module Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should show venv path

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# If still fails, check specific module
pip install streamlit torch torchvision faiss-cpu
```

---

## Development Setup

### For Contributing or Training

If you want to train your own model or contribute:

#### 1. Download Training Data

```bash
# Setup Kaggle API
pip install kaggle

# Get API key from kaggle.com/settings
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

#### 2. Train Model

```bash
python src/train_model.py \
    --epochs 10 \
    --batch_size 32 \
    --lr 0.001
```

**Note**: Training takes 20-30 minutes on GPU, 2-3 hours on CPU

#### 3. Generate Embeddings

```bash
python src/generate_embeddings.py \
    --model_path models/pneumonia_classifier.pth \
    --data_dir data/chest_xray/train
```

#### 4. Upload to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login
python upload_to_huggingface.py
```

See [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) for GPU training on Google Colab.

---

## Project Structure

```
X-Ray-Transparency-Lab/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── run_app.sh                     # Launch script (Mac)
│
├── src/                           # Source code
│   ├── explanations.py            # XAI methods
│   ├── report_generator.py       # Report generation
│   ├── train_model.py            # Model training
│   └── generate_embeddings.py    # Embedding generation
│
├── models/                        # Downloaded from HF
│   └── pneumonia_classifier.pth
│
├── embeddings/                    # Downloaded from HF
│   ├── embeddings.npy
│   ├── labels.npy
│   ├── paths.pkl
│   └── similarity_index.faiss
│
├── data/                          # Only for training
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
│
├── docs/                          # Documentation
│   ├── SETUP_GUIDE.md
│   ├── DEPLOYMENT.md
│   └── COLAB_TRAINING_GUIDE.md
│
└── upload_to_huggingface.py      # HF upload script
```

---

## Configuration

### Streamlit Configuration

Create `.streamlit/config.toml` for custom settings:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200  # MB
```

### Environment Variables

Optional `.env` file:
```bash
# Only needed for development
PYTHONPATH=src
STREAMLIT_SERVER_PORT=8501
```

---

## System Requirements

### Minimum
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 2GB free
- **OS**: macOS 10.14+, Windows 10+, Ubuntu 18.04+
- **Python**: 3.10 or 3.11

### Recommended
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 5GB free
- **GPU**: Optional (CUDA 11.7+ for faster processing)

### Browser
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Performance Tips

### Faster Model Loading
```bash
# Pre-download models
python -c "
from huggingface_hub import hf_hub_download
files = ['models/pneumonia_classifier.pth', 'embeddings/embeddings.npy']
for f in files:
    hf_hub_download('Mu-niu13/xray-pneumonia-model', f, local_dir='.')
"
```

### Reduce Processing Time
Edit `src/explanations.py`:
```python
# Line ~180 - reduce occlusion stride
occlusion = OcclusionSensitivity(model, patch_size=32, stride=32)  # was 16
```

### Clear Cache
```bash
# Clear Streamlit cache
streamlit cache clear

# Or manually delete
rm -rf ~/.streamlit/cache/
rm -rf .cache/
```

---

## Uninstall

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf X-Ray-Transparency-Lab

# Remove Python cache (optional)
rm -rf ~/.cache/huggingface
```

---

## Next Steps

### Local Development
- Modify `app.py` for UI changes
- Update `src/explanations.py` for new XAI methods
- Edit `src/report_generator.py` for report customization

### Training Your Own Model
- Follow [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
- Use your own dataset
- Upload to your Hugging Face account

### Deploying Your Version
- Follow [DEPLOYMENT.md](DEPLOYMENT.md)
- Deploy to Streamlit Cloud
- Share your custom version

---

## Support

### Documentation
- **Setup**: This file
- **Deployment**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Training**: [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
- **Live Demo**: https://mu-niu13-x-ray-transparency-lab-app-y74grj.streamlit.app/

### Community
- **GitHub Issues**: https://github.com/Mu-niu13/X-Ray-Transparency-Lab/issues
- **Discussions**: https://github.com/Mu-niu13/X-Ray-Transparency-Lab/discussions
- **Streamlit Forum**: https://discuss.streamlit.io/

---

## Checklist

Before running:
- [ ] Python 3.10 or 3.11 installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Internet connection active
- [ ] 2GB free disk space

First run:
- [ ] App starts without errors
- [ ] Model downloads successfully
- [ ] Sample images work
- [ ] Upload feature works
- [ ] All explanation views display
- [ ] Report generates
