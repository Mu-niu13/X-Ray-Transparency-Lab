# Complete Setup Guide - Step by Step

Follow these steps in order to get the X-Ray Transparency Lab running.

## Prerequisites

- Python 3.8 or higher
- Internet connection
- ~6 GB free disk space

---

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/x-ray-transparency-lab.git
cd x-ray-transparency-lab
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env file with your credentials
# You need to add:
# - Kaggle credentials (for dataset download)
# - Google Drive file ID (for model download)
```

**Edit `.env` file:**
```bash
# Use nano, vim, or any text editor
nano .env
```

Add your credentials:
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
GDRIVE_MODEL_ID=your_file_id
```

**How to get credentials:**

**Kaggle:**
1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token"
3. Open downloaded `kaggle.json`
4. Copy username and key to `.env`

**Google Drive File ID:**
1. Ask the project maintainer for the file ID
2. Or upload your own trained model (see COLAB_TRAINING_GUIDE.md)
3. Share link looks like: `https://drive.google.com/file/d/1ABC123XYZ/view`
4. File ID is: `1ABC123XYZ`

### 5. Load Environment Variables

```bash
# On Linux/Mac:
export $(cat .env | xargs)

# On Windows (PowerShell):
Get-Content .env | ForEach-Object {
    $name, $value = $_.split('=')
    Set-Item -Path env:$name -Value $value
}

# On Windows (Command Prompt):
for /f "delims=" %i in (.env) do set %i
```

### 6. Download Dataset (~5.3 GB)

```bash
python download_data.py
```

**What happens:**
- Downloads Chest X-Ray Pneumonia dataset from Kaggle
- Extracts to `data/chest_xray/`
- Takes 5-10 minutes depending on internet speed

**Expected output:**
```
✅ Dataset downloaded and extracted!
📈 Dataset Statistics:
  Training:
    - Normal: 1341 images
    - Pneumonia: 3875 images
  Testing:
    - Normal: 234 images
    - Pneumonia: 390 images
```

### 7. Download Pre-trained Model (~142 MB)

```bash
python download_trained_model.py
```

**Alternative methods if environment variable doesn't work:**

**Method A: Command line argument**
```bash
python download_trained_model.py --file_id YOUR_FILE_ID
```

**Method B: Edit the file directly**
```python
# Open download_trained_model.py
# Find this line:
FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
# Replace with:
FILE_ID = "your_actual_file_id"
```

**What happens:**
- Downloads `trained_model.zip` from Google Drive
- Extracts `models/` and `embeddings/` folders
- Takes 2-5 minutes

**Expected output:**
```
✅ Download complete! (142.3 MB)
📦 Extracting files...
✅ Extraction complete!
🔍 Verifying files...
  ✅ Model weights: models/pneumonia_classifier.pth (87.2 MB)
  ✅ Feature embeddings: embeddings/embeddings.npy (10.4 MB)
  ✅ Training labels: embeddings/labels.npy (0.0 MB)
  ✅ Image paths: embeddings/paths.pkl (0.2 MB)
  ✅ Search index: embeddings/similarity_index.faiss (10.4 MB)
```

### 8. Fix Embedding Paths (CRITICAL!)

```bash
python fix_embedding_paths.py
```

**Why is this needed?**
The model was trained on Google Colab with different file paths. This script updates the paths to match your local setup.

**Expected output:**
```
📂 Found 5216 paths
Example old path: /content/data/chest_xray/train/NORMAL/...
💾 Backing up original to embeddings/paths.pkl.backup
✅ Updated 5216 paths
Example new path: data/chest_xray/train/NORMAL/...
✅ All sample paths verified!
```

### 9. Launch the Web Application

```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.x:8501
```

The app will automatically open in your default browser!

---

## Verification Checklist

Before running the app, verify these files exist:

```bash
# Check dataset
ls data/chest_xray/train/NORMAL/ | head -5
ls data/chest_xray/train/PNEUMONIA/ | head -5

# Check model files
ls -lh models/
ls -lh embeddings/

# Expected output:
# models/pneumonia_classifier.pth (~87 MB)
# embeddings/embeddings.npy (~10 MB)
# embeddings/labels.npy (~20 KB)
# embeddings/paths.pkl (~200 KB)
# embeddings/similarity_index.faiss (~10 MB)
```

---

## Troubleshooting

### Dataset Download Issues

**Problem: "Kaggle API credentials not found"**
```bash
# Solution: Verify .env file
cat .env | grep KAGGLE

# Re-export variables
export $(cat .env | xargs)
```

**Problem: "Dataset already exists"**
```bash
# Solution: Force re-download if needed
python download_data.py --force
```

### Model Download Issues

**Problem: "GOOGLE DRIVE FILE ID NOT SET"**
```bash
# Solution 1: Set environment variable
export GDRIVE_MODEL_ID='your_file_id'

# Solution 2: Use command line
python download_trained_model.py --file_id your_file_id

# Solution 3: Edit the file directly
nano download_trained_model.py
# Update FILE_ID variable
```

**Problem: Download is very slow**
- Google Drive can be slow for large files
- Try downloading manually and extracting:
  ```bash
  # Download from: https://drive.google.com/file/d/YOUR_FILE_ID/view
  # Then:
  unzip ~/Downloads/trained_model.zip
  python fix_embedding_paths.py
  ```

### Path Fixing Issues

**Problem: "Some files missing, re-downloading..."**
- Make sure you downloaded the dataset first
- Check if `data/chest_xray/train/` exists
- Run: `python download_data.py`

**Problem: "Warning: X/10 sample paths don't exist locally"**
```bash
# Verify dataset location
ls data/chest_xray/train/

# If wrong location, specify it:
python fix_embedding_paths.py --data_dir path/to/your/data
```

### Streamlit Issues

**Problem: "ModuleNotFoundError"**
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

**Problem: "Port 8501 is already in use"**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

---

## Quick Commands Reference

```bash
# Full setup (run in order)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
export $(cat .env | xargs)
python download_data.py
python download_trained_model.py
python fix_embedding_paths.py
streamlit run app.py

# Restart everything
deactivate
source venv/bin/activate
export $(cat .env | xargs)
streamlit run app.py
```

---

## Alternative: Train Model Yourself

If you prefer to train the model yourself instead of downloading:

1. Skip step 7 (download_trained_model.py)
2. Follow COLAB_TRAINING_GUIDE.md
3. Train on Google Colab (30 minutes with free GPU)
4. Download your trained model
5. Continue from step 8 (fix_embedding_paths.py)

---

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Verify all files exist (see Verification Checklist)
3. Check GitHub Issues for similar problems
4. Create a new issue with:
   - Error message
   - Output of `ls -R` showing your directory structure
   - Python version: `python --version`
   - OS information

---

## Success! 🎉

Once the app is running:

1. **Upload an X-ray image** or select a sample
2. Click **"Run AI Analysis"**
3. **Explore** the multi-view explanations:
   - Prediction results
   - Grad-CAM heatmap
   - Occlusion sensitivity
   - Similar training cases
4. **Read** the generated report

Enjoy exploring explainable AI for medical imaging!