"""
Upload trained model to Hugging Face Hub for Streamlit Cloud deployment.

Setup:
1. Create account at https://huggingface.co
2. Create access token: Settings → Access Tokens → New token (READ role is enough)
3. Install: pip install huggingface_hub
4. Login: huggingface-cli login (paste your token)
5. Run this script: python upload_to_huggingface.py
"""

from huggingface_hub import HfApi, create_repo, login
from pathlib import Path
import os
import sys

def check_files():
    """Check if all required files exist"""
    files = [
        "models/pneumonia_classifier.pth",
        "embeddings/embeddings.npy",
        "embeddings/labels.npy",
        "embeddings/paths.pkl",
        "embeddings/similarity_index.faiss"
    ]
    
    missing = []
    for file in files:
        if not Path(file).exists():
            missing.append(file)
        else:
            size = Path(file).stat().st_size / (1024 * 1024)
            print(f"✅ {file} ({size:.1f} MB)")
    
    if missing:
        print(f"\n❌ Missing files:")
        for file in missing:
            print(f"   - {file}")
        print("\nPlease run these first:")
        print("  1. python src/train_model.py")
        print("  2. python src/generate_embeddings.py")
        return False
    
    return True

def upload_to_huggingface(
    repo_name="xray-pneumonia-model",
    username=None,
    make_private=False
):
    """Upload model files to Hugging Face Hub"""
    
    print("=" * 70)
    print("📤 Uploading to Hugging Face Hub")
    print("=" * 70)
    
    # Check files
    print("\n📋 Checking files...")
    if not check_files():
        return False
    
    # Initialize API
    api = HfApi()
    
    # Get username
    if username is None:
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"\n✅ Logged in as: {username}")
        except Exception as e:
            print(f"\n❌ Not logged in to Hugging Face!")
            print("\nPlease run: huggingface-cli login")
            print("Get your token from: https://huggingface.co/settings/tokens")
            return False
    
    repo_id = f"{username}/{repo_name}"
    
    # Create repo
    print(f"\n📦 Creating repository: {repo_id}")
    try:
        url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=make_private
        )
        print(f"✅ Repository ready: {url}")
    except Exception as e:
        print(f"❌ Error creating repo: {e}")
        return False
    
    # Create README
    print("\n📝 Creating README...")
    readme_content = f"""---
license: mit
tags:
- medical
- pneumonia
- chest-xray
- explainable-ai
- pytorch
---

# Pneumonia Detection Model

Trained ResNet-18 model for pneumonia detection from chest X-rays with explainable AI features.

## Model Details

- **Architecture**: ResNet-18 with transfer learning
- **Task**: Binary classification (Normal vs Pneumonia)
- **Input**: 224x224 RGB chest X-ray images
- **Training Data**: Kaggle Chest X-Ray Pneumonia Dataset

## Files

- `models/pneumonia_classifier.pth`: Trained model weights (~87 MB)
- `embeddings/embeddings.npy`: Feature embeddings for similarity search
- `embeddings/similarity_index.faiss`: FAISS index for fast retrieval
- `embeddings/labels.npy`: Training labels
- `embeddings/paths.pkl`: Training image paths

## Usage

```python
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="models/pneumonia_classifier.pth"
)
```

## Streamlit App

This model is used in the X-Ray Transparency Lab Streamlit app with multi-view explainable AI.

**⚠️ Disclaimer**: This is a research/educational model and should NOT be used for clinical diagnosis.
"""
    
    try:
        with open('README.md', 'w') as f:
            f.write(readme_content)
        
        api.upload_file(
            path_or_fileobj='README.md',
            path_in_repo='README.md',
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ README.md uploaded")
        os.remove('README.md')
    except Exception as e:
        print(f"⚠️  Could not upload README: {e}")
    
    # Upload files
    files_to_upload = [
        "models/pneumonia_classifier.pth",
        "embeddings/embeddings.npy",
        "embeddings/labels.npy",
        "embeddings/paths.pkl",
        "embeddings/similarity_index.faiss"
    ]
    
    print(f"\n📤 Uploading {len(files_to_upload)} files...")
    print("This may take 5-10 minutes...\n")
    
    for i, file in enumerate(files_to_upload, 1):
        try:
            size = Path(file).stat().st_size / (1024 * 1024)
            print(f"[{i}/{len(files_to_upload)}] Uploading {file} ({size:.1f} MB)...")
            
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=repo_id,
                repo_type="model"
            )
            print(f"     ✅ Done\n")
        except Exception as e:
            print(f"     ❌ Error: {e}\n")
            return False
    
    # Success message
    print("=" * 70)
    print("🎉 SUCCESS!")
    print("=" * 70)
    print(f"\n✅ Model available at: https://huggingface.co/{repo_id}")
    print(f"\n📋 Next steps for Streamlit Cloud deployment:")
    print(f"\n1. Go to your Streamlit Cloud app → Settings → Secrets")
    print(f'2. Add this line:')
    print(f'   HUGGINGFACE_REPO = "{repo_id}"')
    print(f"\n3. Update requirements.txt to include:")
    print(f"   huggingface_hub>=0.19.0")
    print(f"\n4. Deploy!")
    print("=" * 70)
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Upload model to Hugging Face Hub')
    parser.add_argument('--repo_name', type=str, default='xray-pneumonia-model',
                       help='Repository name (default: xray-pneumonia-model)')
    parser.add_argument('--username', type=str, default=None,
                       help='Hugging Face username (optional, auto-detected)')
    parser.add_argument('--private', action='store_true',
                       help='Make repository private (default: public)')
    
    args = parser.parse_args()
    
    # Check if logged in
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
    except Exception:
        print("=" * 70)
        print("⚠️  Not logged in to Hugging Face")
        print("=" * 70)
        print("\n1. Create account: https://huggingface.co/join")
        print("2. Get token: https://huggingface.co/settings/tokens")
        print("3. Login: huggingface-cli login")
        print("\nThen run this script again.")
        return
    
    upload_to_huggingface(
        repo_name=args.repo_name,
        username=args.username,
        make_private=args.private
    )

if __name__ == '__main__':
    main()