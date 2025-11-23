"""
Download and setup the Chest X-Ray Pneumonia dataset from Kaggle.
Requires Kaggle API credentials.
"""

import os
import zipfile
import argparse
from pathlib import Path


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up"""
    # Check for environment variables first
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    
    if kaggle_username and kaggle_key:
        print("✅ Using Kaggle credentials from environment variables")
        return True
    
    # Fallback to kaggle.json file
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Using Kaggle credentials from ~/.kaggle/kaggle.json")
        return True
    
    print("❌ Kaggle API credentials not found!")
    print("\n📋 Setup Instructions (Choose ONE method):")
    print("\n**Method 1: Environment Variables (Recommended)**")
    print("Set these environment variables:")
    print("  KAGGLE_USERNAME=your_kaggle_username")
    print("  KAGGLE_KEY=your_kaggle_api_key")
    print("\nTo get your credentials:")
    print("  1. Go to https://www.kaggle.com/settings/account")
    print("  2. Scroll to 'API' section")
    print("  3. Click 'Create New Token' (downloads kaggle.json)")
    print("  4. Open kaggle.json to find your username and key")
    print("\nOn Linux/Mac (temporary):")
    print("  export KAGGLE_USERNAME='your_username'")
    print("  export KAGGLE_KEY='your_key'")
    print("\nOn Windows (temporary):")
    print("  set KAGGLE_USERNAME=your_username")
    print("  set KAGGLE_KEY=your_key")
    print("\n**Method 2: Configuration File**")
    print("  Move kaggle.json to ~/.kaggle/ (or C:\\Users\\YourUsername\\.kaggle\\ on Windows)")
    print("  On Linux/Mac, run: chmod 600 ~/.kaggle/kaggle.json")
    
    return False


def download_dataset(data_dir='data', force=False):
    """Download dataset using Kaggle API"""
    
    # Check credentials
    if not check_kaggle_credentials():
        return False
    
    # Import kaggle (only if credentials exist)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Kaggle package not installed!")
        print("Install it with: pip install kaggle")
        return False
    
    # Setup paths
    data_path = Path(data_dir)
    chest_xray_path = data_path / 'chest_xray'
    zip_path = data_path / 'chest-xray-pneumonia.zip'
    
    # Check if already exists
    if chest_xray_path.exists() and not force:
        print(f"✅ Dataset already exists at {chest_xray_path}")
        print("Use --force to re-download")
        return True
    
    # Create data directory
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize Kaggle API
    print("🔐 Authenticating with Kaggle...")
    api = KaggleApi()
    api.authenticate()
    
    # Download dataset
    print("⬇️  Downloading dataset (this will take several minutes, ~5.3 GB)...")
    print("Dataset: paultimothymooney/chest-xray-pneumonia")
    
    try:
        api.dataset_download_files(
            'paultimothymooney/chest-xray-pneumonia',
            path=data_path,
            unzip=False
        )
        print("✅ Download complete!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False
    
    # Unzip
    print("📦 Extracting files...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("✅ Extraction complete!")
        
        # Remove zip file to save space
        zip_path.unlink()
        print("🗑️  Cleaned up zip file")
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False
    
    # Verify structure
    print("\n📊 Verifying dataset structure...")
    
    required_dirs = [
        chest_xray_path / 'train' / 'NORMAL',
        chest_xray_path / 'train' / 'PNEUMONIA',
        chest_xray_path / 'test' / 'NORMAL',
        chest_xray_path / 'test' / 'PNEUMONIA',
        chest_xray_path / 'val' / 'NORMAL',
        chest_xray_path / 'val' / 'PNEUMONIA',
    ]
    
    all_exist = all(d.exists() for d in required_dirs)
    
    if all_exist:
        # Count images
        train_normal = len(list((chest_xray_path / 'train' / 'NORMAL').glob('*.jpeg')))
        train_pneumonia = len(list((chest_xray_path / 'train' / 'PNEUMONIA').glob('*.jpeg')))
        test_normal = len(list((chest_xray_path / 'test' / 'NORMAL').glob('*.jpeg')))
        test_pneumonia = len(list((chest_xray_path / 'test' / 'PNEUMONIA').glob('*.jpeg')))
        
        print("✅ Dataset structure verified!")
        print(f"\n📈 Dataset Statistics:")
        print(f"  Training:")
        print(f"    - Normal: {train_normal} images")
        print(f"    - Pneumonia: {train_pneumonia} images")
        print(f"  Testing:")
        print(f"    - Normal: {test_normal} images")
        print(f"    - Pneumonia: {test_pneumonia} images")
        print(f"\n🎉 Setup complete! You can now run training.")
        return True
    else:
        print("❌ Dataset structure incomplete")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download Chest X-Ray Pneumonia dataset from Kaggle'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory to download data to (default: data/)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📥 Chest X-Ray Pneumonia Dataset Downloader")
    print("=" * 60)
    
    success = download_dataset(args.data_dir, args.force)
    
    if success:
        print("\n✅ All done! Next steps:")
        print("  1. Train the model: python src/train_model.py")
        print("  2. Generate embeddings: python src/generate_embeddings.py")
        print("  3. Launch app: streamlit run app.py")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")


if __name__ == '__main__':
    main()