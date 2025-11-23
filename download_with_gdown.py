"""
Alternative download script using gdown library (more reliable for Google Drive).
This is a simpler and more robust method.

Install first: pip install gdown
"""

import os
import zipfile
import subprocess
import sys
from pathlib import Path


def install_gdown():
    """Install gdown if not available"""
    try:
        import gdown
        return True
    except ImportError:
        print("📦 Installing gdown library...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        print("✅ gdown installed!")
        return True


def download_with_gdown(file_id, output_path):
    """Download using gdown library"""
    import gdown
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print(f"⬇️  Downloading from: {url}")
    gdown.download(url, str(output_path), quiet=False)
    
    return output_path.exists()


def extract_and_verify(zip_path):
    """Extract zip and verify contents"""
    print(f"\n📦 Extracting {zip_path.name}...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print(f"   Found {len(file_list)} files")
            zip_ref.extractall('.')
        
        print("✅ Extraction complete!")
        
        # Remove zip
        zip_path.unlink()
        print(f"🗑️  Cleaned up {zip_path.name}")
        
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False


def main():
    """Main download function"""
    
    # Get file ID from environment or set it here
    FILE_ID = os.environ.get('GDRIVE_MODEL_ID', None)
    
    if not FILE_ID:
        FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"  # Replace with your file ID
    
    if FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID" or not FILE_ID:
        print("=" * 70)
        print("⚠️  FILE ID NOT SET")
        print("=" * 70)
        print("\nUsage:")
        print("  1. Set environment variable:")
        print("     export GDRIVE_MODEL_ID='your_file_id'")
        print("     python download_with_gdown.py")
        print("\n  2. Or edit this file and set FILE_ID")
        print("\n  3. Or use command line:")
        print("     python download_with_gdown.py YOUR_FILE_ID")
        return
    
    # Install gdown if needed
    install_gdown()
    
    output_path = Path('trained_model.zip')
    
    print("=" * 70)
    print("📥 Downloading Pre-trained Model (using gdown)")
    print("=" * 70)
    print(f"File ID: {FILE_ID}")
    
    try:
        # Download
        success = download_with_gdown(FILE_ID, output_path)
        
        if not success or not output_path.exists():
            print("\n❌ Download failed!")
            print("\nPlease check:")
            print("  1. File is shared: https://drive.google.com/file/d/{}/view".format(FILE_ID))
            print("  2. Share settings: 'Anyone with the link can view'")
            return
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Download complete! ({file_size:.1f} MB)")
        
        # Verify size
        if file_size < 1:
            print("⚠️  File is suspiciously small. Check if download worked correctly.")
            return
        
        # Extract
        if extract_and_verify(output_path):
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            print("\n📋 Next steps:")
            print("  1. python download_data.py")
            print("  2. python fix_embedding_paths.py")
            print("  3. streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTry manual download:")
        print(f"  https://drive.google.com/file/d/{FILE_ID}/view")


if __name__ == '__main__':
    # Allow file ID as command line argument
    if len(sys.argv) > 1:
        os.environ['GDRIVE_MODEL_ID'] = sys.argv[1]
    
    main()