"""
Download pre-trained model from Google Drive.
Upload your trained_model.zip to Google Drive and share with 'Anyone with the link'.
"""

import requests
import zipfile
import os
from pathlib import Path
from tqdm import tqdm
import time


def download_file_from_google_drive(file_id, destination):
    """Download large file from Google Drive with proper handling"""
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, "wb") as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=str(destination.name)) as pbar:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # No content-length header, download without progress bar
                print(f"Downloading {destination.name}...")
                downloaded = 0
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end='')
                print()
    
    # Google Drive download URL
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    # First request
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check for virus scan warning
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Check if we got HTML instead of file
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        # We got an HTML page, not the file - need alternative method
        print("⚠️  Standard download failed, trying alternative method...")
        
        # Alternative: Use the direct download link format
        alt_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
        response = session.get(alt_url, stream=True)
        
        # If still HTML, the file might not be properly shared
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            raise Exception(
                "Cannot download file. Please verify:\n"
                "  1. File is shared with 'Anyone with the link'\n"
                "  2. Link permissions are set to 'Viewer' or 'Editor'\n"
                "  3. File ID is correct"
            )
    
    save_response_content(response, destination)


def verify_zip_file(zip_path):
    """Verify the downloaded file is actually a valid zip"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Try to read the file list
            file_list = zip_ref.namelist()
            return True, file_list
    except zipfile.BadZipFile:
        return False, []


def extract_and_cleanup(zip_path, extract_to='.'):
    """Extract zip and remove zip file"""
    
    # First verify it's a valid zip
    is_valid, file_list = verify_zip_file(zip_path)
    
    if not is_valid:
        file_size = zip_path.stat().st_size
        print(f"\n❌ Downloaded file is not a valid zip file (size: {file_size} bytes)")
        
        if file_size < 10000:
            # Likely an error page
            print("\n📄 File content (first 500 chars):")
            with open(zip_path, 'r', errors='ignore') as f:
                print(f.read(500))
            print("\n⚠️  This is likely an HTML error page from Google Drive.")
        
        raise Exception("File is not a valid zip file")
    
    print(f"\n📦 Extracting {zip_path.name}...")
    print(f"   Found {len(file_list)} files in archive")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress bar
        total_size = sum(f.file_size for f in zip_ref.filelist)
        
        with tqdm(total=total_size, unit='B', unit_scale=True, 
                 desc="Extracting") as pbar:
            for file in zip_ref.filelist:
                zip_ref.extract(file, extract_to)
                pbar.update(file.file_size)
    
    print(f"✅ Extraction complete!")
    
    # Remove zip file
    zip_path.unlink()
    print(f"🗑️  Cleaned up {zip_path.name}")


def verify_files():
    """Verify all required files exist"""
    required_files = {
        'models/pneumonia_classifier.pth': 'Model weights',
        'embeddings/embeddings.npy': 'Feature embeddings',
        'embeddings/labels.npy': 'Training labels',
        'embeddings/paths.pkl': 'Image paths',
        'embeddings/similarity_index.faiss': 'Search index'
    }
    
    print("\n🔍 Verifying files...")
    all_exist = True
    
    for file_path, description in required_files.items():
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size / (1024 * 1024)
            print(f"  ✅ {description}: {file_path} ({size:.1f} MB)")
        else:
            print(f"  ❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist


def main():
    """
    Download pre-trained model from Google Drive
    
    SETUP INSTRUCTIONS:
    1. Upload your trained_model.zip to Google Drive
    2. Right-click → Share → Change to "Anyone with the link"
    3. Copy the share link
    4. Extract file ID from link:
       Link: https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing
       File ID: 1ABC123XYZ
    5. Set FILE_ID below or use environment variable GDRIVE_MODEL_ID
    """
    
    # Check environment variable first
    FILE_ID = os.environ.get('GDRIVE_MODEL_ID', None)
    
    # Or set it directly here (replace YOUR_FILE_ID with actual ID)
    if not FILE_ID:
        FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
    
    if FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID" or not FILE_ID:
        print("=" * 70)
        print("⚠️  GOOGLE DRIVE FILE ID NOT SET")
        print("=" * 70)
        print("\n📋 Setup Instructions:")
        print("\n1. Upload trained_model.zip to Google Drive")
        print("2. Right-click → Share → 'Anyone with the link'")
        print("3. Copy the share link")
        print("4. Extract the file ID:")
        print("   Link: https://drive.google.com/file/d/1ABC123XYZ/view?usp=sharing")
        print("   File ID: 1ABC123XYZ")
        print("\n5. Set the file ID using ONE of these methods:")
        print("\n   Method A: Environment Variable (Recommended)")
        print("   export GDRIVE_MODEL_ID='your_file_id'")
        print("   python download_trained_model.py")
        print("\n   Method B: Edit this file")
        print("   Open download_trained_model.py")
        print("   Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with your actual file ID")
        print("   python download_trained_model.py")
        print("\n   Method C: Command line argument")
        print("   python download_trained_model.py --file_id your_file_id")
        print("\n" + "=" * 70)
        return
    
    output_path = Path('trained_model.zip')
    
    # Check if already downloaded
    if Path('models/pneumonia_classifier.pth').exists():
        print("✅ Model files already exist!")
        if verify_files():
            print("\n🎉 All files present. You can run: streamlit run app.py")
            choice = input("\nRe-download anyway? (y/N): ").strip().lower()
            if choice != 'y':
                return
        else:
            print("\n⚠️  Some files missing, re-downloading...")
    
    print("=" * 70)
    print("📥 Downloading Pre-trained Model from Google Drive")
    print("=" * 70)
    print(f"\nFile ID: {FILE_ID}")
    print(f"Expected size: ~142 MB")
    print(f"Destination: {output_path}")
    
    try:
        # Download
        print("\n⬇️  Downloading from Google Drive...")
        download_file_from_google_drive(FILE_ID, output_path)
        
        if not output_path.exists():
            raise Exception("Download failed - file not created")
        
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Download complete! ({file_size:.1f} MB)")
        
        # Extract
        extract_and_cleanup(output_path)
        
        # Verify
        if verify_files():
            print("\n" + "=" * 70)
            print("✅ SUCCESS!")
            print("=" * 70)
            print("\n📋 Next steps:")
            print("  1. Make sure you have the dataset:")
            print("     python download_data.py")
            print("  2. Fix embedding paths (REQUIRED):")
            print("     python fix_embedding_paths.py")
            print("  3. Launch the app:")
            print("     streamlit run app.py")
        else:
            print("\n❌ Some files are missing. Please check the extraction.")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  - Check your internet connection")
        print("  - Verify the file ID is correct")
        print("  - Ensure the file is shared with 'Anyone with the link'")
        print("  - Try downloading manually from Google Drive")
        print(f"    https://drive.google.com/file/d/{FILE_ID}/view")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download pre-trained model from Google Drive')
    parser.add_argument('--file_id', type=str, help='Google Drive file ID')
    
    args = parser.parse_args()
    
    # Override with command line argument if provided
    if args.file_id:
        os.environ['GDRIVE_MODEL_ID'] = args.file_id
    
    main()