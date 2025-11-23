"""
Diagnostic script to check if everything is set up correctly.
Run this before launching the Streamlit app.
"""

import os
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ Python 3.8+ required")
        return False
    print("  ✅ Python version OK")
    return True


def check_imports():
    """Check if all required packages can be imported"""
    print("\n📦 Checking imports...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'faiss': 'FAISS',
        'sklearn': 'scikit-learn',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_custom_modules():
    """Check if custom modules can be imported"""
    print("\n🔧 Checking custom modules...")
    
    # Add src to path
    sys.path.append('src')
    
    modules = {
        'explanations': ['get_all_explanations', 'GradCAM', 'OcclusionSensitivity'],
        'report_generator': ['generate_report', 'generate_summary'],
    }
    
    all_ok = True
    for module_name, functions in modules.items():
        try:
            module = __import__(module_name)
            print(f"  ✅ {module_name}.py")
            
            # Check if functions exist
            for func in functions:
                if hasattr(module, func):
                    print(f"     ✅ {func}")
                else:
                    print(f"     ❌ {func} - NOT FOUND")
                    all_ok = False
                    
        except ImportError as e:
            print(f"  ❌ {module_name}.py - IMPORT ERROR")
            print(f"     Error: {e}")
            all_ok = False
    
    return all_ok


def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking files...")
    
    required_files = {
        'Model': 'models/pneumonia_classifier.pth',
        'Embeddings': 'embeddings/embeddings.npy',
        'Labels': 'embeddings/labels.npy',
        'Paths': 'embeddings/paths.pkl',
        'FAISS Index': 'embeddings/similarity_index.faiss',
    }
    
    all_ok = True
    for name, path in required_files.items():
        path_obj = Path(path)
        if path_obj.exists():
            size = path_obj.stat().st_size / (1024 * 1024)
            print(f"  ✅ {name}: {path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {name}: {path} - NOT FOUND")
            all_ok = False
    
    return all_ok


def check_dataset():
    """Check if dataset exists"""
    print("\n📊 Checking dataset...")
    
    data_path = Path('data/chest_xray')
    
    if not data_path.exists():
        print(f"  ❌ Dataset not found: {data_path}")
        return False
    
    splits = ['train', 'test', 'val']
    classes = ['NORMAL', 'PNEUMONIA']
    
    all_ok = True
    for split in splits:
        for cls in classes:
            cls_path = data_path / split / cls
            if cls_path.exists():
                count = len(list(cls_path.glob('*.jpeg')))
                print(f"  ✅ {split}/{cls}: {count} images")
            else:
                print(f"  ❌ {split}/{cls}: NOT FOUND")
                all_ok = False
    
    return all_ok


def test_model_loading():
    """Try to load the model"""
    print("\n🧠 Testing model loading...")
    
    try:
        import torch
        from torchvision import models
        import torch.nn as nn
        
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)
        
        checkpoint = torch.load('models/pneumonia_classifier.pth', 
                               map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print("  ✅ Model loaded successfully")
        print(f"     Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"     Val Acc: {checkpoint.get('val_acc', 'unknown'):.2f}%")
        return True
        
    except FileNotFoundError:
        print("  ❌ Model file not found")
        return False
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False


def test_embeddings_loading():
    """Try to load embeddings"""
    print("\n🔍 Testing embeddings loading...")
    
    try:
        import numpy as np
        import pickle
        import faiss
        
        embeddings = np.load('embeddings/embeddings.npy')
        labels = np.load('embeddings/labels.npy')
        
        with open('embeddings/paths.pkl', 'rb') as f:
            paths = pickle.load(f)
        
        index = faiss.read_index('embeddings/similarity_index.faiss')
        
        print(f"  ✅ Embeddings loaded successfully")
        print(f"     Embeddings shape: {embeddings.shape}")
        print(f"     Number of labels: {len(labels)}")
        print(f"     Number of paths: {len(paths)}")
        print(f"     FAISS index size: {index.ntotal}")
        
        # Check if paths exist
        sample_paths = paths[:5]
        existing = sum(1 for p in sample_paths if Path(p).exists())
        print(f"     Sample paths exist: {existing}/{len(sample_paths)}")
        
        if existing < len(sample_paths):
            print(f"     ⚠️  Some paths don't exist. Run: python fix_embedding_paths.py")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Embeddings loading failed: {e}")
        return False


def main():
    """Run all diagnostics"""
    print("=" * 70)
    print("🔬 X-Ray Transparency Lab - System Diagnostic")
    print("=" * 70)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("Custom Modules", check_custom_modules),
        ("Required Files", check_files),
        ("Dataset", check_dataset),
        ("Model Loading", test_model_loading),
        ("Embeddings Loading", test_embeddings_loading),
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("🎉 All checks passed! You can run the app:")
        print("   streamlit run app.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n📋 Common fixes:")
        print("   Missing packages: pip install -r requirements.txt")
        print("   Missing model: python download_with_gdown.py")
        print("   Missing dataset: python download_data.py")
        print("   Wrong paths: python fix_embedding_paths.py")
    
    print("=" * 70)


if __name__ == '__main__':
    main()