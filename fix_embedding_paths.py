"""
Fix embedding paths after downloading from Colab.
Colab paths won't match your local paths, so we need to update them.
"""

import pickle
import argparse
from pathlib import Path


def fix_paths(embeddings_dir='embeddings', data_dir='data/chest_xray/train'):
    """
    Update the paths in paths.pkl to match local data directory
    
    Args:
        embeddings_dir: Directory containing paths.pkl
        data_dir: Local path to training data
    """
    
    paths_file = Path(embeddings_dir) / 'paths.pkl'
    
    if not paths_file.exists():
        print(f"❌ Error: {paths_file} not found!")
        print("Make sure you've extracted trained_model.zip")
        return False
    
    # Load existing paths
    with open(paths_file, 'rb') as f:
        old_paths = pickle.load(f)
    
    print(f"📂 Found {len(old_paths)} paths")
    print(f"Example old path: {old_paths[0]}")
    
    # Create new paths by extracting filename and class
    new_paths = []
    data_path = Path(data_dir)
    
    for old_path in old_paths:
        # Extract class (NORMAL or PNEUMONIA) and filename
        path_parts = Path(old_path).parts
        
        # Find NORMAL or PNEUMONIA in path
        if 'NORMAL' in path_parts:
            class_name = 'NORMAL'
        elif 'PNEUMONIA' in path_parts:
            class_name = 'PNEUMONIA'
        else:
            print(f"⚠️  Warning: Could not determine class for {old_path}")
            continue
        
        # Get filename
        filename = Path(old_path).name
        
        # Create new path
        new_path = str(data_path / class_name / filename)
        new_paths.append(new_path)
    
    # Save updated paths
    backup_file = paths_file.with_suffix('.pkl.backup')
    
    # Backup original
    print(f"💾 Backing up original to {backup_file}")
    with open(backup_file, 'wb') as f:
        pickle.dump(old_paths, f)
    
    # Save new paths
    with open(paths_file, 'wb') as f:
        pickle.dump(new_paths, f)
    
    print(f"✅ Updated {len(new_paths)} paths")
    print(f"Example new path: {new_paths[0]}")
    
    # Verify a few paths exist
    missing = 0
    for path in new_paths[:10]:
        if not Path(path).exists():
            missing += 1
    
    if missing > 0:
        print(f"\n⚠️  Warning: {missing}/10 sample paths don't exist locally")
        print("Make sure you've downloaded the dataset with: python download_data.py")
    else:
        print(f"\n✅ All sample paths verified!")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix embedding paths after training on Colab'
    )
    parser.add_argument(
        '--embeddings_dir',
        type=str,
        default='embeddings',
        help='Directory containing paths.pkl (default: embeddings/)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/chest_xray/train',
        help='Local path to training data (default: data/chest_xray/train)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔧 Fixing Embedding Paths")
    print("=" * 60)
    
    success = fix_paths(args.embeddings_dir, args.data_dir)
    
    if success:
        print("\n✅ Done! You can now run: streamlit run app.py")
    else:
        print("\n❌ Failed to fix paths")


if __name__ == '__main__':
    main()