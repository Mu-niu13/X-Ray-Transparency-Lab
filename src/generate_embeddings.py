"""
Generate embeddings for all training images for similarity search.
Creates a FAISS index for fast nearest neighbor retrieval.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pickle
import faiss
from tqdm import tqdm
from pathlib import Path


class ChestXRayDataset(torch.utils.data.Dataset):
    """Dataset that also returns image paths"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.paths = []
        
        # Load NORMAL images (label=0)
        normal_dir = self.data_dir / 'NORMAL'
        if normal_dir.exists():
            for img_path in normal_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(0)
                self.paths.append(str(img_path))
        
        # Load PNEUMONIA images (label=1)
        pneumonia_dir = self.data_dir / 'PNEUMONIA'
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob('*.jpeg'):
                self.images.append(str(img_path))
                self.labels.append(1)
                self.paths.append(str(img_path))
        
        print(f"Found {len(self.images)} images for embedding generation")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        path = self.paths[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, path


class FeatureExtractor(nn.Module):
    """Extract features from penultimate layer"""
    
    def __init__(self, model):
        super().__init__()
        # Remove the final classification layer
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_trained_model(model_path):
    """Load the trained model"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


def generate_embeddings(model_path, data_dir, output_dir='embeddings', 
                       batch_size=32):
    """Generate embeddings for all training images"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading trained model...")
    model = load_trained_model(model_path)
    feature_extractor = FeatureExtractor(model).to(device)
    feature_extractor.eval()
    
    # Data transform (same as validation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading dataset...")
    dataset = ChestXRayDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    # Generate embeddings
    print("Generating embeddings...")
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader):
            images = images.to(device)
            embeddings = feature_extractor(images)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.array(all_labels)
    
    print(f"Generated {len(all_embeddings)} embeddings of dimension {all_embeddings.shape[1]}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings and metadata
    print("Saving embeddings...")
    np.save(f'{output_dir}/embeddings.npy', all_embeddings)
    np.save(f'{output_dir}/labels.npy', all_labels)
    
    with open(f'{output_dir}/paths.pkl', 'wb') as f:
        pickle.dump(all_paths, f)
    
    # Create FAISS index for fast similarity search
    print("Creating FAISS index...")
    dimension = all_embeddings.shape[1]
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(all_embeddings)
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    index.add(all_embeddings.astype('float32'))
    
    # Save index
    faiss.write_index(index, f'{output_dir}/similarity_index.faiss')
    
    print(f"Embeddings saved to {output_dir}/")
    print(f"  - embeddings.npy: {all_embeddings.shape}")
    print(f"  - labels.npy: {all_labels.shape}")
    print(f"  - paths.pkl: {len(all_paths)} paths")
    print(f"  - similarity_index.faiss")
    
    return all_embeddings, all_labels, all_paths


def test_similarity_search(output_dir='embeddings', k=5):
    """Test the similarity search with a random query"""
    print("\nTesting similarity search...")
    
    # Load data
    embeddings = np.load(f'{output_dir}/embeddings.npy')
    labels = np.load(f'{output_dir}/labels.npy')
    with open(f'{output_dir}/paths.pkl', 'rb') as f:
        paths = pickle.load(f)
    
    index = faiss.read_index(f'{output_dir}/similarity_index.faiss')
    
    # Pick a random query
    query_idx = np.random.randint(0, len(embeddings))
    query_embedding = embeddings[query_idx:query_idx+1].astype('float32')
    
    # Search
    distances, indices = index.search(query_embedding, k+1)  # +1 to skip self
    
    print(f"\nQuery image: {paths[query_idx]}")
    print(f"Label: {'PNEUMONIA' if labels[query_idx] == 1 else 'NORMAL'}")
    print(f"\nTop {k} similar images:")
    
    for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:]), 1):
        print(f"{i}. Distance: {dist:.4f}, Label: {'PNEUMONIA' if labels[idx] == 1 else 'NORMAL'}")
        print(f"   Path: {paths[idx]}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for similarity search')
    parser.add_argument('--model_path', type=str, 
                       default='models/pneumonia_classifier.pth',
                       help='Path to trained model')
    parser.add_argument('--data_dir', type=str, 
                       default='data/chest_xray/train',
                       help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, 
                       default='embeddings',
                       help='Directory to save embeddings')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for embedding generation')
    parser.add_argument('--test', action='store_true',
                       help='Run similarity search test after generation')
    
    args = parser.parse_args()
    
    # Generate embeddings
    generate_embeddings(args.model_path, args.data_dir, 
                       args.output_dir, args.batch_size)
    
    # Test if requested
    if args.test:
        test_similarity_search(args.output_dir)