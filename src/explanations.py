"""
Multi-view explanation methods for chest X-ray pneumonia classification:
- Grad-CAM: Visual attention heatmaps
- Occlusion Sensitivity: Perturbation-based importance
- Example-Based: Similar training cases via embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import faiss
import pickle


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, input_image, class_idx=None):
        """Generate Grad-CAM heatmap for given input"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Generate heatmap
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=0)  # [H, W]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), class_idx
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """Overlay heatmap on original image"""
        # Resize heatmap to match image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        if len(original_image.shape) == 2:  # Grayscale
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        overlayed = cv2.addWeighted(original_image, 1-alpha, 
                                   heatmap_colored, alpha, 0)
        
        return overlayed


class OcclusionSensitivity:
    """Occlusion-based sensitivity analysis"""
    
    def __init__(self, model, patch_size=32, stride=16):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
    
    def generate_heatmap(self, input_tensor, class_idx=None, baseline='gray'):
        """Generate occlusion sensitivity map"""
        self.model.eval()
        
        # Get original prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            
            original_prob = probs[0, class_idx].item()
        
        # Get image dimensions
        batch, channels, height, width = input_tensor.shape
        
        # Initialize sensitivity map
        sensitivity_map = np.zeros((height, width))
        count_map = np.zeros((height, width))
        
        # Create baseline patch
        if baseline == 'gray':
            baseline_value = input_tensor.mean()
        elif baseline == 'zero':
            baseline_value = 0.0
        else:
            baseline_value = baseline
        
        # Slide window across image
        positions = []
        for y in range(0, height - self.patch_size + 1, self.stride):
            for x in range(0, width - self.patch_size + 1, self.stride):
                positions.append((y, x))
        
        # Evaluate each occlusion
        for y, x in positions:
            # Create occluded image
            occluded = input_tensor.clone()
            occluded[:, :, y:y+self.patch_size, x:x+self.patch_size] = baseline_value
            
            # Get prediction
            with torch.no_grad():
                output = self.model(occluded)
                probs = F.softmax(output, dim=1)
                occluded_prob = probs[0, class_idx].item()
            
            # Calculate importance (drop in probability)
            importance = original_prob - occluded_prob
            
            # Update sensitivity map
            sensitivity_map[y:y+self.patch_size, x:x+self.patch_size] += importance
            count_map[y:y+self.patch_size, x:x+self.patch_size] += 1
        
        # Average overlapping regions
        sensitivity_map = np.divide(sensitivity_map, count_map, 
                                   where=count_map > 0,
                                   out=np.zeros_like(sensitivity_map))
        
        # Normalize
        sensitivity_map = sensitivity_map - sensitivity_map.min()
        sensitivity_map = sensitivity_map / (sensitivity_map.max() + 1e-8)
        
        return sensitivity_map, original_prob
    
    def overlay_heatmap(self, heatmap, original_image, alpha=0.4):
        """Overlay sensitivity map on original image"""
        # Resize heatmap to match image size
        h, w = original_image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        
        # Convert heatmap to RGB
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        if len(original_image.shape) == 2:  # Grayscale
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        overlayed = cv2.addWeighted(original_image, 1-alpha, 
                                   heatmap_colored, alpha, 0)
        
        return overlayed


class SimilaritySearch:
    """Find similar training examples using embeddings"""
    
    def __init__(self, embeddings_dir='embeddings'):
        self.embeddings_dir = embeddings_dir
        
        # Load embeddings and metadata
        self.embeddings = np.load(f'{embeddings_dir}/embeddings.npy')
        self.labels = np.load(f'{embeddings_dir}/labels.npy')
        with open(f'{embeddings_dir}/paths.pkl', 'rb') as f:
            self.paths = pickle.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(f'{embeddings_dir}/similarity_index.faiss')
        
        print(f"Loaded {len(self.embeddings)} training embeddings")
    
    def find_similar(self, query_embedding, k=5):
        """Find k most similar training images"""
        # Normalize query
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            results.append({
                'distance': float(dist),
                'similarity': float(dist),  # Cosine similarity (higher = more similar)
                'label': int(self.labels[idx]),
                'label_name': 'PNEUMONIA' if self.labels[idx] == 1 else 'NORMAL',
                'path': self.paths[idx]
            })
        
        return results


class FeatureExtractor(nn.Module):
    """Extract features from penultimate layer for similarity search"""
    
    def __init__(self, model):
        super().__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def load_model(model_path):
    """Load trained pneumonia classifier"""
    model = models.resnet18(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def preprocess_image(image_path, image_size=224):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def get_all_explanations(image_path, model_path='models/pneumonia_classifier.pth',
                        embeddings_dir='embeddings', device='cpu'):
    """
    Generate all explanations for a given image
    
    Returns:
        dict with keys: prediction, probability, gradcam, occlusion, similar_cases
    """
    device = torch.device(device)
    
    # Load model
    model = load_model(model_path).to(device)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()
    
    class_name = 'PNEUMONIA' if pred_class == 1 else 'NORMAL'
    
    # Grad-CAM
    target_layer = model.layer4[-1]  # Last conv layer
    gradcam = GradCAM(model, target_layer)
    gradcam_heatmap, _ = gradcam.generate_heatmap(image_tensor, class_idx=pred_class)
    
    # Occlusion Sensitivity
    occlusion = OcclusionSensitivity(model, patch_size=32, stride=16)
    occlusion_heatmap, orig_prob = occlusion.generate_heatmap(
        image_tensor, class_idx=pred_class
    )
    
    # Similar cases
    feature_extractor = FeatureExtractor(model).to(device)
    with torch.no_grad():
        query_embedding = feature_extractor(image_tensor).cpu().numpy()[0]
    
    similarity_search = SimilaritySearch(embeddings_dir)
    similar_cases = similarity_search.find_similar(query_embedding, k=6)
    
    # Convert image for visualization
    original_np = np.array(original_image)
    
    # Overlay heatmaps
    gradcam_overlay = gradcam.overlay_heatmap(gradcam_heatmap, original_np)
    occlusion_overlay = occlusion.overlay_heatmap(occlusion_heatmap, original_np)
    
    return {
        'prediction': class_name,
        'predicted_class': pred_class,
        'probability': pred_prob,
        'gradcam_heatmap': gradcam_heatmap,
        'gradcam_overlay': gradcam_overlay,
        'occlusion_heatmap': occlusion_heatmap,
        'occlusion_overlay': occlusion_overlay,
        'similar_cases': similar_cases,
        'original_image': original_np
    }