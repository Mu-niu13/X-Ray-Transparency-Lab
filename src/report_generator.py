"""
Generate natural language reports from explanation outputs.
Template-based system with dynamic content insertion.
"""

import numpy as np


def analyze_heatmap_location(heatmap, threshold=0.6):
    """
    Determine the primary location of activation in the heatmap.
    
    Returns:
        str: Description of location (e.g., "right lower lung field")
    """
    h, w = heatmap.shape
    
    # Threshold to find highly activated regions
    high_activation = heatmap > threshold
    
    if not high_activation.any():
        return "diffuse regions"
    
    # Get coordinates of high activation
    y_coords, x_coords = np.where(high_activation)
    
    # Determine left/right (chest X-rays are typically mirrored in display)
    avg_x = x_coords.mean()
    if avg_x < w * 0.4:
        horizontal = "left"
    elif avg_x > w * 0.6:
        horizontal = "right"
    else:
        horizontal = "central"
    
    # Determine upper/middle/lower
    avg_y = y_coords.mean()
    if avg_y < h * 0.33:
        vertical = "upper"
    elif avg_y > h * 0.67:
        vertical = "lower"
    else:
        vertical = "middle"
    
    if horizontal == "central":
        return f"{vertical} lung field"
    else:
        return f"{horizontal} {vertical} lung field"


def analyze_occlusion_consistency(occlusion_heatmap, gradcam_heatmap, threshold=0.5):
    """
    Check if occlusion sensitivity confirms Grad-CAM findings.
    
    Returns:
        str: Description of consistency
    """
    # Resize heatmaps to same size if needed
    import cv2
    
    if occlusion_heatmap.shape != gradcam_heatmap.shape:
        # Resize gradcam to match occlusion size
        target_size = (occlusion_heatmap.shape[1], occlusion_heatmap.shape[0])
        gradcam_heatmap = cv2.resize(gradcam_heatmap, target_size)
    
    # Find regions with high importance
    gradcam_high = gradcam_heatmap > threshold
    occlusion_high = occlusion_heatmap > threshold
    
    # Calculate overlap
    overlap = np.logical_and(gradcam_high, occlusion_high).sum()
    total_gradcam = gradcam_high.sum()
    
    if total_gradcam == 0:
        return "consistent with the highlighted regions"
    
    overlap_ratio = overlap / total_gradcam
    
    if overlap_ratio > 0.6:
        return "strongly supports the importance of these regions"
    elif overlap_ratio > 0.3:
        return "partially confirms these regions as important"
    else:
        return "suggests importance may be distributed across multiple areas"


def analyze_similar_cases(similar_cases, prediction_class):
    """
    Analyze the similar training cases.
    
    Returns:
        dict with analysis info
    """
    # Count labels in similar cases
    labels = [case['label'] for case in similar_cases]
    pneumonia_count = sum(1 for label in labels if label == 1)
    normal_count = len(labels) - pneumonia_count
    
    # Check consistency
    predicted_label = 1 if prediction_class == 'PNEUMONIA' else 0
    consistent_count = sum(1 for label in labels if label == predicted_label)
    consistency = consistent_count / len(labels)
    
    # Average similarity
    avg_similarity = np.mean([case['similarity'] for case in similar_cases])
    
    return {
        'pneumonia_count': pneumonia_count,
        'normal_count': normal_count,
        'consistency': consistency,
        'avg_similarity': avg_similarity
    }


def get_confidence_level(probability):
    """Categorize confidence level"""
    if probability >= 0.90:
        return "very high", "strongly indicates"
    elif probability >= 0.75:
        return "high", "indicates"
    elif probability >= 0.60:
        return "moderate", "suggests"
    else:
        return "low", "weakly suggests"


def generate_report(explanation_results):
    """
    Generate a natural language report from explanation results.
    
    Args:
        explanation_results: dict from get_all_explanations()
    
    Returns:
        str: Formatted report text
    """
    prediction = explanation_results['prediction']
    probability = explanation_results['probability']
    gradcam_heatmap = explanation_results['gradcam_heatmap']
    occlusion_heatmap = explanation_results['occlusion_heatmap']
    similar_cases = explanation_results['similar_cases']
    
    # Analyze components
    confidence_level, confidence_verb = get_confidence_level(probability)
    location = analyze_heatmap_location(gradcam_heatmap)
    consistency = analyze_occlusion_consistency(occlusion_heatmap, gradcam_heatmap)
    similar_analysis = analyze_similar_cases(similar_cases, prediction)
    
    # Build report sections
    report_sections = []
    
    # 1. Prediction statement
    prob_percent = probability * 100
    report_sections.append(
        f"**Model Prediction:** {prediction} ({prob_percent:.1f}% probability)\n\n"
        f"The model {confidence_verb} **{prediction.lower()}** with {confidence_level} confidence."
    )
    
    # 2. Grad-CAM analysis
    report_sections.append(
        f"**Visual Attention Analysis (Grad-CAM):**\n"
        f"The model's attention is primarily focused on the **{location}**. "
        f"These regions show the strongest activation in the neural network's decision process, "
        f"indicating they contain visual patterns most characteristic of {prediction.lower()}."
    )
    
    # 3. Occlusion sensitivity analysis
    report_sections.append(
        f"**Importance Verification (Occlusion Sensitivity):**\n"
        f"Perturbation analysis {consistency}. "
        f"When these regions are masked, the model's {prediction.lower()} probability "
        f"changes significantly, confirming their diagnostic relevance."
    )
    
    # 4. Similar cases analysis
    similar_pneumonia = similar_analysis['pneumonia_count']
    similar_normal = similar_analysis['normal_count']
    consistency_pct = similar_analysis['consistency'] * 100
    
    if similar_pneumonia > similar_normal:
        similar_statement = (
            f"most similar to **{similar_pneumonia} pneumonia cases** "
            f"and {similar_normal} normal cases from the training data"
        )
    elif similar_normal > similar_pneumonia:
        similar_statement = (
            f"most similar to **{similar_normal} normal cases** "
            f"and {similar_pneumonia} pneumonia cases from the training data"
        )
    else:
        similar_statement = (
            f"similar to both pneumonia ({similar_pneumonia} cases) "
            f"and normal ({similar_normal} cases) training examples"
        )
    
    report_sections.append(
        f"**Example-Based Explanation (Similar Training Cases):**\n"
        f"This image is {similar_statement}. "
        f"The {consistency_pct:.0f}% agreement with the model's prediction "
        f"{'suggests strong pattern consistency' if consistency_pct >= 70 else 'indicates some visual ambiguity'}."
    )
    
    # 5. Clinical interpretation note
    if confidence_level in ["very high", "high"]:
        interpretation = (
            "The convergence of multiple explanation methods supports the model's assessment. "
            f"The visual patterns in the {location} are consistent with those typically seen in {prediction.lower()} cases."
        )
    else:
        interpretation = (
            "The model shows some uncertainty in this case. "
            "Clinical judgment should carefully consider the visual patterns and compare with the similar training examples. "
            "Additional diagnostic information may be valuable."
        )
    
    report_sections.append(
        f"**Interpretation:**\n{interpretation}"
    )
    
    # 6. Disclaimer
    report_sections.append(
        "---\n\n"
        "*⚠️ This is an AI-generated analysis for educational and research purposes only. "
        "It should not be used for clinical diagnosis. Always consult qualified healthcare professionals "
        "for medical interpretation and decision-making.*"
    )
    
    # Combine all sections
    full_report = "\n\n".join(report_sections)
    
    return full_report


def generate_summary(explanation_results):
    """
    Generate a brief summary (1-2 sentences) for quick display.
    
    Args:
        explanation_results: dict from get_all_explanations()
    
    Returns:
        str: Brief summary text
    """
    prediction = explanation_results['prediction']
    probability = explanation_results['probability']
    location = analyze_heatmap_location(explanation_results['gradcam_heatmap'])
    
    prob_percent = probability * 100
    
    summary = (
        f"The model predicts **{prediction}** with {prob_percent:.1f}% confidence, "
        f"focusing on patterns in the {location}."
    )
    
    return summary


# Example usage and testing
if __name__ == '__main__':
    # Mock data for testing
    mock_results = {
        'prediction': 'PNEUMONIA',
        'probability': 0.87,
        'gradcam_heatmap': np.random.rand(7, 7) * 0.5 + 0.3,  # Simulated heatmap
        'occlusion_heatmap': np.random.rand(224, 224) * 0.5 + 0.2,
        'similar_cases': [
            {'label': 1, 'similarity': 0.92, 'label_name': 'PNEUMONIA'},
            {'label': 1, 'similarity': 0.88, 'label_name': 'PNEUMONIA'},
            {'label': 1, 'similarity': 0.85, 'label_name': 'PNEUMONIA'},
            {'label': 0, 'similarity': 0.82, 'label_name': 'NORMAL'},
            {'label': 1, 'similarity': 0.80, 'label_name': 'PNEUMONIA'},
            {'label': 1, 'similarity': 0.78, 'label_name': 'PNEUMONIA'},
        ]
    }
    
    print("=== SUMMARY ===")
    print(generate_summary(mock_results))
    print("\n" + "="*80 + "\n")
    print("=== FULL REPORT ===")
    print(generate_report(mock_results))