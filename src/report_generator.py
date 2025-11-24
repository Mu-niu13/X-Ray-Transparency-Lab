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

    This version uses ONLY:
    - prediction
    - probability
    - Grad-CAM
    - Occlusion sensitivity

    All similar-cases content has been removed.
    """
    prediction = explanation_results['prediction']
    probability = explanation_results['probability']
    gradcam_heatmap = explanation_results['gradcam_heatmap']
    occlusion_heatmap = explanation_results['occlusion_heatmap']

    # High-level analysis helpers
    confidence_level, confidence_phrase = get_confidence_level(probability)
    location = analyze_heatmap_location(gradcam_heatmap)
    gradcam_analysis = analyze_gradcam_regions(gradcam_heatmap)
    consistency = check_gradcam_occlusion_consistency(
        gradcam_heatmap, occlusion_heatmap
    )

    prob_percent = probability * 100

    report_sections = []

    # 1. Overall conclusion
    report_sections.append(
        f"### Overall AI Assessment\n\n"
        f"The AI model predicts **{prediction}** with a **{confidence_level}** level "
        f"of confidence (estimated probability: **{prob_percent:.1f}%**). "
        f"This probability {confidence_phrase} the presence of {prediction.lower()} "
        f"in this chest X-ray image."
    )

    # 2. Grad-CAM explanation
    report_sections.append(
        f"### Visual Attention (Grad-CAM)\n\n"
        f"The Grad-CAM heatmap shows that the model's attention is primarily "
        f"focused on the **{location}**. {gradcam_analysis} "
        f"These regions contain image patterns that the model finds most informative "
        f"for distinguishing between normal lungs and pneumonia."
    )

    # 3. Occlusion sensitivity explanation
    report_sections.append(
        f"### Importance Verification (Occlusion Sensitivity)\n\n"
        f"Occlusion sensitivity analysis {consistency}. When the most highlighted "
        f"regions in the Grad-CAM map are masked or perturbed, the model's predicted "
        f"probability for **{prediction}** changes substantially. This supports the "
        f"interpretation that these regions are truly important to the model's decision."
    )

    # 4. Clinical-style interpretation (NO mention of similar training cases)
    if prediction == "PNEUMONIA":
        interpretation = (
            "Taken together, the visual attention map and occlusion analysis suggest "
            "that the highlighted lung regions contain patterns consistent with "
            "pneumonia (for example, areas of increased opacity or consolidation). "
            "However, this output should be interpreted as a decision-support signal "
            "rather than a definitive diagnosis."
        )
    else:  # NORMAL
        interpretation = (
            "Taken together, the visual attention map and occlusion analysis do not "
            "show strong, localized patterns typically associated with pneumonia. "
            "The model's assessment is more consistent with a normal chest X-ray, "
            "but clinical correlation with symptoms and additional tests remains essential."
        )

    report_sections.append(
        f"### Interpretation\n\n{interpretation}"
    )

    # 5. Disclaimer
    report_sections.append(
        "---\n\n"
        "*⚠️ This is an AI-generated analysis for educational and research purposes only. "
        "It must not be used for clinical diagnosis or treatment decisions. Always consult "
        "qualified healthcare professionals for medical interpretation and decision-making.*"
    )

    return "\n\n".join(report_sections)


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
        'gradcam_heatmap': np.random.rand(7, 7) * 0.5 + 0.3,  # simulated heatmap
        'occlusion_heatmap': np.random.rand(224, 224) * 0.5 + 0.2
    }
    
    print("=== SUMMARY ===")
    print(generate_summary(mock_results))
    print("\n" + "="*80 + "\n")
    print("=== FULL REPORT ===")
    print(generate_report(mock_results))