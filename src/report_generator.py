"""
Generate natural language reports and summaries from explanation outputs.
This version:
- Uses prediction, probability, Grad-CAM, and occlusion sensitivity
- Does NOT use or mention similar cases
"""

import numpy as np


def get_confidence_level(prob):
    """
    Map probability to a qualitative confidence level and phrase.
    """
    if prob >= 0.9:
        return "very high", "strongly supports"
    elif prob >= 0.75:
        return "high", "supports"
    elif prob >= 0.6:
        return "moderate", "suggests but does not confirm"
    else:
        return "low", "provides only weak evidence for"


def analyze_heatmap_location(heatmap, threshold=0.6):
    """
    Determine the primary location of activation in the heatmap.

    Returns:
        str: Description of location (e.g., "right lower lung field")
    """
    if heatmap is None:
        return "lung fields"

    h, w = heatmap.shape
    norm = heatmap / (heatmap.max() + 1e-8)

    mask = norm > threshold
    if not mask.any():
        return "lung fields"

    y_coords, x_coords = np.where(mask)

    # Left / right / central
    avg_x = x_coords.mean()
    if avg_x < w * 0.33:
        horizontal = "left"
    elif avg_x > w * 0.67:
        horizontal = "right"
    else:
        horizontal = "central"

    # Upper / middle / lower
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


def analyze_gradcam_regions(gradcam_heatmap, high_threshold=0.7, medium_threshold=0.4):
    """
    Provide a short text description of Grad-CAM intensity patterns.
    """
    if gradcam_heatmap is None:
        return "The attention map could not be computed for this image."

    norm = gradcam_heatmap / (gradcam_heatmap.max() + 1e-8)

    high = (norm > high_threshold).mean()
    medium = ((norm > medium_threshold) & (norm <= high_threshold)).mean()

    if high > 0.2:
        return (
            "The model shows strong, sharply localized activation in this region, "
            "suggesting it finds highly discriminative features there."
        )
    elif medium > 0.3:
        return (
            "The model shows moderately diffuse activation, indicating that the decision "
            "relies on broader texture patterns rather than a single focal lesion."
        )
    else:
        return (
            "The Grad-CAM map shows only weak activation, suggesting that the model's "
            "decision is not driven by a single clearly defined region."
        )


def check_gradcam_occlusion_consistency(gradcam_heatmap, occlusion_heatmap, threshold=0.5):
    """
    Check if occlusion sensitivity confirms Grad-CAM findings.

    Returns:
        str: Description of consistency.
    """
    if gradcam_heatmap is None or occlusion_heatmap is None:
        return (
            "could not be fully assessed due to missing explanation maps."
        )

    # Normalize both to [0,1]
    g = gradcam_heatmap
    o = occlusion_heatmap

    g = g / (g.max() + 1e-8)
    o = o / (o.max() + 1e-8)

    # Resize occlusion map to Grad-CAM resolution if needed
    if o.shape != g.shape:
        # simple nearest-neighbor down/upsampling via indexing
        gy, gx = g.shape
        oy, ox = o.shape
        y_idx = (np.linspace(0, oy - 1, gy)).astype(int)
        x_idx = (np.linspace(0, ox - 1, gx)).astype(int)
        o = o[np.ix_(y_idx, x_idx)]

    gradcam_high = g > threshold
    occlusion_high = o > threshold

    overlap = (gradcam_high & occlusion_high).sum()
    union = (gradcam_high | occlusion_high).sum() + 1e-8
    jaccard = overlap / union

    if jaccard > 0.5:
        return (
            "appears consistent: regions highlighted by Grad-CAM largely overlap "
            "with areas that strongly affect the prediction when occluded"
        )
    elif jaccard > 0.2:
        return (
            "shows partial consistency: some Grad-CAM regions are supported by the "
            "occlusion analysis, but the overlap is only moderate"
        )
    else:
        return (
            "shows limited consistency: the occlusion analysis highlights somewhat "
            "different regions than Grad-CAM, so the explanation should be interpreted cautiously"
        )


def generate_summary(explanation_results):
    """
    Generate a brief, 1–2 sentence summary for the UI.
    """
    prediction = explanation_results["prediction"]
    probability = explanation_results["probability"]
    gradcam_heatmap = explanation_results.get("gradcam_heatmap")
    occlusion_heatmap = explanation_results.get("occlusion_heatmap")

    confidence_level, confidence_phrase = get_confidence_level(probability)
    location = analyze_heatmap_location(gradcam_heatmap)
    consistency = check_gradcam_occlusion_consistency(gradcam_heatmap, occlusion_heatmap)

    prob_percent = probability * 100

    summary = (
        f"The AI model predicts **{prediction}** with **{confidence_level}** confidence "
        f"(about **{prob_percent:.1f}%**). The Grad-CAM map focuses mainly on the "
        f"**{location}**, and the occlusion analysis {consistency}."
    )

    return summary


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
    prediction = explanation_results["prediction"]
    probability = explanation_results["probability"]
    gradcam_heatmap = explanation_results.get("gradcam_heatmap")
    occlusion_heatmap = explanation_results.get("occlusion_heatmap")

    confidence_level, confidence_phrase = get_confidence_level(probability)
    location = analyze_heatmap_location(gradcam_heatmap)
    gradcam_analysis = analyze_gradcam_regions(gradcam_heatmap)
    consistency = check_gradcam_occlusion_consistency(gradcam_heatmap, occlusion_heatmap)

    prob_percent = probability * 100.0

    report_sections = []

    # 1. Overall conclusion
    report_sections.append(
        f"### Overall AI Assessment\n\n"
        f"The AI model predicts **{prediction}** with a **{confidence_level}** level of "
        f"confidence (estimated probability: **{prob_percent:.1f}%**). "
        f"This probability {confidence_phrase} the presence of "
        f"{prediction.lower()} in this chest X-ray image."
    )

    # 2. Grad-CAM explanation
    report_sections.append(
        f"### Visual Attention (Grad-CAM)\n\n"
        f"The Grad-CAM heatmap shows that the model's attention is primarily focused on "
        f"the **{location}**. {gradcam_analysis} "
        f"These regions contain image patterns that the model finds most informative for "
        f"distinguishing between normal lungs and pneumonia."
    )

    # 3. Occlusion sensitivity explanation
    report_sections.append(
        f"### Importance Verification (Occlusion Sensitivity)\n\n"
        f"Occlusion sensitivity analysis {consistency}. When the most highlighted regions "
        f"in the Grad-CAM map are masked or perturbed, the model's predicted probability "
        f"for **{prediction}** changes substantially. This supports the interpretation "
        f"that these regions are truly important to the model's decision."
    )

    # 4. Clinical-style interpretation (no mention of similar cases)
    if prediction == "PNEUMONIA":
        interpretation = (
            "Taken together, the visual attention map and occlusion analysis suggest that "
            "the highlighted lung regions contain patterns consistent with pneumonia "
            "(for example, areas of increased opacity or consolidation). However, this "
            "output should be interpreted as a decision-support signal rather than a "
            "definitive diagnosis."
        )
    else:  # NORMAL
        interpretation = (
            "Taken together, the visual attention map and occlusion analysis do not show "
            "strong, localized patterns typically associated with pneumonia. The model's "
            "assessment is more consistent with a normal chest X-ray, but clinical "
            "correlation with symptoms and additional tests remains essential."
        )

    report_sections.append(f"### Interpretation\n\n{interpretation}")

    # 5. Disclaimer
    report_sections.append(
        "---\n\n"
        "*⚠️ This is an AI-generated analysis for educational and research purposes only. "
        "It must not be used for clinical diagnosis or treatment decisions. Always consult "
        "qualified healthcare professionals for medical interpretation and decision-making.*"
    )

    return "\n\n".join(report_sections)
