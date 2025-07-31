import numpy as np

def dice_coefficient(prediction, ground_truth):
    """
    Calculate the Dice coefficient between prediction and ground truth.
    
    Args:
        prediction (np.ndarray): Predicted segmentation.
        ground_truth (np.ndarray): Ground truth segmentation.
        
    Returns:
        float: Dice coefficient value.
    """
    intersection = np.sum(prediction * ground_truth)
    return 2.0 * intersection / (np.sum(prediction) + np.sum(ground_truth))

def display_results(results):
    for image, score in results.items():
        if score is not None:
            print(f"Image: {image}, Dice Coefficient: {score:.6f}")