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


def evaluate_segmentations(ground_truths, segmentations):
    """
    Evaluate segmentations against ground truths using Dice coefficient.
    
    Parameters:
    - ground_truths: Dictionary of ground truth segmentations.
    - segmentations: Dictionary of model segmentations.
    
    Returns:
    - Dictionary with Dice coefficients for each image.
    """
    results = {}
    for image in ground_truths:
        if image in segmentations:
            gt_data = ground_truths[image]['data']
            seg_data = segmentations[image]
            dice_score = dice_coefficient(seg_data, gt_data)
            results[image] = dice_score
        else:
            results[image] = None  # No segmentation available for this image
    return results

def threshold_segmentation(segmentation, threshold=0.5):
    """
    Threshold the segmentation data to create a binary mask.
    
    Parameters:
    - segmentation: Numpy array of segmentation data.
    - threshold: Threshold value to apply.
    
    Returns:
    - Binary mask as a numpy array.
    """
    return (segmentation > threshold).astype(np.uint8)