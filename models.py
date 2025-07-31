import numpy as np
from analytics import dice_coefficient, display_results 
from load_data import Data
from sklearn.linear_model import LogisticRegression

def linear_ensemble(images, segmentation_models, data, weights=None):
    """
    Perform linear ensemble on the given images and segmentations.
    
    Parameters:
    - images: List of image names.
    - segmentation_models: List of model names to ensemble.
    - data: Data object containing all loaded segmentations.
    - weights: Optional list of weights for each segmentation. If None, equal weights are used.
    
    Returns:
    - Ensemble result as a dictionary of numpy arrays.
    """

    if weights is None:
        weights = np.ones(len(segmentation_models)) / len(segmentation_models)

    ensembled_images = {}
    for image in images:
        if image in data:
            first_model = segmentation_models[0]
            base_img = np.zeros_like(data[image][first_model]['data'], dtype=np.float32)
            
            for i, model in enumerate(segmentation_models):
                if model in data[image]:
                    model_data = data[image][model]['data']
                    base_img += weights[i] * model_data
                else:
                    print(f"Warning: {image} not found in model {model}")
            
            ensembled_images[image] = base_img
        else:
            print(f"Warning: Image {image} not found in data")

    return ensembled_images

class LogisticRegressor:
    def __init__(self, labels, ground_truths, segmentation_models, data):
        """
        Initialize the logistic regressor with labels and ground truths.
        
        Args:
            labels (list): List of labels for the regression.
            ground_truths (dict): Dictionary of ground truth segmentations.
            segmentation_models (list): List of segmentation model names.
            data: Data object containing all loaded segmentations.
        """
        self.labels = labels
        self.ground_truths = ground_truths
        self.segmentation_models = segmentation_models
        self.data = data

    def train_model(self, training_images):
        """
        Train the logistic regression model on the provided training images.
        Features are built using all segmentation models as columns.
        
        Args:
            training_images (list): List of image names to train on.
        
        Returns:
            LogisticRegression: Trained logistic regression model.
        """
        X = self.build_feature_vector(training_images, self.segmentation_models)
        y = []

        for image in training_images:
            y += self.ground_truths[image]['data'].flatten().tolist()
        y = np.array(y)

        model = LogisticRegression(
            C=100,                    # Try 0.1, 1.0, 10.0
            penalty='elasticnet',     # Combines L1 and L2 regularization
            solver='saga',            # Required for elasticnet
            l1_ratio=0.5,            # Balance between L1 and L2 (0=L2, 1=L1, 0.5=equal mix)
            max_iter=1000,            # Increase if convergence issues
            class_weight='balanced'   # Handle imbalanced segmentation
        )
        print(X.shape, y.shape)
        model.fit(X, y)
        
        return model
    
    def build_feature_vector(self, images, segmentation_models):
        X = []
        for model in segmentation_models:
            feature = []
            for image in images:
                feature += self.data[image][model]['data'].flatten().tolist()
            X.append(feature)
        return np.array(X).T
    


    
    def predict(self, model, test_images):
        """
        Make predictions on test images using the trained model.
        
        Args:
            model: Trained LogisticRegression model.
            test_images (list): List of image names to predict on.
        
        Returns:
            dict: Dictionary of predictions for each test image.
        """
        X = self.build_feature_vector(test_images, self.segmentation_models)
        predictions = model.predict(X)
        
        # Reshape predictions back to image format
        result_dict = {}
        pixel_idx = 0
        
        for image in test_images:
            if image in self.data:
                # Get original shape from the first available model
                original_shape = self.data[image][self.segmentation_models[0]]['data'].shape
                num_pixels = np.prod(original_shape)
                
                # Extract pixels for this image and reshape
                image_predictions = predictions[pixel_idx:pixel_idx + num_pixels]
                result_dict[image] = image_predictions.reshape(original_shape)
                
                pixel_idx += num_pixels
        
        return result_dict



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

data = Data()
data.load_models()

img_data = data.load_models()
ground_truths = data.load_ground_truths()

images_numbers = [7, 9, 17, 21]
image_names = [f"AAA_{num:03d}" for num in images_numbers]
segmentation_models = ["L_6_inference", "M_6_inference", "DA5_segs"]
linear_ensembles = linear_ensemble(image_names, segmentation_models, img_data)
linear_ensembles = {name: threshold_segmentation(seg) for name, seg in linear_ensembles.items()}

linear_results = evaluate_segmentations(ground_truths, linear_ensembles)
display_results(linear_results)

# Extract ground truths for evaluation


l = LogisticRegressor(image_names, ground_truths, segmentation_models, img_data)
train_images_nos = [1, 2, 3, 4, 5, 6]
train_images = [f"AAA_{num:03d}" for num in train_images_nos]

test_image_nos = [7, 9, 17, 21]
test_images = [f"AAA_{num:03d}" for num in test_image_nos]

l_model = l.train_model(train_images)
l_predictions = l.predict(l_model, test_images)

l_result = evaluate_segmentations(ground_truths, l_predictions)
display_results(l_result)