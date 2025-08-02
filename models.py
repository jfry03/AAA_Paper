import numpy as np
from analytics import dice_coefficient, display_results, evaluate_segmentations
import nrrd
from load_data import Data
from sklearn.linear_model import LogisticRegression
import os
from sklearn.svm import SVC


class Model:
    def __init__(self, groundtruths, segmentation_models, segmentation_data):
        """
        Initialize the model with ground truths and segmentation models.
        
        Args:
            ground_truths (dict): Dictionary of ground truth segmentations.
            segmentation_models (list): List of segmentation model names.
            data: Data object containing all loaded segmentations.
        """
        self.ground_truths = ground_truths
        self.segmentation_models = segmentation_models
        self.segmentation_data = segmentation_data
        self.model = None

    def build_feature_vector(self, images, segmentation_models):
        X = []
        for model in segmentation_models:
            feature = []
            for image in images:
                feature += self.segmentation_data[image][model]['data'].flatten().tolist()
            X.append(feature)
        return np.array(X).T
    

    def predict(self, test_images):
        X = self.build_feature_vector(test_images, self.segmentation_models)
        predictions = self.model.predict(X)

        # Reshape predictions back to image format
        result_dict = {}
        pixel_idx = 0

        for image in test_images:
            if image in self.segmentation_data:
                # Get original shape from the first available model
                original_shape = self.segmentation_data[image][self.segmentation_models[0]]['data'].shape
                num_pixels = np.prod(original_shape)

                # Extract pixels for this image and reshape
                image_predictions = predictions[pixel_idx:pixel_idx + num_pixels]
                result_dict[image] = image_predictions.reshape(original_shape)

                pixel_idx += num_pixels             
        return result_dict
    
class LogisticRegressor(Model):
    def __init__(self, ground_truths, segmentation_models, segmentation_data):
        super().__init__(ground_truths, segmentation_models, segmentation_data)

    def train_model(self, training_images, model_params={}):
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

        self.model = LogisticRegression(**model_params)
        self.model.fit(X, y)
        
        return self.model
    
class SVMClassifier(Model):
    def __init__(self, ground_truths, segmentation_models, segmentation_data):
        super().__init__(ground_truths, segmentation_models, segmentation_data)

    def train_model(self, training_images, model_params={}):
        """
        Train the SVM classifier on the provided training images.
        Features are built using all segmentation models as columns.
        
        Args:
            training_images (list): List of image names to train on.
        
        Returns:
            SVC: Trained SVM classifier.
        """        
        X = self.build_feature_vector(training_images, self.segmentation_models)
        y = []

        for image in training_images:
            y += self.ground_truths[image]['data'].flatten().tolist()
        y = np.array(y)

        self.model = SVC(**model_params)
        self.model.fit(X, y)
        
        return self.model









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

uni_pc_path = r"E:"
laptop_path = r"/media/joshua/Expansion"    

segmentations_path = os.path.join(laptop_path, "HonoursData", "mySegmentations")
ground_truths_path = os.path.join(laptop_path, "HonoursData", "GroundTruths")

data.load_models(segmentations_path)

data.load_ground_truths(ground_truths_path)

segmentation_data = data.data
ground_truths = data.groundtruths


segmentation_models = ["L_6_inference", "M_6_inference", "DA5_segs"]

l = LogisticRegressor(ground_truths, segmentation_models, segmentation_data)
train_images_nos = [1, 2, 3, 4, 5, 6]
train_images = [f"AAA_{num:03d}" for num in train_images_nos]

test_image_nos = [7, 9, 17, 21]
test_images = [f"AAA_{num:03d}" for num in test_image_nos]

l_model = l.train_model(train_images)
l_predictions = l.predict(test_images)

l_result = evaluate_segmentations(ground_truths, l_predictions)
display_results(l_result)