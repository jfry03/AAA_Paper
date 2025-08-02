import numpy as np
from analytics import dice_coefficient, display_results, evaluate_segmentations
import nrrd
from load_data import Data
from sklearn.linear_model import LogisticRegression, SGDClassifier
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline


class Model:
    def __init__(self, groundtruths, segmentation_models, segmentation_data):
        """
        Initialize the model with ground truths and segmentation models.
        
        Args:
            ground_truths (dict): Dictionary of ground truth segmentations.
            segmentation_models (list): List of segmentation model names.
            segmentation_data (dict): Data object containing all loaded segmentations.
        """
        self.ground_truths = groundtruths
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
                original_shape = self.segmentation_data[image][self.segmentation_models[0]]['data'].shape
                num_pixels = np.prod(original_shape)
                image_predictions = predictions[pixel_idx:pixel_idx + num_pixels]
                result_dict[image] = image_predictions.reshape(original_shape)
                pixel_idx += num_pixels
        return result_dict


class LogisticRegressor(Model):
    def __init__(self, ground_truths, segmentation_models, segmentation_data):
        super().__init__(ground_truths, segmentation_models, segmentation_data)

    def train_model(self, training_images, model_params={}):
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
        Train a scalable approximate RBF kernel SVM using RBFSampler + SGDClassifier.
        
        Args:
            training_images (list): List of image names to train on.
            model_params (dict): Optional parameters for SGDClassifier.
        
        Returns:
            Pipeline: Trained pipeline with RBF approximation and linear classifier.
        """
        X = self.build_feature_vector(training_images, self.segmentation_models)
        y = []
        for image in training_images:
            y += self.ground_truths[image]['data'].flatten().tolist()
        y = np.array(y)

        # Default model params if not provided
        default_params = {'loss': 'hinge', 'max_iter': 1000, 'tol': 1e-3}
        default_params.update(model_params)

        # RBF approximation + linear SVM
        rbf_feature = RBFSampler(gamma=1.0, n_components=50, random_state=42)
        sgd_clf = SGDClassifier(**default_params)
        self.model = make_pipeline(rbf_feature, sgd_clf)
        X = X.astype(np.float32)
        self.model.fit(X, y)
        return self.model


def evaluate_segmentations(ground_truths, segmentations):
    results = {}
    for image in ground_truths:
        if image in segmentations:
            gt_data = ground_truths[image]['data']
            seg_data = segmentations[image]
            dice_score = dice_coefficient(seg_data, gt_data)
            results[image] = dice_score
        else:
            results[image] = None
    return results


def threshold_segmentation(segmentation, threshold=0.5):
    return (segmentation > threshold).astype(np.uint8)


# === Load data ===
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

# === Train and predict ===
l = SVMClassifier(ground_truths, segmentation_models, segmentation_data)
train_images_nos = [1, 2, 3, 4, 5, 6]
train_images = [f"AAA_{num:03d}" for num in train_images_nos]

test_image_nos = [7, 9, 17, 21]
test_images = [f"AAA_{num:03d}" for num in test_image_nos]

l_model = l.train_model(train_images)
l_predictions = l.predict(test_images)

l_result = evaluate_segmentations(ground_truths, l_predictions)
display_results(l_result)
