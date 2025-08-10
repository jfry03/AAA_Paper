import numpy as np
import pandas as pd
from load_segmentations import Data
from analytics import dice_coefficient
import os
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self, name):
        self.name = name
        self.trained_model = None

  
    
    def load_data(self, data):
        self.data = data

    def build_design_matrix(self, images, models):
        self.models = models
        X = []
        for model in models:
            feature = []
            for image in images:
                feature += self.data.data[image][model]['data'].flatten().tolist()
            X.append(feature)
        
        y = []
        for image in images:
            y += self.data.groundtruths[image]['data'].flatten().tolist()

        return np.array(X).T, y
    

    def predict(self, images):
        if self.trained_model is None:
            raise ValueError(f"Model {self.name} has not yet been trained")
        
        X, _ = self.build_design_matrix(images, self.models)

        self.predictions = self.trained_model.predict(X)
        self.test_images = images
        return self.predictions
    
    def get_dice(self):
        _, y = self.build_design_matrix(self.test_images, self.models)
        return dice_coefficient(self.predictions, y)
        

class LogisticRegressionAAA(Model):
    def train(self, images, models, params={}):
        X, y = self.build_design_matrix(images ,models)
    
        self.trained_model = LogisticRegression(**params)
        self.trained_model.fit(X, y)

   


uni_pc_path = r"E:"
segmentations_path = os.path.join(uni_pc_path, "HonoursData", "mySegmentations")
ground_truths_path = os.path.join(uni_pc_path, "HonoursData", "GroundTruths")

data = Data()


training_images = [f"AAA_{num:03d}" for num in [1, 2, 3, 4, 5, 6, 7, 9]]
testing_images = [f"AAA_{num:03d}" for num in [17, 21]]

models = ["DA5_segs", "L_8_inference", "M_8_inference"]

logreg = LogisticRegressionAAA("LogisticRegression")

data.load_models(segmentations_path)
data.load_ground_truths(ground_truths_path)

logreg.load_data(data)
logreg.train(training_images, models)


logreg.predict(testing_images)

print(logreg.get_dice())






