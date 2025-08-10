import os
import nrrd

uni_pc_path = r"E:"
laptop_path = r"Expansion:"

class Data:
    def __init__(self):
        pass

    def load_models(self, root = r"E:\HonoursData\mySegmentations", models = ["L_6_inference", "L_8_inference", "M_6_inference", "M_8_inference", "DA5_segs"], display=False):
        """
        Load .nrrd files from model directories, indexed by image name first.
        
        Returns: dict[image_name][model_name] -> {'filename', 'path', 'data', 'header'}
        """
        image_results = {}
        
        for model in models:
            
            model_path = os.path.join(root, model)
            image_names = [image for image in os.listdir(model_path) if image.endswith('.nrrd')]

            for image in image_names:
                full_path = os.path.join(model_path, image)
                image_key = image.split('.')[0]
                data_entry = {
                    "data": nrrd.read(full_path)[0],
                    "header": nrrd.read(full_path)[1],
                    "filename": image,
                    "path": full_path
                }
                if image_key not in image_results:
                    image_results[image_key] = {}
                image_results[image_key][model] = data_entry

        self.data = image_results
        return image_results

    def load_ground_truths(self, root = r"E:\HonoursData\GroundTruths", display=False):
        """
        Load ground truth data from the specified directory.
        
        Returns: dict[image_name] -> {'filename', 'path', 'data', 'header'}
        """
        groundtruths = {}
        gt_images = os.listdir(root)

        for image in gt_images:
            img_key = image.split('.')[0]
            groundtruths[img_key] = {
                "filename": image,
                "path": os.path.join(root, image),
                "data": nrrd.read(os.path.join(root, image))[0],
                "header": nrrd.read(os.path.join(root, image))[1]
            }

        self.groundtruths = groundtruths
        return groundtruths

if __name__ == "__main__":
    c = Data()

    uni_pc_path = r"E:"
    laptop_path = r"/media/joshua/Expansion"    
    
    segmentations_path = os.path.join(laptop_path, "HonoursData", "mySegmentations")
    ground_truths_path = os.path.join(laptop_path, "HonoursData", "GroundTruths")

    #print(c.load_models(segmentations_path))
    #print(c.load_ground_truths(ground_truths_path).keys())