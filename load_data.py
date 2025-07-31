import os
import nrrd

class Data:
    def __init__(self):
        self.data = {}

    def load_models(self, root = r"E:\HonoursData\mySegmentations", models = ["L_6_inference", "L_8_inference", "M_6_inference", "M_8_inference", "DA5_segs"], display=False):
        """
        Load .nrrd files from model directories, indexed by image name first.
        
        Returns: dict[image_name][model_name] -> {'filename', 'path', 'data', 'header'}
        """
        image_results = {}
        
        for model in models:
            model_path = os.path.join(root, model)
            if os.path.exists(model_path):
                for file in os.listdir(model_path):
                    if file.endswith('.nrrd'):
                        full_path = os.path.join(model_path, file)
                        try:
                            # Load the NRRD file data
                            data, header = nrrd.read(full_path)
                            
                            # Remove file extension for the key
                            file_key = os.path.splitext(file)[0]
                            
                            # Initialize image entry if it doesn't exist
                            if file_key not in image_results:
                                image_results[file_key] = {}
                            
                            # Add model result for this image
                            image_results[file_key][model] = {
                                'filename': file,
                                'path': full_path,
                                'data': data,
                                'header': header
                            }
                            if display:
                                print(f"Loaded: {model}/{file} - Shape: {data.shape}")
                        except Exception as e:
                            print(f"Error loading {full_path}: {e}")
            else:
                print(f"Directory not found: {model_path}")
        
        self.data = image_results
        return image_results

    def load_ground_truths(self, root = r"E:\HonoursData\GroundTruths", display=False):
        """
        Load ground truth data from the specified directory.
        
        Returns: dict[image_name] -> {'filename', 'path', 'data', 'header'}
        """
        image_results = {}
        
        if os.path.exists(root):
            for file in os.listdir(root):
                if file.endswith('.nrrd'):
                    full_path = os.path.join(root, file)
                    try:
                        # Load the NRRD file data
                        data, header = nrrd.read(full_path)
                        
                        # Remove file extension for the key
                        file_key = os.path.splitext(file)[0]
                        
                        # Initialize image entry if it doesn't exist
                        if file_key not in image_results:
                            image_results[file_key] = {}
                        
                        # Add ground truth result for this image
                        image_results[file_key] = {
                            'filename': file,
                            'path': full_path,
                            'data': data,
                            'header': header
                        }
                        if display:
                            print(f"Loaded ground truth: {file} - Shape: {data.shape}")
                    except Exception as e:
                        print(f"Error loading {full_path}: {e}")
        else:
            print(f"Ground truths directory not found: {root}")
        
        #self.data.update(image_results)
        return image_results

if __name__ == "__main__":
    c = Data()
    print(c.load_models().keys())
    print(c.load_ground_truths().keys())