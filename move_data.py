import os
import nrrd

path = r"D:\nnUNetFrame\Dataset\nnUNet_raw\organized_ct_data"


files = {}
results = {}
for file_name in os.listdir(path):
    if file_name.endswith(".nii"):
        continue
    is_segmentation = "segmentation" in file_name.lower()
    if is_segmentation:
        dimensions = file_name.split("_")[0]
        number = file_name.split("_")[1][4:]
        if (dimensions, number) not in results:
            results[(dimensions, number)] = {}
        results[dimensions, number]["segmentation"] = nrrd.read(os.path.join(path, file_name))[0]
    else:
        dimensions = file_name.split("_")[0]
        number = file_name.split("_")[1][4:-5]
        if (dimensions, number) not in results:
            results[(dimensions, number)] = {}
        results[dimensions, number]["image"] = nrrd.read(os.path.join(path, file_name))[0]

number_mapping = {"099": 20, "110": 21, "111": 22, "112": 23}
def mapping(dimensions, number):
    if dimensions == "3d":
        if number in number_mapping:
            number = number_mapping[number]
        else:
            number = int(number)
    
    else:
        number = int(number)
        number = number + 30
    return number

for key, item in results.items():
    results[key]["path"] = key

    

results = {mapping(*key): item for key, item in results.items()}

print("Items where Size does not match")
for key, val in results.items():
    if val['image'].shape != val['segmentation'].shape:
        print(f"Image: {val['image'].shape}, Label: {val['segmentation'].shape}")
        print(f"The image/segmentation for {val['path']} do not match")

results = {key: val for key, val in results.items() if val["image"].shape == val["segmentation"].shape}
"""
# Create sequential ordering from 1 to 43
original_keys = sorted(results.keys())
sequential_mapping = {original_key: i + 1 for i, original_key in enumerate(original_keys)}

# Apply sequential mapping
results = {sequential_mapping[key]: item for key, item in results.items()}

all_train = list(range(1, 33))

for fold in range(4):
    validation_set = all_train[fold * 8:(fold + 1) * 8]
    training_set = [x for x in all_train if x not in validation_set]
    
    # Create directories for this fold
    dataset_num = fold + 4  # Start from Dataset004
    fold_dir = os.path.join(os.path.dirname(path), f"Dataset{dataset_num:03d}_paper_fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Create subdirectories
    for subset in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']:
        os.makedirs(os.path.join(fold_dir, subset), exist_ok=True)
    
    # Save training images and labels
    for case_id in training_set:
        if case_id in results:
            # Save image
            if 'image' in results[case_id]:
                image_filename = f"case_{case_id:03d}_0000.nrrd"
                nrrd.write(os.path.join(fold_dir, 'imagesTr', image_filename), results[case_id]['image'])
            
            # Save segmentation
            if 'segmentation' in results[case_id]:
                seg_filename = f"case_{case_id:03d}.nrrd"
                nrrd.write(os.path.join(fold_dir, 'labelsTr', seg_filename), results[case_id]['segmentation'])
    
    # Save validation images and labels
    for case_id in validation_set:
        if case_id in results:
            # Save image
            if 'image' in results[case_id]:
                image_filename = f"case_{case_id:03d}_0000.nrrd"
                nrrd.write(os.path.join(fold_dir, 'imagesTs', image_filename), results[case_id]['image'])
            
            # Save segmentation
            if 'segmentation' in results[case_id]:
                seg_filename = f"case_{case_id:03d}.nrrd"
                nrrd.write(os.path.join(fold_dir, 'labelsTs', seg_filename), results[case_id]['segmentation'])
    
    print(f"Fold {fold}: Training set {len(training_set)} cases, Validation set {len(validation_set)} cases") 

# Create full training dataset (Dataset008)
full_train_dir = os.path.join(os.path.dirname(path), "Dataset008_paper_full_train")
os.makedirs(full_train_dir, exist_ok=True)

# Create subdirectories
for subset in ['imagesTr', 'imagesTs', 'labelsTr', 'labelsTs']:
    os.makedirs(os.path.join(full_train_dir, subset), exist_ok=True)

# Save all training cases (1-32) to training set
for case_id in all_train:
    if case_id in results:
        # Save image
        if 'image' in results[case_id]:
            image_filename = f"case_{case_id:03d}_0000.nrrd"
            nrrd.write(os.path.join(full_train_dir, 'imagesTr', image_filename), results[case_id]['image'])
        
        # Save segmentation
        if 'segmentation' in results[case_id]:
            seg_filename = f"case_{case_id:03d}.nrrd"
            nrrd.write(os.path.join(full_train_dir, 'labelsTr', seg_filename), results[case_id]['segmentation'])

# Save out-of-sample cases (33+) to test set
out_sample = [x for x in results.keys() if x > 32]
for case_id in out_sample:
    if case_id in results:
        # Save image
        if 'image' in results[case_id]:
            image_filename = f"case_{case_id:03d}_0000.nrrd"
            nrrd.write(os.path.join(full_train_dir, 'imagesTs', image_filename), results[case_id]['image'])
        
        # Save segmentation
        if 'segmentation' in results[case_id]:
            seg_filename = f"case_{case_id:03d}.nrrd"
            nrrd.write(os.path.join(full_train_dir, 'labelsTs', seg_filename), results[case_id]['segmentation'])

print(f"Full training dataset: {len(all_train)} training cases, {len(out_sample)} test cases")
"""