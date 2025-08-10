import nrrd
import os
path = r"C:\Users\23139741\nnUNetFrame\Dataset\nnUNet_raw\organized_ct_data"

f1 = "3d_INNS04_segmentation.nrrd"
f2 = "3d_INNS04.nrrd"

g1 = nrrd.read(os.path.join(path, f1))[0]
g2 = nrrd.read(os.path.join(path, f2))[0]

print(g1.shape)
print(g2.shape)