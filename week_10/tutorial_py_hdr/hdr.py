import cv2 as cv
import numpy as np

# Loading exposure images into a list
img_fn = ["/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/week_10/tutorial_py_hdr/image0.jpeg", "/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/week_10/tutorial_py_hdr/image1.jpeg", "/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/week_10/tutorial_py_hdr/image2.jpeg", "/Users/janpoonthong/University/Year-3/Semester-1/selected_topic_in_computer_vision/week_10/tutorial_py_hdr/image3.jpeg"]
img_list = [cv.imread(fn) for fn in img_fn]

# Check if images are loaded correctly
if any(img is None for img in img_list):
    raise ValueError("One or more images failed to load. Check your file paths.")

# Resize images to the size of the first image
target_size = (img_list[0].shape[1], img_list[0].shape[0])  # (width, height)
img_list = [cv.resize(img, target_size) for img in img_list]

exposure_times = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

# Merge exposures to HDR image using Debevec method
merge_debevec = cv.createMergeDebevec()
hdr_debevec = merge_debevec.process(img_list, times=exposure_times.copy())

# Merge exposures to HDR image using Robertson method
merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(img_list, times=exposure_times.copy())

# Tonemap HDR images
tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())

# Exposure fusion using Mertens method
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(img_list)

# Convert datatype to 8-bit and save
res_debevec_8bit = np.clip(res_debevec * 255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')

# Save results
cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)

# Check if the Robertson method was successful before saving
if hdr_robertson is not None:
    res_robertson = tonemap1.process(hdr_robertson.copy())
    res_robertson_8bit = np.clip(res_robertson * 255, 0, 255).astype('uint8')
    cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
else:
    print("Robertson HDR processing failed.")

cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)
