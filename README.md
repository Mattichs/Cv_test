# Object Detection with BoVW and SIFT (Cognolatto Federico & Bastianello Mattia)

This project implements an object detection pipeline based on the Bag of Visual Words (BoVW) model and SIFT features. The approach was chosen for its strong connection to classical Computer Vision techniques covered in the course.

## 🔍 Methodology

### 1. Feature Extraction
- **SIFT** (Scale-Invariant Feature Transform) was used to detect keypoints and extract descriptors from each image.

### 2. Visual Vocabulary and BoVW
- Descriptors from all training images were clustered using **k-means** to create a visual vocabulary.
- Each image was then represented as a **histogram of visual word occurrences**.

### 3. Classifier Training
- A machine learning classifier (Random Forest) was trained on the BoVW histograms to distinguish between object classes.

### 4. Object Detection
- A **sliding window** approach was applied to test images, we used fixed 256x256 windows witha stride of 16 (this can be extend to more window dimensions and different strides to be more precise)
- Each window was processed through the same BoVW pipeline, and the classifier predicted the object presence probability.

### 5. Post-Processing: Non-Maximum Suppression (NMS)
- **NMS** was used to refine the results by selecting the window with the highest number of overlapping predictions among those with high confidence.

## 📌 Observations

- The algorithm is conceptually valid and integrates key Computer Vision techniques like feature descriptors, BoVW, classification, and NMS.
- However, the **detection performance was limited**, likely due to the **small size of the test dataset** or to an our miss understanding of the methods, but we try differents methods and we decide to use this one at the end (we spend many days trying to find the best method XD).
- Despite this, the approach was particularly interesting and educational, demonstrating how classical vision techniques can be applied in detection scenarios.
- We developed the idea togheter, Mattia suggest to use BoVW, Federico gives te idea to use SIFT for feature extraction.

## Prepare the data to be process
Create, if not already present, a folder name "test_images " with test images inside. Then, create another folder named "ground_truth_labels", if not already present,  with inside the labels associated to the test images.

## To run 
mkdir build
cd build
cmake ..
make
./detect

