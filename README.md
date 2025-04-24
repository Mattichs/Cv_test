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
- A **sliding window** approach was applied to test images across multiple scales.
- Each window was processed through the same BoVW pipeline, and the classifier predicted the object presence probability.

### 5. Post-Processing: Non-Maximum Suppression (NMS)
- **NMS** was used to refine the results by selecting the window with the highest number of overlapping predictions among those with high confidence.

## 📌 Observations

- The algorithm is conceptually valid and integrates key Computer Vision techniques like feature descriptors, BoVW, classification, and NMS.
- However, the **detection performance was limited**, likely due to the **small size of the test dataset**.
- Despite this, the approach was particularly interesting and educational, demonstrating how classical vision techniques can be applied in detection scenarios.
- We developed the idea togheter, Mattia suggest to use BoVW, Federico gives te idea to use SIFT for feature extraction.
