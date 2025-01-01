# Hand Sign Recognition Program

This project is a hand sign recognition system designed to identify hand signs for letters A-Z. It involves collecting data, training a model using the collected data, and testing the trained model for recognition. The system utilizes Python, OpenCV, and TensorFlow.

## Features

### 1. **Data Collection**
- **Folder Structure:**
  - A `Data` folder contains subfolders named A-Z, one for each letter.
  - Each folder stores images of hand signs for the corresponding letter.

- **Image Capture:**
  - The `data.py` script is used to capture and save images for training.
  - Steps to capture images:
    1. Set the folder to save images by editing the `folder` variable in the `data.py` file.
       ```python
       folder = "Data/anyLetterFolder"
       ```
       Replace `anyLetterFolder` with the desired folder, e.g., for letter A:
       ```python
       folder = "Data/A"
       ```
    2. Run the `data.py` script.
    3. The camera will turn on. Make the hand sign symbol for the specified letter.
    4. Press `s` on the keyboard to save the image to the folder.
    5. Repeat this step to collect at least 300+ images per letter for better accuracy.

### 2. **Model Training**
- Use the [Teachable Machine](https://teachablemachine.withgoogle.com/train/image) to train the model:
  1. Upload the collected images for each letter.
  2. Train the model using the provided interface.
  3. Export the trained model. You will get two files:
     - `keras_model.h5`
     - `labels.txt`

### 3. **Model Integration**
- Save the exported files in the following locations:
  - Place the `keras_model.h5` file in the `Model` folder.
  - If using custom hand signs or different labels, replace the `labels.txt` file in the `Model` folder with your new labels.

### 4. **Testing the Model**
- Run the `test.py` file to test the trained model.
- The program will use your webcam to detect hand signs and recognize them in real-time.

## How to Set Up

1. **Folder Structure:**
   - Under the `Data` folder, delete the tempFile file and create subfolders named A-Z.

2. **Data Collection:**
   - Use the `data.py` script to capture images for each letter as described above.

3. **Model Training:**
   - Train your model using the [Teachable Machine](https://teachablemachine.withgoogle.com/train/image) platform.
   - Export the trained model and save the files as specified above.

4. **Testing:**
   - Run the `test.py` script to validate your model's performance.

## Technologies Used

![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/tensorflow-%23FF6F00.svg?style=for-the-badge&logo=tensorflow&logoColor=white)

## Files and Structure

- **Python Files:**
  - `data.py`: Script for collecting images.
  - `test.py`: Script for testing the trained model.

- **Folders:**
  - `Data`: Contains subfolders A-Z for storing collected images.
  - `Model`: Contains the trained model (`keras_model.h5`) and `labels.txt`.

## Error Handling

- **Image Capture:**
  - Validates the presence of the specified folder before saving images.

- **Model Testing:**
  - Displays user-friendly error messages for invalid or missing model files.

## Future Enhancements

- Add real-time feedback to improve hand sign accuracy during testing.
- Implement a GUI for easier navigation and usability.
- Expand the system to include additional hand signs or gestures.

---

This project is designed to simplify the process of recognizing hand signs. Ensure your `Data` folder is properly populated before training your model.
