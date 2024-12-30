import cv2
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Make sure keras_model.h5 was trained on 512x512
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

#for testingn I used 27k images from kaggle to train on you can use whatever resource and trained machine you have
#but make sure to change the imgsize accordingle
offset = 20
imgsize = 512

labels = ["A", "B", "C", "D", "E", "F", "G", "H",
          "I", "J", "K", "L", "M", "N", "O", "P",
          "Q", "R", "S", "T", "U", "V", "W", "X",
          "Y", "Z"]

while True:
    success, img = capture.read()
    if not success:
        break

    # This is our display image (unmodified)
    imgOutput = img.copy()

    # Detect the hand
    hands, img = detector.findHands(img)  # draws a bounding box / landmarks

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank 512x512 white image
        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255

        # Crop around the bounding box with offset
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            # If hand is taller than wide
            if aspectRatio > 1:
                scale = imgsize / h
                wCal = math.ceil(scale * w)
                # Resize to (wCal x 512)
                imgResize = cv2.resize(imgCrop, (wCal, imgsize))
                # center it horizontally
                wGap = math.ceil((imgsize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize

            # If hand is wider than tall
            else:
                scale = imgsize / w
                hCal = math.ceil(scale * h)
                # Resize to (512 x hCal)
                imgResize = cv2.resize(imgCrop, (imgsize, hCal))
                # center it vertically
                hGap = math.ceil((imgsize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # run the classifier
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

            # show classification result on the webcam image
            cv2.putText(imgOutput, labels[index], (x, y - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

            # draw bounding box for visualization
            cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

            # debug windows
            cv2.imshow("imageCrop", imgCrop)
            cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):  # press 'q' to quit
        break

capture.release()
cv2.destroyAllWindows()
