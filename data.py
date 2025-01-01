import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

capture = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgsize = 512

folder = "Data/anyLetterFolder" # Here, like instructed in the README file just change to whatever file you are trying to capture data to
counter = 0

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    # Cropping
    if hands:
        hand = hands[0]  # The one hand we have
        # Bounding box info
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + offset + h, x - offset:x + offset + w]

        if imgCrop.size == 0:  # Check if imgCrop is empty
            continue

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgsize))
            wGap = math.ceil((imgsize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgsize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgsize, hCal))
            hGap = math.ceil((imgsize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)  # One Second delay
    if key == ord("s"):
        if 'imgWhite' in locals():  # Check if imgWhite exists
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)
        else:
            print("No image to save")
