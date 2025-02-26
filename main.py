import os
import cvzone
import cv2
from cvzone.PoseModule import PoseDetector
import time
import numpy as np

# Initialize video capture and pose detector
cap = cv2.VideoCapture(0)
detector = PoseDetector()

# Load shirt images
shirtsFolderPath = "Resources/Shirts"
listShirts = os.listdir(shirtsFolderPath)
print("Available Shirts:", listShirts)
fixedRatio = 262 / 190  # width of shirt / width of point 11 to 12
shirtRatioHeightWidth = 581 / 440
imageNumber = 0  # Start with the first shirt

# Cooldown variables
cooldown_time = 1  # Cooldown time in seconds
last_activation_time = 0  # Time of the last shirt change

# Shoulder region size (adjust as needed)
shoulder_region_width = 200  # Width of the shoulder region
shoulder_region_height = 200  # Height of the shoulder region

# Transparency level for the region boxes (0 = fully transparent, 1 = fully opaque)
transparency = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1277, 752))
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, draw=False, bboxWithHands=False)

    if lmList:
        # Get landmarks for shoulders (points 11 and 12) and wrists (points 15 and 16)
        lm11 = lmList[11][0:2]  # Left shoulder
        lm12 = lmList[12][0:2]  # Right shoulder
        lm15 = lmList[15][0:2]  # Left wrist
        lm16 = lmList[16][0:2]  # Right wrist

        # Load and resize the shirt image
        imgShirt = cv2.imread(os.path.join(shirtsFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)
        widthOfShirt = int((lm11[0] - lm12[0]) * fixedRatio)
        print("Shirt Width:", widthOfShirt)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        # Overlay the shirt on the image
        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

        # Get current time
        current_time = time.time()

        # Define shoulder regions
        left_shoulder_region = (
            lm11[0] - shoulder_region_width // 2,  # x1
            lm11[1] - shoulder_region_height // 2,  # y1
            lm11[0] + shoulder_region_width // 2,  # x2
            lm11[1] + shoulder_region_height // 2,  # y2
        )
        right_shoulder_region = (
            lm12[0] - shoulder_region_width // 2,  # x1
            lm12[1] - shoulder_region_height // 2,  # y1
            lm12[0] + shoulder_region_width // 2,  # x2
            lm12[1] + shoulder_region_height // 2,  # y2
        )

        # Check if the right wrist is in the right shoulder region
        if lm16:  # Ensure right wrist is detected
            x, y = lm16[0], lm16[1]
            if (
                right_shoulder_region[0] <= x <= right_shoulder_region[2]
                and right_shoulder_region[1] <= y <= right_shoulder_region[3]
            ):
                if current_time - last_activation_time > cooldown_time:  # Cooldown check
                    imageNumber = (imageNumber + 1) % len(listShirts)  # Cycle to the next shirt
                    last_activation_time = current_time  # Update last activation time
                    print("Right Wrist in Shoulder Region - Next Shirt:", listShirts[imageNumber])

        # Check if the left wrist is in the left shoulder region
        if lm15:  # Ensure left wrist is detected
            x, y = lm15[0], lm15[1]
            if (
                left_shoulder_region[0] <= x <= left_shoulder_region[2]
                and left_shoulder_region[1] <= y <= left_shoulder_region[3]
            ):
                if current_time - last_activation_time > cooldown_time:  # Cooldown check
                    imageNumber = (imageNumber - 1) % len(listShirts)  # Cycle to the previous shirt
                    last_activation_time = current_time  # Update last activation time
                    print("Left Wrist in Shoulder Region - Previous Shirt:", listShirts[imageNumber])

        # Create a copy of the image for transparent overlay
        overlay = img.copy()

        # Draw shoulder regions with transparency
        cv2.rectangle(overlay, (left_shoulder_region[0], left_shoulder_region[1]),
                      (left_shoulder_region[2], left_shoulder_region[3]), (0, 255, 0), -1)  # Filled rectangle
        cv2.rectangle(overlay, (right_shoulder_region[0], right_shoulder_region[1]),
                      (right_shoulder_region[2], right_shoulder_region[3]), (0, 255, 0), -1)  # Filled rectangle

        # Blend the overlay with the original image
        img = cv2.addWeighted(overlay, transparency, img, 1 - transparency, 0)

    # Display the image
    cv2.imshow("Virtual Try-On", img)
    cv2.waitKey(1)