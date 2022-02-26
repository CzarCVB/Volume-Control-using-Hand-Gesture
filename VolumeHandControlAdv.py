import time
import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

####Height and Width###
wCam, hCam = 1280, 720
####################

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(detect_conf=0.75, track_conf=0.75, maxHands=1)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
area = 0
colorVol = (255, 255, 0)
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmlist, bbox = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        #Filter based on size
        area = (bbox[2] - bbox[0])*(bbox[3] - bbox[1])//100
        if 250 < area < 1350:
            # Find Distance between index and thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # Convert Volume
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])
            # Reduce Resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer/smoothness)
            # Check which fingers are up
            fingers = detector.fingersUp()
            # if pinky is down, set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer/100, None)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (255, 255, 0), cv2.FILLED)
                colorVol = (52, 255, 255)
            else:
                colorVol = (255, 255, 0)

    #Volume Bar
    cv2.putText(img, f'Volume', (40, 140), cv2.FONT_HERSHEY_PLAIN, 2, (153, 0, 76), 3)
    cv2.rectangle(img, (50, 150), (85, 400), (255, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (52, 255, 255), 3)
    cVol = int(volume.GetMasterVolumeLevelScalar()*100)
    cv2.putText(img, f'Vol Set:{int(cVol)}%', (1000, 50), cv2.FONT_HERSHEY_PLAIN, 2, colorVol, 3)

    #fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    cv2.imshow("Volume Control w/ Gesture", img)
    cv2.waitKey(1)