import cv2
import mediapipe as mp
import time
import math

class handDetector:
    def __init__(self, mode=False, maxHands=2, detect_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detect_conf, self.track_conf)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handnum=0, draw=True):
        xList = []
        yList = []
        self.lmlist = []
        bbox = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handnum]
            for ID, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([ID, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (bbox[0]-25, bbox[1]-25), (bbox[2]+25, bbox[3]+25), (0, 255, 0), 2)

        return self.lmlist, bbox

    def fingersUp(self):
        fingers = []

        #thumb
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #4 fingers
        for id in range(1, 5):
            if self.lmlist[self.tipIds[id]][2] < self.lmlist[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        x3, y3 = int((x1+x2)/2), int((y1+y2)/2)

        cv2.circle(img, (x1, y1), 10, (153, 0, 76), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (153, 0, 76), 3)
        cv2.circle(img, (x2, y2), 10, (153, 0, 76), cv2.FILLED)
        cv2.circle(img, (x3, y3), 10, (255, 8, 255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, x3, y3]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPosition(img)
        if len(lmlist) != 0:
            print(lmlist[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()