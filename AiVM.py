import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

# Camera and smoothing settings
wCam, hCam = 640, 480
frameR = 100  # Frame reduction region
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)  # Change to 1 if necessary for your webcam
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, modelC=1, detectionCon=0.7, trackCon=0.7)

# Screen size
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    # 1. Find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    x1 = y1 = x2 = y2 = None

    # 2. Get the tip coordinates of the index and middle fingers if available
    if lmList and len(lmList) > 12:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
    fingers = detector.fingersUp()

    # Draw boundary rectangle
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # 4. Moving Mode - Only index finger up
    if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0 and x1 is not None:
        # 5. Convert coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        # 6. Smooth the mouse movement
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move the mouse
        try:
            autopy.mouse.move(wScr - clocX, clocY)
        except Exception:
            pass
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # 8. Clicking Mode - Index and middle fingers up
    if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1 and x1 is not None and x2 is not None:
        length, img, lineInfo = detector.findDistance(8, 12, img)
        if length < 40:
            # Corrected indexing here: use indices 4 and 5 for midpoint x,y
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            try:
                autopy.mouse.click()
            except Exception:
                pass

    # 9. Calculate and display FPS
    cTime = time.time()
    fps = 1 / max(1e-6, (cTime - pTime))
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 10. Show the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
