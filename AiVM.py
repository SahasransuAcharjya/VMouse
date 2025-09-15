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

dragging = False
drag_counter = 0
drag_start_frames = 5  # Number of consecutive frames pinch must be sustained
release_counter = 0
drag_release_frames = 5  # Number of frames before releasing drag after pinch ends

cap = cv2.VideoCapture(0)  # Change to 1 if needed
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, modelC=1, detectionCon=0.7, trackCon=0.7)

# Screen size
wScr, hScr = autopy.screen.size()

while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    # 1. Detect hands and landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    x1 = y1 = x2 = y2 = None

    # 2. Get fingertip coordinates for index (8) and middle (12)
    if lmList and len(lmList) > 12:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Get fingertips for thumb (4) as well
    x_thumb = y_thumb = None
    if lmList and len(lmList) > 4:
        x_thumb, y_thumb = lmList[4][1:]

    # 4. Get which fingers are up
    fingers = detector.fingersUp()

    # Draw boundary rectangle for valid move zone
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    # Calculate distance between thumb and index finger (for pinch drag)
    length_pinching, img, lineInfo2 = detector.findDistance(4, 8, img)

    # ---- Pinch to drag with debounce logic ----
    if fingers and len(fingers) >= 2 and fingers[1] == 1 and fingers[0] == 1:
        if length_pinching < 30:
            drag_counter += 1
            release_counter = 0
            if drag_counter > drag_start_frames and not dragging:
                dragging = True
                autopy.mouse.toggle(down=True)
            if dragging and x1 is not None and y1 is not None:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                try:
                    autopy.mouse.move(wScr - clocX, clocY)
                except Exception:
                    pass
                plocX, plocY = clocX, clocY
        else:
            release_counter += 1
            drag_counter = 0
            if release_counter > drag_release_frames and dragging:
                dragging = False
                autopy.mouse.toggle(up=True)
    else:
        release_counter += 1
        drag_counter = 0
        if release_counter > drag_release_frames and dragging:
            dragging = False
            autopy.mouse.toggle(up=True)

    # ---- Moving mode: only index finger up (no drag) ----
    if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0 and x1 is not None:
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening
        try:
            autopy.mouse.move(wScr - clocX, clocY)
        except Exception:
            pass
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

    # ---- Clicking mode: index and middle fingers up ----
    if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1 and x1 is not None and x2 is not None:
        length, img, lineInfo = detector.findDistance(8, 12, img)
        if length < 40:
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
            try:
                autopy.mouse.click()
            except Exception:
                pass

    # Calculate and show FPS
    cTime = time.time()
    fps = 1 / max(1e-6, (cTime - pTime))
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
