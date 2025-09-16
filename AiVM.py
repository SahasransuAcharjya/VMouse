import cv2
import numpy as np
import HandTrackingModule as htm
import time
import pyautogui

# Disable PyAutoGUI pause for faster interaction
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# Camera and smoothing settings
wCam, hCam = 320, 240
frameR = 100
smoothening = 7

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

dragging = False
drag_counter = 0
drag_start_frames = 5
release_counter = 0
drag_release_frames = 5

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1, modelC=0, detectionCon=0.5, trackCon=0.5)

wScr, hScr = pyautogui.size()

process_rate = 2
frame_counter = 0

try:
    while True:
        success, img = cap.read()
        if not success or img is None:
            continue

        frame_counter += 1
        if frame_counter % process_rate != 0:
            cv2.imshow("Image", img)
            cv2.waitKey(1)
            continue

        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)

        x1 = y1 = x2 = y2 = None

        if lmList and len(lmList) > 12:
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

        if lmList and len(lmList) > 4:
            x_thumb, y_thumb = lmList[4][1:]
        else:
            x_thumb = y_thumb = None

        fingers = detector.fingersUp()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        length_pinching, img, lineInfo2 = detector.findDistance(4, 8, img)

        # Pinch to drag
        if fingers and len(fingers) >= 2 and fingers[1] == 1 and fingers[0] == 1:
            if length_pinching < 30:
                drag_counter += 1
                release_counter = 0
                if drag_counter > drag_start_frames and not dragging:
                    dragging = True
                    pyautogui.mouseDown()
                if dragging and x1 is not None and y1 is not None:
                    x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                    y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                    clocX = plocX + (x3 - plocX) / smoothening
                    clocY = plocY + (y3 - plocY) / smoothening
                    try:
                        pyautogui.moveTo(wScr - clocX, clocY)
                    except Exception:
                        pass
                    plocX, plocY = clocX, clocY
            else:
                release_counter += 1
                drag_counter = 0
                if release_counter > drag_release_frames and dragging:
                    pyautogui.mouseUp()
                    dragging = False
        else:
            release_counter += 1
            drag_counter = 0
            if release_counter > drag_release_frames and dragging:
                pyautogui.mouseUp()
                dragging = False

        # Move mode - only index finger (no drag)
        if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 0 and x1 is not None and not dragging:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            try:
                pyautogui.moveTo(wScr - clocX, clocY)
            except Exception:
                pass
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Clicking mode - index + middle fingers (no drag)
        if fingers and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1 and x1 is not None and x2 is not None and not dragging:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                try:
                    pyautogui.click()
                except Exception:
                    pass

        cTime = time.time()
        fps = 1 / max(1e-6, (cTime - pTime))
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("Program interrupted by user.")
finally:
    cap.release()
    cv2.destroyAllWindows()
