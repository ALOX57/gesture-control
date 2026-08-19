import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Gesture Control", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()