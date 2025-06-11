import cv2

for index in [0, 1]:
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera with index {index} is working.")
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"Camera {index}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print(f"Cannot access camera {index}.")
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"Camera with index {index} is not accessible.")
