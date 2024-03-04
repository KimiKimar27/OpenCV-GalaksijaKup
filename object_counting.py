import cv2
import urllib.request
import numpy as np
from scaler import scaleDownImage

url = 'http://0.0.0.0:8000/img2.jpg'
cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)

while True:
    # Load image from server
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgnp, -1)
    img2 = img.copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find and draw contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (11, 11), 0), 30, 150, 3)
    dilated = cv2.dilate(canny, (1, 1), iterations=2)
    (cnt, _) = cv2.findContours(dilated.copy(),
                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # k = img
    cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)

    # Display images
    scale = 0.4
    canny = scaleDownImage(canny, scale)
    img = scaleDownImage(img, scale)
    cv2.imshow("Contours", canny)
    cv2.imshow("Source", img)

    key = cv2.waitKey(5)

    if key == ord('q'):
        break

    elif key == ord('a'):
        print(f"Contours found: {len(cnt)}")
        for contour in cnt:
            print(cv2.contourArea(contour))

    elif key == ord('j'):
        threshold_area = 900
        # Iterate through each contour
        for contour in cnt:
            # Filter out small contours if necessary
            if cv2.contourArea(contour) < threshold_area:
                continue

            # create an empty image the size of the original
            mask = np.zeros_like(img2)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        
            # apply closing operation
            kernel = np.ones((100, 100), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            roi = cv2.bitwise_and(img2, mask)

            # Extract the region of interest (ROI)
            cv2.imshow("yabadpp", roi)

            # Calculate the average color within the ROI
            average_color = np.mean(roi, axis=(0, 1))

            print("Average color (BGR) for contour:", average_color)

            # Convert average_color to integer
            average_color = np.round(average_color).astype(int)

            # Display average color
            print("Average color (BGR) for contour (rounded):", average_color)

    elif key == ord('z'):
        # discard contours with areas under 900
        conts = []
        threshold_area = 900
        for contour in cnt:
            if cv2.contourArea(contour) > threshold_area:
                conts.append(contour)

        # create an empty image the size of the original
        mask = np.zeros_like(img2)
        cv2.drawContours(mask, conts, -1, (255, 255, 255), cv2.FILLED)
        

        # apply closing operation
        kernel = np.ones((100, 100), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        masked_image = cv2.bitwise_and(img2, mask)

        mask = scaleDownImage(mask, 0.4)
        cv2.imshow("mask", mask)
        masked_image = scaleDownImage(masked_image, 0.4)
        cv2.imshow("masked image", masked_image)

    elif key == ord('x'):
        # discard contours with areas under 900
        conts = []
        threshold_area = 900
        for contour in cnt:
            if cv2.contourArea(contour) > threshold_area:
                conts.append(contour)

        for cont in conts:
            # create an empty image the size of the original
            mask = np.zeros_like(img2)
            cv2.drawContours(mask, [cont], -1, (255, 255, 255), cv2.FILLED)

            # apply closing operation
            kernel = np.ones((100, 100), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masked_image = cv2.bitwise_and(img2, mask)

            mask = scaleDownImage(mask, 0.4)
            cv2.imshow("mask", mask)
            masked_image = scaleDownImage(masked_image, 0.4)
            cv2.imshow("masked image", masked_image)
            cv2.waitKey(0)
            cv2.destroyWindow("mask")
            cv2.destroyWindow("masked image")

cv2.destroyAllWindows()
