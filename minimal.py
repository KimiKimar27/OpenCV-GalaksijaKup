import cv2
import urllib.request
import numpy as np
from scaler import scaleDownImage

cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)

# Load image from server
url = 'http://0.0.0.0:8000/img2.jpg'
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
cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)

# Discard contours with areas under 900
threshold_area = 900
contours = [contour for contour in cnt if cv2.contourArea(contour) > threshold_area]
        
for cont in contours:
    # Create an empty image the size of the original
    mask = np.zeros_like(img2)
    cv2.drawContours(mask, [cont], -1, (255, 255, 255), cv2.FILLED)

    # Apply closing operation
    # This is done because the contours sometimes have small gaps and thus can't be filled in
    kernel = np.ones((100, 100), np.uint8) # These numbers can be tweaked to fill smaller or larger gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # ???

    # Get the current object from the image
    masked_image = cv2.bitwise_and(img2, mask)

    # Display current object and its mask
    mask = scaleDownImage(mask, 0.4)
    masked_image = scaleDownImage(masked_image, 0.4)
    cv2.imshow("mask", mask)
    cv2.imshow("masked image", masked_image)

    # Destroy those windows after any key press
    cv2.waitKey(0)
    cv2.destroyWindow("mask")
    cv2.destroyWindow("masked image")


# Display images
scale = 0.4
canny = scaleDownImage(canny, scale)
img = scaleDownImage(img, scale)
cv2.imshow("Contours", canny)
cv2.imshow("Source", img)

# Exit program
cv2.waitKey(0)
cv2.destroyAllWindows()
