import  cv2
import numpy as np
import imutils
import  pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
resim=cv2.imread("araba5.jpeg",cv2.IMREAD_COLOR)
resim=cv2.resize(resim,(600,400))

gri=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY)
gri=cv2.bilateralFilter(gri,13,18,18)

kenar = cv2.Canny(gri,80,200)
contours = cv2.findContours(kenar.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

for c in contours:
    yay = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018* yay, True)

    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print("No contour detected")
else:
    detected = 1

if detected == 1:
    cv2.drawContours(resim, [screenCnt], -1, (0, 0, 255), 3)

mask=np.zeros(gri.shape, np.uint8)
yeni_resim = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
yeni_resim=cv2.bitwise_and(resim, resim, mask=mask)

(x,y)= np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
kirpilmis = gri[topx:bottomx + 1, topy:bottomy + 1]

text = pytesseract.image_to_string(kirpilmis, config='--psm 11')

resim=cv2.resize(resim,(600,400))
Cropped = cv2.resize(kirpilmis, (600, 400))
cv2.imshow('car', resim)
cv2.imshow('Cropped', kirpilmis)

cv2.waitKey(0)
cv2.destroyAllWindows()