import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_img(img):
    img = np.asarray(img)
    plt.imshow(img)
    plt.show()

# 画像の読み込み
img = cv2.imread('./water_coins.jpg', 0)

    # ハフ変換による円検出
def detect_circles(img):
    img = cv2.medianBlur(img,7)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(cimg,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)

    return circles

    """
    for i in circles[0, :]:
        print(i)

    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('detected circles',cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """