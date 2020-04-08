import pytesseract
import cv2
from lib_detection import load_model, detect_lp, im2single
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path of the image")
args = vars(ap.parse_args())

img_path =args["image"]
char_list =  '0123456789ABCDEFGHKLMNPRSTUVXYZ-.'

# Ham fine tune bien so, loai bo cac ki tu khong hop ly
def fine_tune(lp):
    newString = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            newString += lp[i]
    return newString

wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)


Ivehicle = cv2.imread(img_path)
cv2.imshow("Anh goc",Ivehicle)
cv2.waitKey(0)



Dmax = 608
Dmin = 288


ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
side = int(ratio * Dmin)
bound_dim = min(side, Dmax)

_ , LpImg, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)


if (len(LpImg)):

    # Chuyen doi anh bien so
    LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
    cv2.waitKey()
    # Chuyen anh bien so ve gray
    gray = cv2.cvtColor( LpImg[0], cv2.COLOR_BGR2GRAY)
    #gr = cv2.threshold( gr,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU )[1]
    #cv2.imwrite('GrayLP', gr)
    cv2.imshow("Bien so xam", gray)
    cv2.waitKey()
    binary = cv2.threshold(gray, 127, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow("Anh bien so sau threshold", binary)
    cv2.waitKey()
    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 7")
    print(text)
    cv2.putText(Ivehicle,fine_tune(text),(50, 50), cv2.FONT_HERSHEY_PLAIN, 4.0, (255, 255, 255), lineType=cv2.LINE_AA)

    # Hien thi anh va luu anh ra file output.png
    cv2.imshow("Anh input", Ivehicle)
    cv2.imwrite("output.png",Ivehicle)
    cv2.waitKey()
cv2.destroyAllWindows()