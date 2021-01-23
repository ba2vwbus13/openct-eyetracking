import cv2
import numpy as np


def gstreamer_pipeline(
    capture_width=320,
    capture_height=180,
    display_width=320,
    display_height=180,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def red_detect(img,b,g,r,mb,mg,mr,gmin,gmax):
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # s_magnification = 2
    # v_magnification = 3

    # img_hsv[:, :, (1)] = img_hsv[:, :, (1)]*s_magnification
    # img_hsv[:, :, (2)] = img_hsv[:, :, (2)]*v_magnification

    # img_bar = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


    # HSV色空間に変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # 赤色のHSVの値域1
    hsv_min = np.array([mb,mg,mr])
    hsv_max = np.array([b,g,r])
    mask1 = cv2.inRange(hsv, hsv_min, hsv_max)
    _, threshold = cv2.threshold(mask1, 5, 255, cv2.THRESH_BINARY)    
    _, _gray = cv2.threshold(gray, gmin, gmax, cv2.THRESH_BINARY_INV) 

    return threshold,hsv,_gray

def contour(threshold, img):
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        cv2.circle(img, (cx, cy), 8, (255, 0, 0), 2)
    except:
        pass


def nothing(x):
    pass


def main():
    # カメラのキャプチャ
    #cap = cv2.VideoCapture(0)
    print(gstreamer_pipeline(flip_method=0))
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    fmt = cv2.VideoWriter_fourcc('m','p','4','v')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(cap.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter('./result.mp4',fmt,int(cap.get(cv2.CAP_PROP_FPS)),size)
    #img2 = cv2.imread("eye.jpg")

    cv2.namedWindow("gui")
    cv2.createTrackbar("R","gui",0,255,nothing)
    cv2.createTrackbar("G","gui",0,255,nothing)
    cv2.createTrackbar("B","gui",0,255,nothing)
    cv2.createTrackbar("MR","gui",0,255,nothing)
    cv2.createTrackbar("MG","gui",0,255,nothing)
    cv2.createTrackbar("MB","gui",0,255,nothing)
    cv2.createTrackbar("Switch","gui",0,1,nothing)
    cv2.createTrackbar("gmin","gui",0,255,nothing)
    cv2.createTrackbar("gmax","gui",0,255,nothing)
    img = np.zeros((300,512,3), np.uint8)

    isRecoading = False

    
    while(cap.isOpened()):

        cv2.imshow("gui",img)
        r = cv2.getTrackbarPos('R','gui')
        g = cv2.getTrackbarPos('G','gui')
        b = cv2.getTrackbarPos('B','gui')
        mr = cv2.getTrackbarPos("MR","gui")
        mg = cv2.getTrackbarPos("MG","gui")
        mb = cv2.getTrackbarPos("MB","gui")
        switch = cv2.getTrackbarPos("Switch","gui")
        gmin = cv2.getTrackbarPos("gmin","gui")
        gmax = cv2.getTrackbarPos("gmax","gui")
        if switch == 0:
            img[:] = [b,g,r]
        else:
            img[:] = [mb,mg,mr]

        # フレームを取得
        ret, frame = cap.read()
        #frame = img2
        #ret, frame = cv2.rotate(cap.read(),cv2.ROTATE_90_CLOCKWISE)


        # 目検出
        mask,hsv,gray = red_detect(frame,b,g,r,mb,mg,mr,gmin,gmax)
        contour(mask, frame)
        # 結果表示
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("hsv", hsv)
        cv2.imshow("gray",gray)
        
        writer.write(frame)

        # qキーが押されたら途中終了
        if cv2.waitKey(25) & 0xFF == ord('q'):
            #cv2.imwrite("eye.jpg",frame)
            break
        if cv2.waitKey(25) & 0xFF == ord('r'):
            isRecoading = not isRecoading

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
