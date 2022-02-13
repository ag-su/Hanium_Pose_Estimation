from numpy.core.fromnumeric import mean
from openpose import *
from yolo import *
import cv2 
import tensorflow as tf
import tensorflow.keras as keras
import pandas as pd



nms_boxes = {}
points = []


# 입력 사이즈 리스트 (Yolo 에서 사용되는 네크워크 입력 이미지 사이즈)
size_list = [320, 416, 608]

# open webcam (웹캠 열기)
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()
    
# loop through frames
while webcam.isOpened():
    t = time.time()
    hasFrame, frame =  webcam.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break
        
    points_s = []
    # 이미지 읽어오기
    frame_origin = frame  # 네트워크에 넣을 프레임
    frame_drawn = frame  # 텍스트 및 테두리가 그려질 프레임

    yolo(frame_origin=frame_origin, frame_drawn=frame_drawn, size=size_list[1], score_threshold=0.4, nms_threshold=0.4, nms_boxes=nms_boxes, points=points)
    if 'person_1' in nms_boxes.keys(): 
        proc(frame_origin=frame_origin, frame_drawn=frame_drawn, module='mpi', points_s=points_s)
        in_box(nms_boxes, points)
    
    #out = in_box(nms_boxes, points)
        means = [(139.24369068025726, 55.90679694845807), (137.22747512658003, 85.40134511206317), (124.62761542341919, 90.74913729350631), (127.13638397761926, 118.99152147232687), (128.019850005737, 123.55126663331406), (143.69175406645434, 90.90856045075718), (157.3921051590598, 118.91910458152358), (155.40593908762602, 129.95321948333205), (130.97522460460007, 136.5466913395634), (130.727788718079, 165.9675898869267), (127.69605423555723, 193.54245680696272), (138.16573875200817, 136.41232916158086), (143.78800391908496, 168.22905543195643), (144.774567455453, 197.5601670785256), (138.10705064215418, 115.88314110824592)]

        point_t = []
        for idx, point in enumerate(points_s):
            # 결측값 채우기
            if point == None:
                point_t.append(means[idx][0])
                point_t.append(means[idx][1])
            else:
                point_t.append(point[0])
                point_t.append(point[1])

        print("point_t: ", point_t)


        # 딥러닝 모델 불러오기 
        model = tf.keras.models.load_model('/dl_model/falldown_a.h5')
        res = model.predict(pd.DataFrame(point_t).T)
        # threshold = 0.7 (임시)
        if (res.argmax() == 1) and (res[0][res.argmax()]) >= 0.7:
             print("넘어짐")
        else:
            print("안 넘어짐")

    else:
        print("사람 없음")
    # display output
    cv2.imshow("webcam", frame_drawn)
    
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# release resources
webcam.release()
cv2.destroyAllWindows()   