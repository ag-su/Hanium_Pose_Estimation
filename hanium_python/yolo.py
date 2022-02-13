import cv2
import numpy as np


def yolo(frame_origin, frame_drawn, size, score_threshold, nms_threshold, nms_boxes, points):
    # YOLO 네트워크 불러오기
    net = cv2.dnn.readNet("C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/yolo_model/yolov4.weights", "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/yolo_model/yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # 이미지의 높이, 너비, 채널 받아오기
    height, width, channels = frame_origin.shape

    # 네트워크에 넣기 위한 전처리
    blob = cv2.dnn.blobFromImage(frame_origin, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    # 전처리된 blob 네트워크에 입력
    net.setInput(blob)

    # 결과 받아오기
    outs = net.forward(output_layers)

    # 각각의 데이터를 저장할 빈 리스트
    class_ids = []
    confidences = []
    boxes = []
    nms_boxes.clear()

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    print('│' + " YOLO ".center(90, '─') + '│')


    # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
    

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = classes[class_ids[i]]

                
            
            # 중복된 객체를 이름 뒤 숫자를 추가하여 구분 
            class_name += '_1'
            num = 1
            while class_name in nms_boxes.keys():
                num += 1
                class_name = class_name[:-1] + str(num)

            # 프레임에 작성할 텍스트 및 색깔 지정 
            label = f"{class_name} {confidences[i]:.2f}"
            color = colors[class_ids[i]]


            # 사각형 테두리 그리기 및 텍스트 쓰기
            cv2.rectangle(frame_drawn, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame_drawn, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
            cv2.putText(frame_drawn, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
         
            print('{0}│'.format(
                f"│ [{class_name}] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}".ljust(91, ' ')))
            nms_boxes[class_name] = [x, y, w, h]

    # return frame


classes = ["person", "bicycle", "car", "motorcycle",
           "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
           "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
           "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
           "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
           "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
           "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
           "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
           "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
           "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

def in_box(nms_boxes, points):
    print('|' + " IN BOX ".center(90, '-') + '|')
    for key, value in nms_boxes.items():
        if key.split('_')[0] != "person":  # person 오브젝트 제외 
            points_string = ""
            for index, point in enumerate(points):
                if point is not None and index != 25: # point가 없을 경우와 배경일 경우 제외 
                    x = value[0]
                    y = value[1]
                    width = value[0] + value[2]
                    height = value[1] + value[3]
                    if (point[0] > x) and (point[0] > width) and (point[1] > y) and (point[1] < height):
                        points_string += str(index)+' '
            if len(points_string) > 0:
                print('│{0}│'.format((f" [{key}] " + points_string).ljust(90, ' ')))
