import cv2
import time
import numpy as np
import sys
import os


BODY_PARTS_mpi = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}

BODY_PARTS_coco = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}

BODY_PARTS_BODY_25 = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}


points_total = []
def proc(frame_origin, frame_drawn, module, points_s):
    if module == "mpi":
        protoFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/mpi.prototxt"
        weightsFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/mpi.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13]]
        BODY_PARTS = {0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                  5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                  10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest",
                  15: "Background"}
    elif module == "coco":
        protoFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/coco.prototxt"
        weightsFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/coco.caffemodel"
        nPoints = 18
        POSE_PAIRS = [[1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
        BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                   5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
                   10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "REye",
                   15: "LEye", 16: "REar", 17: "LEar", 18: "Background"}
    
    else:
        protoFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/body_25.prototxt"
        weightsFile = "C:/Users/sooki/Downloads/opencv_pose_detection-master/opencv_pose_detection-master/src/fall_model/openpose_model/body_25.caffemodel"
        nPoints = 25
        POSE_PAIRS = [[0, 1], [0, 15], [0, 16], [1, 2], [1, 5], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [2, 3],
                      [3, 4], [5, 6], [6, 7], [10, 11], [13, 14], [15, 17], [16, 18], [14, 21], [19, 21], [20, 21],
                      [11, 24], [22, 24], [23, 24]]
        BODY_PARTS = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
                      5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip",
                      10: "RKnee", 11: "RAnkle", 12: "LHip", 13: "LKnee", 14: "LAnkle",
                      15: "REye", 16: "LEye", 17: "REar", 18: "LEar", 19: "LBigToe",
                      20: "LSmallToe", 21: "LHeel", 22: "RBigToe", 23: "RSmallToe", 24: "RHeel", 25: "Background"}

    frameWidth = frame_origin.shape[1]
    frameHeight = frame_origin.shape[0]
    threshold = 0.1
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    t = time.time()
    inWidth = 368
    inHeight = 368
    blob = cv2.dnn.blobFromImage(frame_origin, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    



    # 네트워크에 입력 
    net.setInput(blob)
    # 결과 받아오기 
    output = net.forward()

    print("완료 : {:.3f}".format(time.time() - t))
    H = output.shape[2]
    W = output.shape[3]
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        x = int(x)
        y = (frameHeight * point[1]) / H
        y = int(y)
        if prob > threshold : 
            points_s.append((x, y))
       
        else :
            points_s.append(None)
    print(points_s)
    points_total.append(points_s)
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points_s[partA] and points_s[partB]:
            cv2.line(frame_drawn, points_s[partA], points_s[partB], (0, 255, 255), 2)
            cv2.circle(frame_drawn, points_s[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

        