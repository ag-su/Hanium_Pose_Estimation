import cv2
import datetime
import socket
import time

# 상태변화 리스트에 넣어줌
def stateAdd(preList, nowState):
    emptyList = preList
    emptyList.append(nowState)
    return emptyList

# 상태변화 값들중에 우는상태면 True 리턴, 아니면 False 리턴
def isStateCrying(preList):
    emptyList = preList

    # 평소, 우는, 자는 개수
    normal = emptyList.count(0)
    crying = emptyList.count(1)
    sleeping = emptyList.count(2)

    print("normal : %d, crying : %d, sleeping : %d"%(normal, crying, sleeping))
    if(crying > normal and crying > sleeping):
        return True

    return False


##################### 소켓 ###############################
host = '192.168.0.11' # 현재 자신의 호스트번호를 넣습니다.
port = 9999 # Arbitrary non-privileged port

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)

print("기다리는 중")
client_sock, addr = server_sock.accept()

print('Connected by', addr)

# 서버에서 "안드로이드에서 서버로 연결요청" 한번 받음
data = client_sock.recv(1024)
print(data.decode("utf-8"), len(data))

# 값하나 보냄(1)
i = 7
client_sock.send(i.to_bytes(4, byteorder='little'))

#################### 감지 ###############################
video_host = "192.168.0.12"   # 웹캠 호스트번호를 넣습니다.
video_url = "http://"+video_host+":8090/stream/snapshot.jpeg" # http://xxx.xxx.xx.xx:8090/stream/snapshot.jpeg
print("video_host : %s"%video_host)
print("video_url : %s"%video_url)

state = [] # 우는상태 감지한 순간부터 상태 저장할 리스트
flag = 0   # flag가 1이면 우는 경우(우는상태 발견하면 설정)
startTime = 0 # startTime = 우는순간 start
endTime = 0   # endTime = 10초 지난 후의 시간

try:
    #print("시도함")
    #cap = cv2.VideoCapture(video_url) # 웹캠주소 넣어주세요
    cap = cv2.VideoCapture('crying_Capture5.avi')
except:
    print("웹캠 연결할 수 없음.")

while(cap.isOpened()):
    '''
    try:
        print("시도함")
        cap = cv2.VideoCapture(video_url)
    except:
        print("웹캠 연결할 수 없음.")
    '''
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('frame',frame)
        cv2.imwrite("webcam4.jpg", frame)
        result = face.findStateForImage("webcam4.jpg")

        if(result == 0) :
            print("평소 상태")
            
        elif(result == 1) :
            print("우는 상태")
            flag = 1
            if(len(state) == 0): # 처음 우는상태 감지한 경우 시간 저장
                startTime = datetime.datetime.now()
                after_10second = datetime.timedelta(seconds=7)     # 10초간 우는거 설정(우는것 몇초 감지)
                endTime = startTime + after_10second
                print("-------------------------------------------------starttime : ", startTime.minute, "분", startTime.second, "초")                
        elif(result == 2) :
            print("자는 상태")

    # 우는게 한번 감지되면 10초간 모든 상태 추가
    if(flag == 1):
        state = stateAdd(state, result)
        if(datetime.datetime.now() > endTime):
            if(isStateCrying(state) == True):
                print("-------------------------------------------------7초간 우는상태에요!!!!!") 
                i = 1               # 값하나 보냄(1)
                client_sock.send(i.to_bytes(4, byteorder='little'))
                print("-------------------------------------------------endtime : ", endTime.minute, "분", endTime.second, "초")
                # 서버에서 "안드로이드에서 서버로 연결요청" 한번 받음
                data = client_sock.recv(1024)
                print(data.decode("utf-8"), len(data))
            state = []
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


i=99
client_sock.send(i.to_bytes(4, byteorder='little'))
cap.release()
cv2.destroyAllWindows()