# server.py : 연결해 1 보낼 수 있음.
import socket

host = '192.168.200.107' # 호스트 ip를 적어주세요
port = 9999            # 포트번호를 임의로 설정해주세요

server_sock = socket.socket(socket.AF_INET)
server_sock.bind((host, port))
server_sock.listen(1)

print("기다리는 중")
client_sock, addr = server_sock.accept()

print('Connected by', addr)
data = client_sock.recv(1024)
print(data.decode("utf-8"), len(data))

data2 = int(input("보낼 값 : "))
#print(data2.encode())
client_sock.send(data)
client_sock.send(data2.to_bytes(4, byteorder='little'))

client_sock.close()
server_sock.close()