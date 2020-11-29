import cv2
import io
import socket
import struct
import time
import pickle
import zlib

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('127.0.0.1', 8485))
connection = client_socket.makefile('wb')

img_counter = 0

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

frame = cv2.imread('img.jpg')

result, frame = cv2.imencode('.jpg', frame, encode_param)
data = pickle.dumps(frame, 0)
size = len(data)


print("{}: {}".format(img_counter, size))
client_socket.sendall(struct.pack(">L", size) + data)
img_counter += 1



print("Receiving..")
with open('vidigot.mp4', 'wb') as fw:
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        fw.write(data)
    fw.close()
print("Received..")