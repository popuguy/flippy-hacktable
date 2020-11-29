import socket
import sys
import cv2
import pickle
import numpy as np
import struct ## new
import zlib
from image_to_flip import save_flipbook

HOST=''
PORT=8485

filename = 'vid.mp4'

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
print('Socket created')

s.bind((HOST,PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')

conn,addr=s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
while True:
    while len(data) < payload_size:
        print("Recv: {}".format(len(data)))
        data += conn.recv(4096)

    print("Done Recv: {}".format(len(data)))
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    print("msg_size: {}".format(msg_size))
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    save_flipbook('sdf', frame, save_file=filename)


    # enter me ratting
    # the way I was sending a file before was
    # print('Opening file ', text_file)
    # with open(text_file, 'ab+') as fa:
    #     print('Opened file')
    #     print("Appending string to file.")
    #     string = b"Append this to file."
    #     fa.write(string)
    #     fa.seek(0, 0)
    #     print("Sending file.")
    #     while True:
    #         data = fa.read(1024)
    #         conn.send(data)
    #         if not data:
    #             break
    #     fa.close()
    #     print("Sent file.")










    
    # with open('vid.mp4', 'rb') as fa:
    with open(filename, 'rb') as fa:
        print('Opened file')
        fa.seek(0, 0) #not sure
        print("Sending file.")
        while True:
            data = fa.read(1024)
            conn.send(data)
            if not data:
                break
        fa.close()
        print("Sent file.")





    cv2.imshow('ImageWindow',frame)
    cv2.waitKey(1)