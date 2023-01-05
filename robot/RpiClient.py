import socket
import threading
import pickle
import struct
import time
import numpy as np

import imageio as iio


class RpiClient:
    def __init__(self, addr="192.168.178.106", port=5050):
        self.HEADER = 64
        self.PORT = port
        self.SERVER = addr
        self.ADDR = (self.SERVER, self.PORT)
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "!DISCONNECT"
        self.client = None
        self.init()

        self.cam = iio.get_reader("<video0>")

    def init(self):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect(self.ADDR)

    def process_image(self):
        image = np.array(self.cam.get_data(0))

        data = pickle.dumps(image, 0)
        size = len(data)

        self.client.sendall(struct.pack(">L", size) + data)

        # conn, addr = self.client.accept()
        score = self.client.recv(1028)

        score = pickle.loads(score)
        return score

    def close(self):
        self.client.close()
        print(f"{self.SERVER} closed")


if __name__ == "__main__":
    client = RpiClient(addr="192.168.178.106", port=5050)
    cup_formation = client.process_image()
    print(cup_formation)


