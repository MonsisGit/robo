import socket
import pickle
import struct
import pathlib
import logging

from model.predict import CupPredictor

logger = logging.getLogger(__name__)


class RpiServer:
    def __init__(self, addr="192.168.178.106", port=5050):

        self.HEADER = 64
        self.PORT = port
        self.FORMAT = 'utf-8'
        self.DISCONNECT_MESSAGE = "!DISCONNECT"
        self.SERVER = addr
        self.ADDR = (self.SERVER, self.PORT)
        self.server = None
        self.init()
        self.cup_predictor = CupPredictor(model_path=pathlib.Path('data/models/MLP.pth'),
                                          mlp_dims={'input_dim': 30,
                                                    'hidden_dim': 30,
                                                    'output_dim': 10,
                                                    'num_layers': 3},
                                          fov=[0, 0, 500, 500],
                                          verbose=False)

    def init(self):
        logger.info("[STARTING] server is starting...")
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # self.server.settimeout(1)
        self.server.bind(self.ADDR)

    def start(self):
        self.server.listen()
        logger.info(f"[LISTENING] Server is listening on {self.SERVER}")

        while True:
            try:
                conn, addr = self.server.accept()
                data = self.process_image(conn, addr)
                self.send(data, conn, addr)
            except Exception as e:
                logger.error(e)
                self.server.close()
                break

    def send(self, data, conn, addr):
        conn.send(data)
        logger.info(f'[SENDING] to {addr}')


    def process_image(self, conn, addr):
        logger.info(f'[PROCESSING] image from {addr}')
        data = b""
        payload_size = struct.calcsize(">L")
        while len(data) < payload_size:
            data += conn.recv(4096)

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        #data = data[msg_size:]

        image = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

        preds = self.cup_predictor.predict(image)

        data = pickle.dumps(preds, 0)
        return data


    def close(self):
        self.server.close()
        logger.info("[CLOSING] server is closing...")


if __name__ == '__main__':
    # ps -fA | grep python
    # netstat -tulpn
    server = RpiServer(addr="192.168.178.106")
    server.start()
