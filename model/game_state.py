import numpy as np


class GameState:
    def __init__(self, state):
        self.cup_positions = []

    def update_positions(self, cup_positions: list):
        cup_positions = self.bbox_supressions(cup_positions)
        cup_positions = self.negative_cup_supressions(cup_positions)
        self.cup_positions = cup_positions

    def bbox_supressions(self, bbox: list, cup_positions: list) -> list:
        cup_positions = []
        return cup_positions

    def negative_cup_supressions(self, cup_box: np.array, cup_positions: list) -> list:
        cup_positions = []
        return cup_positions

    def predict_cup_positions(self, predictions: np.array) -> list:
        frame = self.get_frame()
        cup_positions = []
        return cup_positions

    def get_frame(self, ) -> np.array:
        frame = np.zeros((512, 512, 3))
        return frame

    def get_positions(self) -> list:
        return self.cup_positions
