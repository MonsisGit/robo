import RPi.GPIO as GPIO
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class RobotController:
    def __init__(self, DIR_PINS: dict = {"MOTOR_Z": 20, "MOTOR_X": 26},
                 PUL_PINS: dict = {"MOTOR_Z": 21, "MOTOR_X": 19},
                 END_STOP_PINS: dict = {"MOTOR_X": 0},
                 ENABLE_PINS: dict = {"MOTOR_X": 13},
                 MICROSTEP_RES_PINS: tuple = (14, 15, 18),
                 RELAY_PIN: int = 6,
                 ENDSTOP_PIN: int = 22):

        self.DIR_PINS = DIR_PINS
        self.PUL_PINS = PUL_PINS
        self.MICROSTEP_RES_PINS = MICROSTEP_RES_PINS
        self.ENDSTOP_PIN = ENDSTOP_PIN
        self.END_STOP_PINS = END_STOP_PINS
        self.ENABLE_PINS = ENABLE_PINS
        self.MOTORS = dict()
        self.initialize_controller()
        logger.setLevel('INFO')

    def initialize_controller(self):
        GPIO.setmode(GPIO.BCM)
        for pin in [*self.DIR_PINS.values(), *self.PUL_PINS.values(), *self.ENABLE_PINS.values()]:
            GPIO.setup(pin, GPIO.OUT)
            logger.info(f'Initialized {pin}')

    def set_direction(self, motor, clockwise):
        GPIO.output(self.DIR_PINS[motor], clockwise)
        time.sleep(0.1)

    def enable(self, motor):
        GPIO.output(self.ENABLE_PINS[motor], GPIO.LOW)
        time.sleep(0.1)

    def disable(self, motor):
        GPIO.output(self.ENABLE_PINS[motor], GPIO.HIGH)
        time.sleep(0.1)

    def _run(self, motor, steps, step_delay, ramp, ramp_length = 10):
        if not ramp:
            for _ in range(steps):
                GPIO.output(self.PUL_PINS[motor], GPIO.HIGH)
                time.sleep(step_delay)
                GPIO.output(self.PUL_PINS[motor], GPIO.LOW)
                time.sleep(step_delay)

        if ramp == 'linear':
            fac = np.arange(4,0.9,-1/ramp_length)
            for step in range(steps):
                if step < ramp_length:
                    step_delay_temp = step_delay*fac[step] 
                else:
                    step_delay_temp = step_delay

                GPIO.output(self.PUL_PINS[motor], GPIO.HIGH)
                time.sleep(step_delay_temp)
                GPIO.output(self.PUL_PINS[motor], GPIO.LOW)
                time.sleep(step_delay_temp)


    def run_motor(self, motor: str, clockwise: bool = False,
                  steps: int = 100, step_delay: float = 0.01,
                  ramp:str = "linear"):
        
        self.set_direction(motor, clockwise)
        self.enable(motor)
        self._run(motor, steps, step_delay, ramp)
        self.disable(motor)


    def home_all(self):
        for motor, END_STOP_PIN in self.END_STOP_PINS.items():
            self.home(motor,END_STOP_PIN)

    def home(self, motor: str, END_STOP_PIN: int):
        while not GPIO.event_detected(END_STOP_PIN):
            self.run_motor(motor=motor, steps=5)
            time.sleep(0.05)
        logger.info(f'Endstop {END_STOP_PIN} for {motor} reached!')


if __name__ == '__main__':

    controller = RobotController()
    controller.run_motor(motor='MOTOR_X',
                        steps=500,
                        step_delay=0.002)

    controller.run_motor(motor='MOTOR_X',
                        steps=500, clockwise=True,
                        step_delay=0.002)
                        
    GPIO.cleanup()
