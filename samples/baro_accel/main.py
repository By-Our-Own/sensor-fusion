#!/usr/bin/env python3

'''
baros_accel.py - BMP388 with accelerometer compensation using TinyEKF.

We model a double state variable, vertical distance in centimeters and vertical
velocity.
This is obtained by using readings from one barometer pressure in Pascals,
compensated with an accelerometer.
'''

import numpy as np
from tinyekf import EKF


# Convert ASL cm to Pascals. See:
# http://www.engineeringtoolbox.com/air-altitude-pressure-d_462.html
def asl2baro(asl):
    return 101325 * pow((1 - 2.25577e-7 * asl), 5.25588)


# Convert Pascals to cm ASL
def baro2asl(pa):
    return (1.0 - pow(pa / 101325.0, 0.190295)) * 4433000.0


class ASL_EKF(EKF):
    '''
    An abstract class for fusing two different baros.
    '''

    def __init__(self):
        # One state (ASL), two measurements from two different baros, with
        # larger-than-usual measurement covariance noise.
        EKF.__init__(self, 2, 1, pval=0.1, qval=1e-3, rval=2500)
        self.dt = 1

    def f(self, x, u):
        # State-transition function is identity
        F = np.array([[1, self.dt], [0, 1]])
        B = np.array([self.dt**2 / 2, self.dt])
        return np.copy(np.dot(F, x) + B * u), F

    def h(self, x):
        # State value is ASL
        distance = x[0]
        velocity = x[1]

        # Convert ASL cm to Pascals
        b = asl2baro(distance)

        h = np.array([b])

        # First derivative of nonlinear baro-measurement function
        # Used http://www.wolframalpha.com
        dpdx = -0.120131 * pow((1 - 2.2577e-7 * distance), 4.25588)
        dv = 0

        H = np.array([[dpdx, dv]])

        return h, H


class App():

    def __init__(self, press_dataset_path, accel_dataset_path):
        self.ekf = ASL_EKF()

        # Read pressure values for the two sensors
        self.y_bmp = np.loadtxt(press_dataset_path, delimiter='\n',
                                unpack=True)
        self.y_lis = np.loadtxt(accel_dataset_path, delimiter='\n',
                                unpack=True)
        self.num_samples = len(self.y_bmp)

        # Generate the timestamps
        self.x = range(0, self.num_samples)

        self.y_bmp_m = [baro2asl(i * 1000) / 100 for i in self.y_bmp]
        self.y_lis = [(i - 9.8) for i in self.y_lis]
        self.fused_d = []
        self.fused_v = []

        self.count = -1

    def getSensors(self):
        self.count += 1

        return self.y_bmp[self.count] * 1000

    def run(self):
        for i in range(self.num_samples):
            self.baro = self.getSensors()

            # Run the EKF on the current baros measurements, getting
            # back an updated state estimate made by fusing them.
            # Fused state comes back as an array, so grab first element and
            # append it to the fused values list.

            x = self.ekf.step((self.baro, ), self.y_lis[i] / 100)
            self.fused_d.append(x[0] / 100)
            self.fused_v.append(x[1] / 100)

    def plot(self):
        import matplotlib.pyplot as plt
        # plt.plot(self.x, self.y_bmp, label='BMP388 pressure in kPa')
        # plt.xlabel('time in seconds')
        # plt.ylabel('pressure in kPa')
        # plt.title('Compare BMP388 with LPS22HH')
        # plt.legend()
        # plt.show()

        plt.plot(self.x, self.y_bmp_m, label='BMP388 pressure in m')
        plt.plot(self.x, self.fused_d, label='Fused in m')
        plt.plot(self.x, self.fused_v, label='Fused in m/s')
        plt.plot(self.x, self.y_lis, label='Acceleration in m/s^2')
        plt.xlabel('time in seconds')
        plt.ylabel('altitude in meters')
        plt.title('Compare BMP388 with LPS22HH')
        plt.legend()
        plt.show()


def main(dataset):
    app = None
    if 'pressure_accel_static_point_50msec' in dataset:
        app = App(dataset + '/press_bmp388.txt',
                  dataset + '/accel_lis2dw12_vertical.txt')
    elif 'pressure_accel_elev_straight_50msec' in dataset:
        app = App(dataset + '/press_bmp388.txt',
                  dataset + '/accel_lis2dw12_vertical.txt')
    elif 'pressure_accel_elev_withbreaks_50msec' in dataset:
        app = App(dataset + '/press_bmp388.txt',
                  dataset + '/accel_lis2dw12_vertical.txt')

    app.run()
    app.plot()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='choose the dataset of experiment',
                        choices=['pressure_accel_static_point_50msec',
                                 'pressure_accel_elev_straight_50msec',
                                 'pressure_accel_elev_withbreaks_50msec'],
                        type=str)
    args = parser.parse_args()

    main('../../raw_data/' + args.dataset)
