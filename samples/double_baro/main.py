#!/usr/bin/env python3

'''
baros_fuser.py - BMP388 / LPS22HH fusion example using TinyEKF.

We model a single state variable, altitude above sea level (ASL) in centimeters.
This is obtained by fusing two barometer pressure readings in Pascals.

This is not much of a use. Just an warm-up on using the library
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
        EKF.__init__(self, 1, 2, rval=.1)

    def f(self, x):
        # State-transition function is identity
        return np.copy(x), np.eye(1)

    def h(self, x):
        # State value is ASL
        asl = x[0]

        # Convert ASL cm to Pascals
        b1 = b2 = asl2baro(asl)

        h = np.array([b1, b2])

        # First derivative of nonlinear baro-measurement function
        # Used http://www.wolframalpha.com
        dp1dx = dp2dx = -0.120131 * pow((1 - 2.2577e-7 * x[0]), 4.25588)

        H = np.array([[dp1dx], [dp2dx]])

        return h, H


class App():

    def __init__(self):
        self.ekf = ASL_EKF()

        # Read pressure values for the two sensors
        self.y_bmp = np.loadtxt('press_bmp.txt', delimiter='\n', unpack=True)
        self.y_lps = np.loadtxt('press_lps.txt', delimiter='\n', unpack=True)
        self.num_samples = len(self.y_bmp)

        # Generate the timestamps
        self.x = range(0, self.num_samples)

        self.y_bmp_m = [baro2asl(i * 1000) / 100 for i in self.y_bmp]
        self.y_lps_m = [baro2asl(i * 1000) / 100 for i in self.y_lps]
        self.fused = []

        self.count = -1

    def getSensors(self):
        self.count += 1

        return self.y_bmp[self.count] * 1000, self.y_lps[self.count] * 1000

    def run(self):
        for _ in range(self.num_samples):
            self.baro, self.sonar = self.getSensors()

            # Run the EKF on the current baros measurements, getting
            # back an updated state estimate made by fusing them.
            # Fused state comes back as an array, so grab first element and
            # append it to the fused values list.
            self.fused.append(
                    self.ekf.step((self.baro, self.sonar))[0] / 100)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.x, self.y_bmp, label='BMP388 pressure in kPa')
        plt.plot(self.x, self.y_lps, label='LPS22HH pressure in kPa')
        plt.xlabel('time in seconds')
        plt.ylabel('pressure in kPa')
        plt.title('Compare BMP388 with LPS22HH')
        plt.legend()
        plt.show()

        plt.plot(self.x, self.y_bmp_m, label='BMP388 pressure in m')
        plt.plot(self.x, self.y_lps_m, label='LPS22HH pressure in m')
        plt.plot(self.x, self.fused, label='Fused in m')
        plt.xlabel('time in seconds')
        plt.ylabel('altitude in meters')
        plt.title('Compare BMP388 with LPS22HH')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    app = App()
    app.run()
    app.plot()
