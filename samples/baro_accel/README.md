Barometer-Accelerometer: BMP388
===============================

Description
-----------
This sample compares and fuses the outputs of two barometric sensors,
BMP388 and LPS22HH. This does not offer anything on the problem of altitude
sensing, but serves as a hello world program to the Extended Kalman Filters.

The input is based on real measurements taken on a specific altitude without
the sensors move in any direction. The frequency is 1 second per measurement
for both of them.

Prerequisites
-------------
The TinyEKF library has been used, which is a git submodule in this repository.
If you haven't initialized the git submodules please do it like so:
```
git submodule init
git submodule update
```

Install python requirements issuing the below command:
```
pip3 install --user -r requirements.txt
```
