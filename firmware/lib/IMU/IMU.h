#ifndef IMU_H
#define IMU_H

#include <Arduino.h>
#include <Wire.h>

class IMU {
private:
    int address;
    int initialAngle;
    int readRawAngle();

public:
    IMU(int i2cAddress = 0x60);

    void begin();
    int getHeading();       // Get heading CW
    int getHeadingCCW();    // Get heading CCW
    void resetHeading();    // Reset heading to current
    void printHeading();    // Print heading data
    int shiftHeading(int angleOffset); // 🔹 New name: shiftHeading
};

#endif