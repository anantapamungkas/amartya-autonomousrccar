#include "IMU.h"

IMU::IMU(int i2cAddress) {
    address = i2cAddress;
    initialAngle = 0;
}

void IMU::begin() {
    Wire.begin();
}

int IMU::readRawAngle() {
    Wire.beginTransmission(address);
    Wire.write(0x02);
    Wire.endTransmission();

    Wire.requestFrom(address, 2);

    if (Wire.available() >= 2) {
        uint8_t high_byte = Wire.read();
        uint8_t low_byte = Wire.read();
        return ((high_byte << 8) | low_byte) / 10.0;
    }
    return -1;
}

int IMU::getHeading() {
    int rawAngle = readRawAngle();
    if (rawAngle == -1) return -1;

    int heading = (rawAngle - initialAngle + 360) % 360;
    return (heading == 0) ? 360 : heading;
}

int IMU::getHeadingCCW() {
    int heading = getHeading();
    return (heading == 360) ? 0 : (360 - heading);
}

void IMU::resetHeading() {
    int rawAngle = readRawAngle();
    if (rawAngle != -1) {
        initialAngle = rawAngle + 180;
    }
}

void IMU::printHeading() {
    Serial.print("CW: ");
    Serial.print(getHeading());
    Serial.print("° | CCW: ");
    Serial.print(getHeadingCCW());
    Serial.println("°");
}

// Shift heading by a given angle offset
int IMU::shiftHeading(int angleOffset) {
    int outputAngle = this->getHeading() + angleOffset;

    if (outputAngle >= 360) outputAngle -= 360;
    else if (outputAngle < 0) outputAngle += 360;

    return outputAngle;
}