#include <Arduino.h>
#include "Motor.h"

Motor motor(0, 2, 400);

void setup() {
    Serial.begin(9600);
}

void loop() {
    Serial.println("Test Motor");
    motor.setSpeed(50);
    delay(5000);
    motor.setSpeed(0);
    delay(5000);
}
