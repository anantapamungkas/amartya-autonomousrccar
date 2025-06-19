#include <Arduino.h>
#include "Motor.h"
#include "IMU.h"

#define PIN_BTN1 0
#define PIN_BTN2 0
#define PIN_BTN3 0

Motor motor(0, 2, 400);

void setup() {
  Serial.begin(9600);

  pinMode(PIN_BTN1, INPUT);
  pinMode(PIN_BTN2, INPUT);
  pinMode(PIN_BTN3, INPUT);
}

void loop() {
  
}

int readButton(int pin) {
  int output = digitalRead(pin);
  return output;
}