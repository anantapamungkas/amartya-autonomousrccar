#include <Arduino.h>
#include <MOTOR.h>
#include <IMU.h>
#include <Servo.h>
#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Char.h>
#include <std_msgs/Int32.h>
#include <RotaryEncoder.h>

//Declaration
#define INR 0
#define INL 2
#define FreqM 100000
#define ServoPin 3
#define ButtonAPin 24
#define ButtonBPin 25
#define ButtonCPin 26
#define X_ENC1 3
#define X_ENC2 4

int motorSpeed = 0, servoSteering;
int posA;
int ButtonAState,ButtonBState,ButtonCState;

RotaryEncoder *encoderA = nullptr;
RotaryEncoder *encoderB = nullptr;
Motor motor(INR,INL,FreqM);
Servo servo;
IMU imu;

void msgSpeed(const std_msgs::Int32 &msg) {
  motorSpeed = msg.data;
}
void msgSteering(const std_msgs::Int32 &msg) {
  servoSteering = msg.data;
}
void checkPositionA() {
  encoderA->tick();
}


void display(){
  Serial.print(ButtonAState);
  Serial.print(" : ");
  Serial.print(ButtonBState);
  Serial.print(" : ");
  Serial.println(servoSteering);
}

void sensor(){
  encoderA->tick();
  posA = encoderA->getPosition();

  ButtonAState = analogRead(ButtonAPin);
  ButtonBState = analogRead(ButtonBPin);
  ButtonCState = analogRead(ButtonCPin);
}

void aktuator(){
  motor.setSpeed(motorSpeed);
  servo.write(servoSteering);  
}

void communicaton(){
  if (Serial.available() > 0) {
    motorSpeed = Serial.readStringUntil(':').toInt();
    servoSteering = Serial.readStringUntil(':').toInt();
  }
}

void setup() {
  Serial.begin(115200);
  imu.begin();
  servo.attach(ServoPin);
  pinMode(ButtonAState, INPUT);
  pinMode(ButtonBState, INPUT);
  pinMode(ButtonCState, INPUT);
  servo.write(180);
  

  encoderA = new RotaryEncoder(X_ENC1, X_ENC2, RotaryEncoder::LatchMode::TWO03);

  attachInterrupt(digitalPinToInterrupt(X_ENC1), checkPositionA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(X_ENC2), checkPositionA, CHANGE);
}

void loop() {
  // communicaton();
  display();
  sensor();
  // aktuator();

  if (ButtonAState == 1023){
    servoSteering += 5;
    delay(50);
  }

  if (ButtonBState == 1023){
    servoSteering -= 5;
    delay(50);
  }
  servo.write(servoSteering); 
}