#include <Arduino.h>
#include <MOTOR.h>
#include <IMU.h>
#include <Servo.h>
#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Char.h>
#include <std_msgs/Int32.h>

//Declaration
#define INR 0
#define INL 2
#define FreqM 100000
#define ServoPin 12
#define ButtonAPin 12
#define ButtonAPin 12
#define ButtonAPin 12

int motorSpeed, servoSteering;

Motor motor(INR,INL,FreqM);
Servo servo;
IMU IMU;

void msgSpeed(const std_msgs::Int32 &msg) {
  motorSpeed = msg.data;
}
void msgSteering(const std_msgs::Int32 &msg) {
  servoSteering = msg.data;
}

void setup() {
  Serial.begin(115200);
  IMU.begin();
  servo.write(90);
  servo.attach(ServoPin);
}

void loop() {
  
}

void display(){
  Serial.print(motorSpeed);
  Serial.print(" : ");
  Serial.print(servoSteering);
}

void aktuator(){
  motor.setSpeed(motorSpeed);
  servo.write(servoSteering);  
}

void Communicaton(){

}