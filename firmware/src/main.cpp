#include <Arduino.h>
#include <MOTOR.h>
#include <IMU.h>
#include <Servo.h>
#include <ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Char.h>
#include <std_msgs/Int32.h>
#include <RotaryEncoder.h>
#include <geometry_msgs/Twist.h>

//Declaration
#define INR 0
#define INL 2
#define FreqM 400
#define ServoPin 3
#define ButtonAPin 24
#define ButtonBPin 25
#define ButtonCPin 26
#define X_ENC1 5
#define X_ENC2 6

int motorSpeed = 0, servoSteering=90, angleObject,distanceObject;
int posA;
int ButtonAState,ButtonBState,ButtonCState;
int heading;
char buffer[50];

RotaryEncoder *encoderA = nullptr;
Motor motor(INR,INL,FreqM);
Servo servo;
IMU imu;

ros::NodeHandle nh;

void velCallback(  const geometry_msgs::Twist& vel){
  motorSpeed = vel.linear.x ;
  distanceObject = vel.angular.x ;
  angleObject = vel.angular.y ;
  servoSteering = vel.angular.z ;
}

std_msgs::String str_msg;
ros::Publisher chatter("chatter", &str_msg);
ros::Subscriber<geometry_msgs::Twist> sub("/cmd_vel" , velCallback);

void checkPositionA() {
  encoderA->tick();
}

void display(){
  Serial.print(ButtonAState);
  Serial.print(" : ");
  Serial.print(ButtonBState);
  Serial.print(" : ");
  Serial.println(servoSteering);

    // Serial.print("motorSpeed ");
  // Serial.print(motorSpeed);
  // Serial.print(" : ");
  // Serial.print("servoSteering ");
  // Serial.print(servoSteering);
  // Serial.println();
}

void sensor(){
  encoderA->tick();
  posA = encoderA->getPosition();

  ButtonAState = analogRead(ButtonAPin);
  ButtonBState = analogRead(ButtonBPin);
  ButtonCState = analogRead(ButtonCPin);

  heading = imu.getHeading();
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

  nh.initNode();
  nh.advertise(chatter);
  nh.subscribe(sub);

  pinMode(ButtonAState, INPUT_PULLUP);
  pinMode(ButtonBState, INPUT_PULLUP);
  pinMode(ButtonCState, INPUT_PULLUP);
  servo.write(90);
  
  encoderA = new RotaryEncoder(X_ENC1, X_ENC2, RotaryEncoder::LatchMode::TWO03);

  attachInterrupt(digitalPinToInterrupt(X_ENC1), checkPositionA, CHANGE);
  attachInterrupt(digitalPinToInterrupt(X_ENC2), checkPositionA, CHANGE);
}

void loop() {
  // communicaton();
  // display();
  sensor();
  // aktuator();

  servoSteering = constrain(servoSteering,60,120);
  servo.write(servoSteering); 

  sprintf(buffer, "motorSpeed: %d, servoSteering: %d", motorSpeed, servoSteering);
  str_msg.data = buffer;
  chatter.publish( &str_msg );
  //motor.setSpeed(20);//15 ws banter //Servo e sudut tengah e 90, mentok kiri 60 deraat, mentok kanan 120 derajat. jangan melebihi rntang 

  nh.spinOnce();
  delay(1);
}