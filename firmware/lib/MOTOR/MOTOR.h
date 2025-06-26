#ifndef MOTOR_H
#define MOTOR_H

class Motor {
  public:
    Motor(int pin1, int pin2, int freq);
    void setSpeed(int speed);
    void stop();

  private:
    int pwmPin1;
    int pwmPin2;
    int frequency;
};

class PID {
  public:
    void init(float p, float i, float d);
    float calculate(float desiredValue, float currentValue);
    void calculatePID();
  
  private:
    float Kp;
    float Ki;
    float Kd;
  
    float integralMax = 0.0;
    float integralTerm = 0.0;
    float previousError = 0.0;
    float controlOutput = 0.0;
  };

#endif