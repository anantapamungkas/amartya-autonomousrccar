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

#endif
