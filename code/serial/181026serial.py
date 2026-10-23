import serial
import time

arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)

def write_read(x):
    arduino.write((x + '\n').encode('utf-8'))
    data = arduino.readline()
    return data.decode('utf-8').strip()

while True:
    num = input("Enter a number: ")
    value = write_read(num)
    print("Arduino says:", value)
