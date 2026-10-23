int x;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(10);
}

void loop() {
  if (Serial.available() > 0) {
    x = Serial.readStringUntil('\n').toInt();

    if (x == 3) {
      Serial.println("The motor is moving");
    } 
    else {
      Serial.println("Unknown command");
    }
  }
}
