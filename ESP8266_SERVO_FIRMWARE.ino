#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>
#include <algorithm>

// =====================
// WIFI CREDENTIALS
// =====================
const char* ssid     = "RCA-OUTDOOR";
const char* password = "RCA@2025";

// =====================
// MQTT SETTINGS
// =====================
const char* mqtt_server   = "broker.hivemq.com";
const int   mqtt_port     = 1883;
const char* mqtt_user     = "";
const char* mqtt_password = "";

const char* status_topic  = "esp8266/status";
const char* command_topic = "servo/command";
const char* state_topic   = "esp8266/servo/state";

// =====================
// SERVO SETTINGS
// =====================
const int SERVO_PIN = 5;  // D1 on NodeMCU
Servo myServo;

volatile int currentAngle = 90;
volatile int targetAngle  = 90;

float smoothedAngle = 90.0;
const float smoothingAlpha = 0.2;   // EMA smoothing factor
const int maxStep = 5;              // max degrees per update
const unsigned long stepInterval = 10; // ms between servo updates

// =====================
// OBJECTS
// =====================
WiFiClient espClient;
PubSubClient client(espClient);

// Timing
unsigned long lastMoveTime = 0;

// =====================================================
// WIFI CONNECT
// =====================================================
void setup_wifi() {
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  int counter = 0;
  while (WiFi.status() != WL_CONNECTED) {
      Serial.print(".");
      delay(500);
      counter++;
      if(counter > 30){
          Serial.println("\nFailed to connect!");
          break;
      }
  }

  if(WiFi.status() == WL_CONNECTED){
      Serial.println("\nWiFi Connected!");
      Serial.print("IP Address: ");
      Serial.println(WiFi.localIP());
  } else {
      Serial.println("Check WiFi credentials or 2.4GHz band.");
  }
}

// =====================================================
// MQTT CALLBACK
// =====================================================
void callback(char* topic, byte* payload, unsigned int length) {
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  Serial.print("Message arrived: ");
  Serial.println(message);

  int newAngle = message.toInt();
  newAngle = constrain(newAngle, 0, 180);
  targetAngle = newAngle;
}

// =====================================================
// MQTT RECONNECT
// =====================================================
void mqtt_reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");

    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);

    String will_message = "{\"status\":\"offline\"}";

    if (client.connect(clientId.c_str(),
                       mqtt_user,
                       mqtt_password,
                       status_topic,
                       1,
                       true,
                       will_message.c_str())) {
      Serial.println("connected");
      client.subscribe(command_topic);
      publish_status("online");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

// =====================================================
// PUBLISH STATUS
// =====================================================
void publish_status(String status) {
  String message = String("{\"status\":\"") + status +
                   String("\",\"ip\":\"") + WiFi.localIP().toString() +
                   String("\",\"servo_angle\":") + String(currentAngle) +
                   String("}");
  client.publish(status_topic, message.c_str(), true);
}

// =====================================================
// PUBLISH SERVO STATE
// =====================================================
void publish_servo_state() {
  String message = String("{\"current\":") + String(currentAngle) +
                   String(",\"target\":") + String(targetAngle) +
                   String(",\"timestamp\":") + String(millis()) +
                   String("}");
  client.publish(state_topic, message.c_str(), true);
}

// =====================================================
// SERVO MOVEMENT - Smooth, proportional, non-blocking
// =====================================================
void move_servo_smooth() {
  unsigned long now = millis();
  if(now - lastMoveTime < stepInterval) return;
  lastMoveTime = now;

  // Exponential smoothing
  smoothedAngle = smoothingAlpha * targetAngle + (1 - smoothingAlpha) * smoothedAngle;

  // Proportional step
  int diff = round(smoothedAngle - currentAngle);
  diff = constrain(diff, -maxStep, maxStep);

  if(diff != 0){
    currentAngle += diff;
    currentAngle = constrain(currentAngle, 0, 180);
    myServo.write(currentAngle);
    publish_servo_state();

    Serial.print("Servo moving: ");
    Serial.print(currentAngle);
    Serial.print(" -> ");
    Serial.println(targetAngle);
  }
}

// =====================================================
// SETUP
// =====================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  myServo.attach(SERVO_PIN);
  myServo.write(currentAngle);

  setup_wifi();

  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

// =====================================================
// LOOP
// =====================================================
void loop() {
  if (!client.connected()) mqtt_reconnect();

  client.loop();
  move_servo_smooth();
}
