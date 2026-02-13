# Setup and Configuration Guide

## System Requirements

- **PC**: Windows/Linux/Mac with Python 3.8+
- **Camera**: USB camera connected to PC (default: index 2)
- **ESP8266**: Microcontroller with servo attached
- **MQTT Broker**: Mosquitto or compatible (local or cloud)

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

- `opencv-python`: Computer vision
- `numpy`: Numerical operations
- `onnxruntime`: ArcFace inference
- `scipy`: Mathematical functions
- `tqdm`: Progress bars
- `mediapipe`: Face landmark detection
- `paho-mqtt`: MQTT communication

### 2. Setup MQTT Broker

#### Option A: Local Mosquitto (Recommended for Development)

**Windows:**

```bash
# Download and install from:
# https://mosquitto.org/download/

# After installation, start Mosquitto:
mosquitto -v
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
sudo systemctl enable mosquitto
```

**macOS:**

```bash
brew install mosquitto
brew services start mosquitto
```

#### Option B: Public MQTT Broker (Testing)

For testing without local setup, you can use:

- `test.mosquitto.org` (insecure, port 1883)
- `broker.hivemq.com` (port 1883)

### 3. Configure System Files

#### A. MQTT Configuration

Edit the connection parameters in your main script:

```python
system = IntegratedFaceTrackingSystem(
    mqtt_broker="broker.hivemq.com",  # or your broker IP
    mqtt_port=1883,           # default MQTT port
    camera_index=2,           # adjust based on your camera
)
```

#### B. Hardware Configuration

Update servo controller settings if needed:

```python
servo = ServoController(
    mqtt_manager,
    center_angle=90,      # Center servo position
    min_angle=0,          # Minimum angle
    max_angle=180,        # Maximum angle
    movement_threshold=5, # Motion sensitivity
)
```

### 4. Setup ESP8266 with Arduino IDE

#### Install Arduino IDE Boards

1. Open Arduino IDE → Preferences
2. Add board URL: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
3. Tools → Board Manager → Search "ESP8266" → Install `esp8266 by ESP8266 Community`

#### Install Required Libraries

Tools → Manage Libraries:

- **PubSubClient** (by Nick O'Leary)
- **ArduinoJson** (optional, for JSON handling)

#### Flash ESP8266 Firmware

See `ESP8266_SERVO_FIRMWARE.ino` in this project.

Key steps:

1. Edit WiFi and MQTT credentials in the sketch
2. Select Board: "NodeMCU 1.0"
3. Select Port: COM port of your ESP8266
4. Upload sketch

### 5. Hardware Connections

#### ESP8266 Wiring

```
ESP8266        Component
------------------------------------
D1 (GPIO5)  → Servo Signal (Yellow)
5V          → Servo Power (Red)
GND         → Servo Ground (Black) + ESP8266 GND
```

#### Servo Calibration

1. Upload firmware with servo at 90° (center)
2. Adjust mechanical zero position if needed
3. Test range: Send MQTT commands 0-180

### 6. Database Setup

Before running recognition:

```bash
# Enroll faces:
python -m src.enroll

# This will create:
# - data/db/face_db.npz (embeddings database)
# - data/enroll/<name>/ (enrollment images)
```

## Verification Checklist

- [ ] Python dependencies installed
- [ ] MQTT broker running and accessible
- [ ] Camera detected and working
- [ ] ESP8266 connected to WiFi and MQTT
- [ ] Face database enrolled (at least 1 person)
- [ ] Servo motor responsive to MQTT commands

## Troubleshooting

### Camera Not Found

```python
# Check available cameras:
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} available")
        cap.release()

# Update camera_index in main script
```

### MQTT Connection Failed

```bash
# Verify broker is running:
mosquitto_sub -h localhost -t "test"

# In another terminal:
mosquitto_pub -h localhost -t "test" -m "hello"
```

### ESP8266 Not Connecting

- Check WiFi credentials in firmware
- Verify MQTT broker IP address
- Check serial console (115200 baud)
- Ensure ESP8266 and PC are on same network

### Servo Not Moving

- Test with direct GPIO control (90°)
- Check servo power supply (5V, sufficient current)
- Verify MQTT topic: `servo/command`
- Check message format (integer 0-180)

## Configuration Files

### mqtt_handler.py

```python
mqtt_manager = MQTTManager(
    broker="localhost",
    port=1883,
    client_id="face-recognition-system",
    username=None,      # Optional
    password=None,      # Optional
)
```

### servo_controller.py

```python
servo = ServoController(
    mqtt_manager,
    center_angle=90,
    min_angle=0,
    max_angle=180,
    movement_threshold=5,
    smoothing_enabled=True,
    min_publish_interval=100,  # milliseconds
)
```

### face_lock.py Settings

```python
system = FaceLockingSystem(db_path)
system.matcher.dist_thresh = 0.35    # Recognition threshold
system.lock_timeout = 3.0             # Seconds before lock release
```

## Network Configuration

### For Remote MQTT Broker

```python
# Use cloud MQTT service
mqtt_manager = MQTTManager(
    broker="broker.hivemq.com",
    port=1883,
    username="your_username",
    password="your_password",
)
```

### Security Recommendation

Enable MQTT authentication in production:

```bash
# Generate password file
mosquitto_passwd -c /etc/mosquitto/passwd username

# Configure in mosquitto.conf
allow_anonymous false
password_file /etc/mosquitto/passwd
```

## Next Steps

1. **Run the integrated system:**

   ```bash
   python -m src.integrated_system
   ```

2. **Monitor MQTT messages:**

   ```bash
   mosquitto_sub -h localhost -t "face/#"
   ```

3. **Control servo manually:**

   ```bash
   mosquitto_pub -h localhost -t "servo/command" -m "45"
   ```

4. **Check ESP8266 logs:**
   - Use Arduino IDE Serial Monitor (115200 baud)
   - Review Wi-Fi and MQTT connection status

## Documentation Files

- `USAGE_GUIDE.md` - Running and using the system
- `MQTT_PROTOCOL.md` - Message formats and topics
- `TROUBLESHOOTING.md` - Common issues and solutions
