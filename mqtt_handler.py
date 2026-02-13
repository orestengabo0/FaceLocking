# mqtt_handler.py
"""
MQTT Communication Manager for Face Recognition System.
Handles publishing face actions and servo control commands.
"""

import paho.mqtt.client as mqtt
import json
from datetime import datetime
from typing import Callable, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MQTTManager:
    """Manages MQTT connections and message publishing for the face recognition system."""
    
    def __init__(
        self,
        broker: str = "broker.hivemq.com",
        port: int = 1883,
        client_id: str = "face-recognition-system",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        Initialize MQTT Manager.
        
        Args:
            broker: MQTT broker address (default: broker.hivemq.com)
            port: MQTT broker port (default: 1883)
            client_id: Unique client identifier
            username: Optional username for broker authentication
            password: Optional password for broker authentication
        """
        self.broker = broker
        self.port = port
        self.client_id = client_id
        self.client = mqtt.Client(client_id=client_id)
        self.is_connected = False
        self.username = username
        self.password = password
        
        # Message callbacks
        self.on_servo_command: Optional[Callable[[int], None]] = None
        
        # Topic subscriptions
        self.subscriptions = [
            "servo/control",
            "system/command",
        ]
        
        self._setup_callbacks()
    
    def _setup_callbacks(self):
        """Setup MQTT client callbacks."""
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.client.on_publish = self._on_publish
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            self.is_connected = True
            logger.info(f"MQTT Connected to {self.broker}:{self.port}")
            # Subscribe to topics
            for topic in self.subscriptions:
                self.client.subscribe(topic)
                logger.debug(f"Subscribed to {topic}")
        else:
            logger.error(f"Failed to connect, return code {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the broker."""
        if rc != 0:
            logger.warning(f"Unexpected MQTT disconnection with code {rc}")
        self.is_connected = False
    
    def _on_message(self, client, userdata, msg):
        """Callback for when a message is received from the broker."""
        try:
            payload = msg.payload.decode()
            logger.debug(f"Received: {msg.topic} -> {payload}")
            
            # Handle servo control messages
            if msg.topic == "servo/control":
                try:
                    angle = int(payload)
                    if self.on_servo_command:
                        self.on_servo_command(angle)
                except ValueError:
                    logger.warning(f"Invalid servo angle: {payload}")
            
            # Handle system commands
            elif msg.topic == "system/command":
                logger.info(f"System command received: {payload}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for when a message is published."""
        logger.debug(f"Message {mid} published")
    
    def connect(self) -> bool:
        """Connect to MQTT broker."""
        try:
            if self.username and self.password:
                self.client.username_pw_set(self.username, self.password)
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False
            logger.info("MQTT Disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT: {e}")
    
    def publish_action(
        self,
        action_type: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publish face action to MQTT broker.
        
        Args:
            action_type: Type of action (e.g., 'BLINK', 'SMILE', 'MOVE')
            description: Human-readable description
            data: Optional additional data dictionary
            
        Returns:
            True if published successfully
        """
        if not self.is_connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "action": action_type,
            "description": description,
            "data": data or {},
        }
        
        try:
            result = self.client.publish(
                "face/actions",
                json.dumps(payload),
                qos=1
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish action: {result.rc}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error publishing action: {e}")
            return False
    
    def publish_servo_angle(self, angle: int) -> bool:
        """Publish servo angle command.
        
        Args:
            angle: Servo angle (0-180 degrees)
            
        Returns:
            True if published successfully
        """
        if not self.is_connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        # Clamp angle to valid range
        angle = max(0, min(180, angle))
        
        try:
            result = self.client.publish(
                "esp8266/servo/set",
                str(angle),
                qos=1
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish servo command: {result.rc}")
                return False
            logger.debug(f"Published servo angle: {angle}")
            return True
        except Exception as e:
            logger.error(f"Error publishing servo command: {e}")
            return False
    
    def publish_face_state(
        self,
        face_name: str,
        is_locked: bool,
        face_position: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> bool:
        """Publish current face state.
        
        Args:
            face_name: Name of the recognized face
            is_locked: Whether the face is currently locked
            face_position: Normalized horizontal position (0.0-1.0)
            confidence: Match confidence/similarity score
            
        Returns:
            True if published successfully
        """
        if not self.is_connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "face_name": face_name,
            "is_locked": is_locked,
            "face_position": face_position,
            "confidence": confidence,
        }
        
        try:
            result = self.client.publish(
                "face/state",
                json.dumps(payload),
                qos=1
            )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.warning(f"Failed to publish face state: {result.rc}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error publishing face state: {e}")
            return False