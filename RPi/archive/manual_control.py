#!/usr/bin/env python3
"""
Simple manual control program for robot movement demonstration.
Allows Android tablet to send movement commands to RPi which forwards them to STM32.
"""

import json
import queue
import time
from multiprocessing import Process, Manager
from typing import Optional
from communication.android import AndroidLink, AndroidMessage
from communication.stm32 import STMLink
from logger import prepare_logger


class ManualControl:
    """
    Simple manual control class for robot movement demonstration.
    """

    def __init__(self):
        """
        Initialize the manual control system.
        """
        self.logger = prepare_logger()
        self.android_link = AndroidLink()
        self.stm_link = STMLink()

        self.manager = Manager()
        self.android_dropped = self.manager.Event()
        self.movement_lock = self.manager.Lock()
        
        # Queues for communication
        self.android_queue = self.manager.Queue()  # Messages to send to Android
        self.command_queue = self.manager.Queue()  # Movement commands to STM32
        
        # Process references
        self.proc_recv_android = None
        self.proc_recv_stm32 = None
        self.proc_android_sender = None
        self.proc_command_executor = None

    def start(self):
        """Start the manual control system"""
        try:
            # Initialize connections
            self.android_link.connect()
            self.android_queue.put(AndroidMessage('info', 'Manual control connected!'))
            self.stm_link.connect()
            
            # Define child processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_recv_stm32 = Process(target=self.recv_stm)
            self.proc_android_sender = Process(target=self.android_sender)
            self.proc_command_executor = Process(target=self.command_executor)

            # Start child processes
            self.proc_recv_android.start()
            self.proc_recv_stm32.start()
            self.proc_android_sender.start()
            self.proc_command_executor.start()

            self.logger.info("Manual control processes started")
            self.android_queue.put(AndroidMessage('info', 'Robot ready for manual control!'))
            self.android_queue.put(AndroidMessage('mode', 'manual'))
            
            # Handle reconnection
            self.reconnect_android()

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop all processes and disconnect gracefully"""
        self.android_link.disconnect()
        self.stm_link.disconnect()
        self.logger.info("Manual control program exited!")

    def reconnect_android(self):
        """Handle Android reconnection"""
        self.logger.info("Reconnection handler is watching...")

        while True:
            # Wait for android connection to drop
            self.android_dropped.wait()
            self.logger.error("Android link is down!")

            # Kill android processes
            self.proc_android_sender.kill()
            self.proc_recv_android.kill()

            # Wait for processes to finish
            self.proc_android_sender.join()
            self.proc_recv_android.join()

            # Clean up and reconnect
            self.android_link.disconnect()
            self.android_link.connect()

            # Recreate android processes
            self.proc_recv_android = Process(target=self.recv_android)
            self.proc_android_sender = Process(target=self.android_sender)

            # Start processes
            self.proc_recv_android.start()
            self.proc_android_sender.start()

            self.logger.info("Android processes restarted")
            self.android_queue.put(AndroidMessage("info", "Reconnected!"))
            self.android_queue.put(AndroidMessage('mode', 'manual'))

            self.android_dropped.clear()

    def recv_android(self) -> None:
        """
        Process messages received from Android tablet.
        """
        while True:
            msg_str: Optional[str] = None
            try:
                msg_str = self.android_link.recv()
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android connection dropped")

            if msg_str is None:
                continue

            try:
                message: dict = json.loads(msg_str)
                self.logger.info(f"Received from Android: {message}")
                
                # Handle manual movement commands
                if message['cat'] == "manual":
                    command_data = message['value']
                    self.handle_manual_command(command_data)
                    
            except json.JSONDecodeError:
                self.logger.error(f"Invalid JSON received: {msg_str}")
            except Exception as e:
                self.logger.error(f"Error processing Android message: {e}")

    def convert_android_to_stm_command(self, android_command, detection="00", units="05"):
        """
        Quick method to convert Android commands to STM32 format (XXYYYY).
        
        Args:
            android_command (str): Command from Android (e.g., "FW--", "STOP", "forward")
            detection (str): Detection code (default: "00")
            units (str): Units for movement commands (default: "05")
            
        Returns:
            str: STM32 command in XXYYYY format
        """
        # Direct STM32 command mapping (uppercase)
        direct_map = {
            "FW--": "FW--",
            "BW--": "BW--", 
            "STOP": "STOP",
            "FL90": "FL90",
            "FR90": "FR90"
        }
        
        # Android-friendly command mapping (lowercase)
        friendly_map = {
            "forward": f"FW{units}",
            "backward": f"BW{units}",
            "left": "FL90",
            "right": "FR90",
            "stop": "STOP",
            "forward_indef": "FW--",
            "backward_indef": "BW--"
        }
        
        # Check direct STM32 commands first
        if android_command in direct_map:
            return f"{detection}{direct_map[android_command]}"
        
        # Check friendly Android commands
        if android_command in friendly_map:
            return f"{detection}{friendly_map[android_command]}"
        
        # Try to parse movement commands with units (e.g., "FW05", "BW10")
        if android_command.startswith(("FW", "BW")) and len(android_command) >= 4:
            return f"{detection}{android_command}"
        
        return None

    def handle_manual_command(self, command_data):
        """
        Handle manual movement commands from Android.
        
        Expected command format:
        - Simple string: "forward", "backward", "left", "right", "stop", "FW--", "STOP"
        - Dictionary with detection and command: {"detection": "00", "command": "forward", "units": "05"}
        """
        # Default detection code for manual control
        default_detection = "00"
        default_units = "05"
        
        # Parse command data
        if isinstance(command_data, str):
            # Simple string command
            command = command_data
            detection = default_detection
            units = default_units
        elif isinstance(command_data, dict):
            # Dictionary command with detection code
            command = command_data.get("command", "")
            detection = command_data.get("detection", default_detection)
            units = command_data.get("units", default_units)
        else:
            self.android_queue.put(AndroidMessage('error', f'Invalid command format: {command_data}'))
            return
        
        # Convert Android command to STM32 format
        stm_command = self.convert_android_to_stm_command(command, detection, units)
        
        if stm_command:
            self.command_queue.put(stm_command)
            self.android_queue.put(AndroidMessage('info', f'Executing: {command} -> {stm_command}'))
            self.logger.info(f"Queued STM32 command: {stm_command}")
        else:
            self.android_queue.put(AndroidMessage('error', f'Unknown command: {command}'))
            self.logger.warning(f"Unknown command received: {command}")

    def recv_stm(self) -> None:
        """
        Receive acknowledgement messages from STM32.
        """
        while True:
            try:
                message: str = self.stm_link.recv()
                
                if message.startswith("ACK"):
                    self.movement_lock.release()
                    self.logger.debug("ACK from STM32 received, movement lock released.")
                    self.android_queue.put(AndroidMessage('info', 'Command completed'))
                else:
                    self.logger.warning(f"Ignored unknown message from STM: {message}")
                    
            except Exception as e:
                self.logger.error(f"Error receiving from STM32: {e}")

    def android_sender(self) -> None:
        """
        Send messages to Android tablet.
        """
        while True:
            try:
                message: AndroidMessage = self.android_queue.get(timeout=0.5)
                self.android_link.send(message)
            except queue.Empty:
                continue
            except OSError:
                self.android_dropped.set()
                self.logger.debug("Event set: Android dropped")

    def command_executor(self) -> None:
        """
        Execute commands from the command queue by sending them to STM32.
        """
        while True:
            try:
                # Get next command
                command: str = self.command_queue.get()
                
                # Acquire movement lock
                self.movement_lock.acquire()
                
                # Send command to STM32
                self.stm_link.send(command)
                self.logger.info(f"Sent to STM32: {command}")
                
            except Exception as e:
                self.logger.error(f"Error in command executor: {e}")


if __name__ == "__main__":
    print("Starting Manual Control Program...")
    print("Available commands from Android:")
    print("- forward: Move forward")
    print("- backward: Move backward") 
    print("- left: Turn left")
    print("- right: Turn right")
    print("- stop: Stop movement")
    print("\nSend commands as JSON: {\"cat\": \"manual\", \"value\": \"forward\"}")
    print("Press Ctrl+C to exit\n")
    
    manual_control = ManualControl()
    manual_control.start()
