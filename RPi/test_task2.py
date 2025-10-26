#!/usr/bin/env python3
"""
Test script for task2.py

This script mocks the STM32 (robot car) connection to allow testing of:
- Android tablet communication
- Image snapping functionality
- Command processing flow

Usage:
    python test_task2.py
    
What is mocked:
- STM32 serial connection (simulates robot car responses)

What is REAL (not mocked):
- Android Bluetooth connection
- Image capture with Picamera2
- API calls for image recognition
"""

import json
import queue
import time
import threading
from multiprocessing import Process, Manager
from typing import Optional
import os
import sys

# Mock the STM32 connection before importing task2
class MockSTMLink:
    """Mock STM32 connection that simulates robot car responses"""
    
    def __init__(self):
        self.connected = False
        self.message_queue = queue.Queue()
        self.command_log = []
        self.auto_ack = True
        self.auto_ack_delay = 1.0  # seconds
        self.simulation_thread = None
        self.running = False
        
        print("[MOCK STM32] Initialized")
    
    def connect(self):
        """Simulate connection to STM32"""
        self.connected = True
        self.running = True
        print("[MOCK STM32] ✓ Connected (simulated)")
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.simulation_thread.start()
    
    def disconnect(self):
        """Simulate disconnection from STM32"""
        self.running = False
        self.connected = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        print("[MOCK STM32] ✓ Disconnected (simulated)")
    
    def send(self, message: str) -> None:
        """Receive a command from RPi (simulating STM32 receiving)"""
        if not self.connected:
            print("[MOCK STM32] ✗ Cannot receive command: not connected")
            return
        
        self.command_log.append({
            'timestamp': time.time(),
            'command': message
        })
        
        print(f"[MOCK STM32] ← Received command from RPi: {message}")
        
        # Auto-send ACK after delay
        if self.auto_ack and message not in ["FIN"]:
            threading.Timer(self.auto_ack_delay, self._send_ack).start()
    
    def recv(self) -> Optional[str]:
        """Send a message to RPi (simulating STM32 sending)"""
        if not self.connected:
            print("[MOCK STM32] ✗ Cannot send message: not connected")
            return None
        
        try:
            message = self.message_queue.get(timeout=0.1)
            print(f"[MOCK STM32] → Sending to RPi: {message}")
            return message
        except queue.Empty:
            # Block until message available
            message = self.message_queue.get()
            print(f"[MOCK STM32] → Sending to RPi: {message}")
            return message
    
    def _send_ack(self):
        """Internal method to send ACK"""
        self.message_queue.put("ACK")
    
    def send_snap(self):
        """Manually trigger a SNAP command"""
        print("[MOCK STM32] 📸 Triggering SNAP command...")
        self.message_queue.put("SNAP")
    
    def _simulation_loop(self):
        """Background thread that can simulate robot actions"""
        print("[MOCK STM32] Simulation thread started")
        
        # Example: Automatically send SNAP commands during testing
        # Uncomment the following to auto-trigger SNAP after start
        """
        time.sleep(10)  # Wait 10 seconds after start
        while self.running:
            print("[MOCK STM32] Auto-triggering SNAP for testing...")
            self.send_snap()
            time.sleep(15)  # SNAP every 15 seconds
        """
        
        while self.running:
            time.sleep(1)
    
    def print_command_log(self):
        """Print all commands received"""
        print("\n" + "="*60)
        print("MOCK STM32 - Command Log")
        print("="*60)
        for i, cmd in enumerate(self.command_log, 1):
            timestamp = time.strftime('%H:%M:%S', time.localtime(cmd['timestamp']))
            print(f"{i}. [{timestamp}] {cmd['command']}")
        print("="*60 + "\n")


# Patch the task2 module to use our mock
import sys
from unittest.mock import MagicMock

# Create mock STM link instance
mock_stm_link = MockSTMLink()

# Now import task2 and patch it
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import original modules but we'll replace STMLink
from task2 import RaspberryPi, PiAction
from communication.android import AndroidLink, AndroidMessage

# Monkey-patch the RaspberryPi class to use our mock STM
original_init = RaspberryPi.__init__

def patched_init(self):
    """Patched __init__ that uses MockSTMLink"""
    original_init(self)
    # Replace the real STM link with our mock
    self.stm_link = mock_stm_link
    print("[TEST] ✓ Patched RaspberryPi to use MockSTMLink")

RaspberryPi.__init__ = patched_init


def print_test_info():
    """Print test information"""
    print("\n" + "="*60)
    print("TASK2.PY TEST SCRIPT")
    print("="*60)
    print("This script tests task2.py with:")
    print("  ✓ REAL Android Bluetooth connection")
    print("  ✓ REAL Image capture (Picamera2)")
    print("  ✓ REAL API calls for image recognition")
    print("  ⚠ MOCKED STM32 (robot car) connection")
    print()
    print("Commands you can send from Android:")
    print('  • Start robot: {"cat": "control", "value": "WN01"}')
    print()
    print("Test Controls:")
    print("  • Press Ctrl+C to stop")
    print("  • Commands sent to STM32 will be logged")
    print("="*60 + "\n")


def interactive_menu():
    """Interactive menu for testing while the system runs"""
    print("\n[TEST MENU] Available commands:")
    print("  1 - Trigger SNAP (simulate image capture request)")
    print("  2 - Show STM32 command log")
    print("  3 - Toggle auto-ACK")
    print("  4 - Set ACK delay")
    print("  q - Quit")
    
    while True:
        try:
            choice = input("\n[TEST] Enter command: ").strip().lower()
            
            if choice == '1':
                mock_stm_link.send_snap()
                print("[TEST] ✓ SNAP command queued")
            
            elif choice == '2':
                mock_stm_link.print_command_log()
            
            elif choice == '3':
                mock_stm_link.auto_ack = not mock_stm_link.auto_ack
                print(f"[TEST] Auto-ACK is now: {'ON' if mock_stm_link.auto_ack else 'OFF'}")
            
            elif choice == '4':
                try:
                    delay = float(input("[TEST] Enter ACK delay in seconds: "))
                    mock_stm_link.auto_ack_delay = delay
                    print(f"[TEST] ✓ ACK delay set to {delay}s")
                except ValueError:
                    print("[TEST] ✗ Invalid delay value")
            
            elif choice == 'q':
                print("[TEST] Exiting...")
                break
            
            else:
                print("[TEST] Unknown command")
                
        except EOFError:
            break
        except Exception as e:
            print(f"[TEST] Error: {e}")


if __name__ == "__main__":
    print_test_info()
    
    # Start the RaspberryPi system
    rpi = RaspberryPi()
    
    # Run the system in a separate thread so we can have interactive control
    system_thread = threading.Thread(target=rpi.start, daemon=True)
    system_thread.start()
    
    # Give system time to start
    time.sleep(2)
    
    # Start interactive menu
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    finally:
        print("\n[TEST] Shutting down...")
        mock_stm_link.print_command_log()
        rpi.stop()
        print("[TEST] ✓ Test complete")

