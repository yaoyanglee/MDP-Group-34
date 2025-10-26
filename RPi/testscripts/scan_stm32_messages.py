#!/usr/bin/env python3
"""
Script to scan for messages from STM32 and send test commands.
This is a lightweight version for testing STM32 communication.
"""

import time
import sys
import os
import signal
import threading

# Add the parent directory to the path to import the stm32 module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from communication.stm32 import STMLink


class STM32MessageScanner:
    """STM32 message scanner and command sender."""
    
    def __init__(self):
        self.stm_link = STMLink()
        self.running = True
        self.send_mode = False
        self.sent_count = 0
        self.received_count = 0
        self.ack_count = 0
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Predefined test commands (new XXYYYY format)
        self.test_commands = {
            '1': '00FW05',    # Move forward 5 units
            '2': '00BW05',    # Move backward 5 units
            '3': '00FL90',    # Turn left 90 degrees
            '4': '00FR90',    # Turn right 90 degrees
            '5': '00FW10',    # Move forward 10 units
            '6': '00BW10',    # Move backward 10 units
            '7': '00FW--',    # Move forward indefinitely
            '8': '00BW--',    # Move backward indefinitely
            '9': '00FW15',    # Move forward 15 units
            '0': '00BW15',    # Move backward 15 units
            's': '00STOP',    # Stop all servos
            't': '00TEST',    # Custom test message
        }
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        print(f"\n[{time.strftime('%H:%M:%S')}] Received signal {signum}. Stopping scanner...")
        self.running = False
    
    def validate_command(self, command: str) -> bool:
        """
        Validate STM32 command format.
        
        Args:
            command: Command string to validate
            
        Returns:
            bool: True if command is valid, False otherwise
        """
        if not command:
            return False
        
        # Check for valid command patterns (XXYYYY format)
        valid_patterns = [
            r'^\d{2}FW\d{3}$',    # XXFW followed by 2 digits
            r'^\d{2}BW\d{3}$',    # XXBW followed by 2 digits
            r'^\d{2}FL\d{3}$',    # XXFL followed by 2 digits
            r'^\d{2}FR\d{3}$',    # XXFR followed by 2 digits
            r'^\d{2}FW---$',       # XXFW--
            r'^\d{2}BW---$',       # XXBW--
            r'^\d{2}FL090$',       # XXFL90
            r'^\d{2}FR090$',       # XXFR90
            r'^\d{2}BR090$',       # XXFR90
            r'^\d{2}BL090$',       # XXFR90
            r'^\d{2}STOP-$',       # XXSTOP
            r'^\d{2}TEST$',       # XXTEST
            # Legacy format support
            r'^FW\d{2}$',         # FW followed by 2 digits
            r'^BW\d{2}$',         # BW followed by 2 digits
            r'^FL\d{2}$',         # FL followed by 2 digits
            r'^FR\d{2}$',         # FR followed by 2 digits
            r'^FW--$',            # FW--
            r'^BW--$',            # BW--
            r'^STOP$',            # STOP
            r'^TEST$',            # TEST
        ]
        
        import re
        return any(re.match(pattern, command) for pattern in valid_patterns)
    
    def send_command(self, command: str) -> bool:
        """
        Send a command to STM32.
        
        Args:
            command: Command to send
            
        Returns:
            bool: True if command was sent successfully, False otherwise
        """
        if not self.validate_command(command):
            print(f"✗ Invalid command format: {command}")
            return False
        
        try:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] Sending command: {command}")
            
            self.stm_link.send(command)
            self.sent_count += 1
            
            print(f"[{timestamp}] ✓ Command sent successfully (#{self.sent_count})")
            return True
            
        except Exception as e:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"[{timestamp}] ✗ Failed to send command: {e}")
            return False
    
    def connect(self) -> bool:
        """Establish serial connection to STM32."""
        try:
            print("Establishing serial connection to STM32...")
            print("Make sure your STM32 is connected via USB...")
            self.stm_link.connect()
            print("✓ Connection established successfully!")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False
    
    def show_help(self):
        """Show help for interactive commands."""
        print("\n=== STM32 Test Commands ===")
        print("Quick commands (XXYYYY format):")
        for key, cmd in self.test_commands.items():
            print(f"  {key}: {cmd}")
        print("\nManual commands:")
        print("  Type any STM32 command (e.g., 00FW05, 00STOP, 00TEST)")
        print("  Format: XXYYYY where XX=detection, YYYY=command")
        print("  Commands: help, quit, stats")
        print("-" * 40)
    
    def interactive_mode(self):
        """Run interactive command sending mode."""
        print(f"\n=== STM32 Interactive Mode ===")
        print("Type commands to send to STM32, or use quick commands above.")
        print("Type 'help' for command reference, 'quit' to exit.")
        print("-" * 60)
        
        self.show_help()
        
        # Start background message receiver
        receiver_thread = threading.Thread(target=self.background_receiver, daemon=True)
        receiver_thread.start()
        
        try:
            while self.running:
                try:
                    user_input = input("\nSTM32> ").strip()
                    
                    if user_input.lower() == 'quit':
                        break
                    elif user_input.lower() == 'help':
                        self.show_help()
                        continue
                    elif user_input.lower() == 'stats':
                        self.show_stats()
                        continue
                    elif user_input in self.test_commands:
                        # Quick command
                        command = self.test_commands[user_input]
                        self.send_command(command)
                    elif user_input:
                        # Manual command
                        self.send_command(user_input)
                        
                except EOFError:
                    break
                except KeyboardInterrupt:
                    break
                    
        except KeyboardInterrupt:
            pass
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Exiting interactive mode...")
    
    def background_receiver(self):
        """Background thread to continuously receive messages."""
        while self.running:
            try:
                message = self.stm_link.recv()
                if message:
                    self.received_count += 1
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"\n[{timestamp}] ← Received: {message}")
                    
                    if message.strip() == "ACK":
                        self.ack_count += 1
                        print(f"[{timestamp}] ✓ ACK received (Total: {self.ack_count})")
                    
                    print("STM32> ", end="", flush=True)  # Re-prompt user
                    
                time.sleep(0.1)
                
            except Exception as e:
                if self.running:  # Only print error if we're still supposed to be running
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"\n[{timestamp}] Error receiving: {e}")
                    print("STM32> ", end="", flush=True)
                time.sleep(1)
    
    def show_stats(self):
        """Show current statistics."""
        print(f"\n=== Statistics ===")
        print(f"Commands sent: {self.sent_count}")
        print(f"Messages received: {self.received_count}")
        print(f"ACKs received: {self.ack_count}")
        print(f"Response rate: {self.ack_count/self.sent_count*100:.1f}%" if self.sent_count > 0 else "Response rate: N/A")
    
    def scan_messages(self, duration: int = 0):
        """
        Continuously scan for messages from STM32.
        
        Args:
            duration: Duration in seconds to scan (0 = infinite until Ctrl+C)
        """
        print(f"\n=== STM32 Message Scanner ===")
        if duration > 0:
            print(f"Scanning for {duration} seconds...")
        else:
            print("Scanning continuously (Press Ctrl+C to stop)...")
        
        print("Timestamp format: [HH:MM:SS]")
        print("Message format: [timestamp] Message #N: content")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            while self.running:
                try:
                    message = self.stm_link.recv()
                    if message:
                        self.received_count += 1
                        timestamp = time.strftime("%H:%M:%S", time.localtime())
                        print(f"[{timestamp}] Message #{self.received_count}: {message}")
                        
                        # Check if it's an ACK
                        if message.strip() == "ACK":
                            self.ack_count += 1
                            print(f"[{timestamp}]   → ACK received (Total ACKs: {self.ack_count})")
                        else:
                            print(f"[{timestamp}]   → Other message: {message.strip()}")
                        
                        print("-" * 60)
                    
                    # Check if duration limit reached
                    if duration > 0 and (time.time() - start_time) >= duration:
                        print(f"\n[{time.strftime('%H:%M:%S')}] Duration limit reached. Stopping...")
                        break
                        
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    timestamp = time.strftime("%H:%M:%S", time.localtime())
                    print(f"[{timestamp}] Error receiving message: {e}")
                    time.sleep(1)  # Wait a bit before retrying
                    
        except KeyboardInterrupt:
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            print(f"\n[{timestamp}] Scanning stopped by user")
        
        elapsed_time = time.time() - start_time
        print(f"\n=== Scan Summary ===")
        print(f"Total messages received: {self.received_count}")
        print(f"ACK messages: {self.ack_count}")
        print(f"Other messages: {self.received_count - self.ack_count}")
        print(f"Scan duration: {elapsed_time:.1f} seconds")
        if self.received_count > 0:
            print(f"Average rate: {self.received_count/elapsed_time:.2f} messages/second")
            print(f"ACK rate: {self.ack_count/elapsed_time:.2f} ACKs/second")
        
        return self.received_count > 0
    
    def disconnect(self):
        """Disconnect from STM32."""
        try:
            print(f"\n[{time.strftime('%H:%M:%S')}] Disconnecting...")
            self.stm_link.disconnect()
            print("✓ Connection closed successfully")
        except Exception as e:
            print(f"✗ Error during cleanup: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='STM32 Message Scanner and Command Sender')
    parser.add_argument('--duration', type=int, default=0,
                       help='Duration for scanning in seconds (0 = infinite)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode for sending commands')
    parser.add_argument('--send', type=str,
                       help='Send a single command and exit')
    parser.add_argument('--help-usage', action='store_true',
                       help='Show usage examples')
    
    args = parser.parse_args()
    
    if args.help_usage:
        print("Usage Examples:")
        print("  python3 scan_stm32_messages.py                    # Scan indefinitely")
        print("  python3 scan_stm32_messages.py --duration 60      # Scan for 60 seconds")
        print("  python3 scan_stm32_messages.py --interactive      # Interactive command mode")
        print("  python3 scan_stm32_messages.py --send 00STOP      # Send single command")
        print("  python3 scan_stm32_messages.py --send 00FW05      # Move forward 5 units")
        print("\nInteractive Mode Commands:")
        print("  1-0: Quick commands (00FW05, 00BW05, 00FL90, 00FR90, etc.)")
        print("  s: 00STOP command")
        print("  t: 00TEST command")
        print("  help: Show command reference")
        print("  stats: Show statistics")
        print("  quit: Exit interactive mode")
        print("\nCommand Format: XXYYYY where XX=detection code, YYYY=command")
        print("Note: Make sure STM32 is connected via USB and powered on.")
        return
    
    scanner = STM32MessageScanner()
    
    try:
        if not scanner.connect():
            print("Failed to establish connection. Exiting.")
            return
        
        if args.interactive:
            # Interactive mode for sending commands
            scanner.interactive_mode()
        elif args.send:
            # Send single command
            print(f"Sending command: {args.send}")
            success = scanner.send_command(args.send)
            if success:
                print("Waiting for response...")
                time.sleep(2)  # Wait a bit for response
                # Try to receive one message
                try:
                    message = scanner.stm_link.recv()
                    if message:
                        print(f"Response: {message}")
                    else:
                        print("No response received")
                except:
                    print("No response received")
        else:
            # Scan mode (original functionality)
            scanner.scan_messages(duration=args.duration)
        
    finally:
        scanner.disconnect()


if __name__ == "__main__":
    main()
