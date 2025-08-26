import mido
import time

# Replace 'PreSonus ATOM' with the exact name of your device's port
port_name = 'ATOM 0' 

try:
    print(f"Opening port: {port_name}...")
    with mido.open_input(port_name) as port:
        for msg in port:
            # Check if the message is a 'note_on' message (i.e., a pad was pressed)
            if msg.type == 'note_on':
                print(f"Pad pressed! Note: {msg.note}, Velocity: {msg.velocity}")
            
            # Check for other message types, like 'control_change'
            elif msg.type == 'control_change':
                print(f"Control changed! Controller: {msg.control}, Value: {msg.value}")
            
            # You can handle other message types here as needed
            else:
                print(f"Received message: {msg}")

except mido.PortNotOpenError:
    print(f"Error: Could not open port '{port_name}'. Make sure your device is connected and recognized.")
    print("Available ports:")
    print(mido.get_input_names())
except KeyboardInterrupt:
    print("\nExiting.")
finally:
    print("Port closed.")