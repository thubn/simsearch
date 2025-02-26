import time
import random
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

class FakeInstrument:
    """A fake VISA instrument that simulates basic communication"""
    
    def __init__(self):
        self.config = {
            "FORM": "ASCII",
            "SENS:CURR:RANG": "2",
            "SENS:VOLT:RANG": "6",
            "SENS:SWE:TINT": "0.01",
            "SENS:SWE:POIN": "100",
            "OUTP": "OFF"
        }
        
    def write(self, command: str):
        """Simulates writing a command to the instrument"""
        if ":" in command:
            cmd_parts = command.split(" ", 1)
            cmd = cmd_parts[0]
            
            # Handle setting values
            if "OUTP OFF" in command:
                self.config["OUTP"] = "OFF"
            elif "OUTP ON" in command:
                self.config["OUTP"] = "ON"
            elif "FORM" in cmd:
                self.config["FORM"] = cmd_parts[1]
            elif "SENS:CURR:RANG" in cmd:
                value = cmd_parts[1].split(",")[0].strip()
                self.config["SENS:CURR:RANG"] = value
            elif "SENS:VOLT:RANG" in cmd:
                value = cmd_parts[1].split(",")[0].strip()
                self.config["SENS:VOLT:RANG"] = value
            elif "SENS:SWE:TINT" in cmd:
                value = cmd_parts[1].split(",")[0].strip()
                self.config["SENS:SWE:TINT"] = value
            elif "SENS:SWE:POIN" in cmd:
                value = cmd_parts[1].split(",")[0].strip()
                self.config["SENS:SWE:POIN"] = value
                
    def query(self, command: str, delay: float = 0) -> str:
        """Simulates querying the instrument and getting a response"""
        # Simulate the device taking time to respond
        if delay > 0:
            time.sleep(delay * 0.1)  # Speed up simulation by using only 10% of the real delay
            
        if command == "*IDN?":
            return "EMULATED,N6705C,12345,1.0"
            
        elif "SENS:SWE:TINT? (@1)" in command:
            return self.config["SENS:SWE:TINT"]
            
        elif "SENS:SWE:POIN? (@1)" in command:
            return self.config["SENS:SWE:POIN"]
            
        elif "SENS:CURR:RANG? (@1)" in command:
            return self.config["SENS:CURR:RANG"]
            
        elif "SENS:VOLT:RANG? (@1)" in command:
            return self.config["SENS:VOLT:RANG"]
            
        elif "FORM?" in command:
            return self.config["FORM"]
            
        elif "MEAS:ARR:POW? (@1)" in command:
            # Generate simulated power measurements
            points = int(self.config["SENS:SWE:POIN"])
            baseline = 10.0  # 10 watts baseline
            
            if self.config["OUTP"] == "OFF":
                # If output is off, return very low power values
                return ",".join([str(random.uniform(0.01, 0.1)) for _ in range(points)])
            
            # Otherwise return realistic power values with fluctuations
            values = []
            for i in range(points):
                # Add some realistic variations
                noise = random.uniform(-0.5, 0.5)  # Small random noise
                trend = 0.1 * np.sin(i / 10)  # Slight sinusoidal pattern
                value = baseline + noise + trend
                values.append(str(value))
            
            return ",".join(values)
            
        elif "FETC:ARR:CURR? (@1)" in command:
            # Generate simulated current measurements
            points = int(self.config["SENS:SWE:POIN"])
            baseline = 0.83  # Assuming ~0.83A for 10W at 12V
            
            if self.config["OUTP"] == "OFF":
                return ",".join([str(random.uniform(0.001, 0.01)) for _ in range(points)])
            
            values = []
            for i in range(points):
                noise = random.uniform(-0.05, 0.05)
                trend = 0.01 * np.sin(i / 10)
                value = baseline + noise + trend
                values.append(str(value))
            
            return ",".join(values)
            
        elif "FETC:ARR:VOLT? (@1)" in command:
            # Generate simulated voltage measurements
            points = int(self.config["SENS:SWE:POIN"])
            baseline = 12.0  # Assuming 12V supply
            
            values = []
            for i in range(points):
                noise = random.uniform(-0.1, 0.1)
                value = baseline + noise
                values.append(str(value))
            
            return ",".join(values)
            
        elif "MEAS:POW? (@1)" in command:
            # Single power measurement
            if self.config["OUTP"] == "OFF":
                return str(random.uniform(0.01, 0.1))
            return str(10.0 + random.uniform(-0.5, 0.5))
            
        elif "MEAS:CURR? (@1)" in command:
            # Single current measurement
            if self.config["OUTP"] == "OFF":
                return str(random.uniform(0.001, 0.01))
            return str(0.83 + random.uniform(-0.05, 0.05))
            
        elif "MEAS:VOLT? (@1)" in command:
            # Single voltage measurement
            return str(12.0 + random.uniform(-0.1, 0.1))
            
        else:
            return "0.0"  # Default response


class N6705C:
    """Emulated N6705C power analyzer that provides the same interface as the real one"""

    def __init__(self):
        # Create a fake instrument instead of connecting to real hardware
        self.instrument = FakeInstrument()
        print(self.instrument.query("*IDN?"))
        
        # Setup profile for power consumption patterns
        self.power_profiles = {
            "float": {"base": 15.0, "variance": 1.0},
            "binary": {"base": 8.0, "variance": 0.8},
            "int8": {"base": 7.5, "variance": 0.5},
            "pq": {"base": 9.0, "variance": 0.7},
            "default": {"base": 10.0, "variance": 1.0}
        }
        self.current_profile = "default"

    def set_power_profile(self, profile_name):
        """Set the current power consumption profile"""
        if profile_name in self.power_profiles:
            self.current_profile = profile_name
            print(f"Power profile set to {profile_name}")
        else:
            print(f"Profile {profile_name} not found, using default")
            self.current_profile = "default"

    def ch0_off(self):
        """Turn off channel 0"""
        self.instrument.write("OUTP OFF, (@1)")
        print("Channel 0 turned OFF")

    def ch0_on(self):
        """Turn on channel 0"""
        self.instrument.write("OUTP ON, (@1)")
        print("Channel 0 turned ON")

    def ch0_measure(self, interval=0.001, mtime=10, curr_range=2, volr_range=6):
        """Emulate taking a set of measurements"""
        # Configure the emulated device
        self.instrument.write("FORM ASCII")
        self.instrument.write(f"SENS:SWE:TINT {interval}, (@1)")
        set_interval = float(self.instrument.query("SENS:SWE:TINT? (@1)"))
        
        npoints = int(mtime/float(set_interval))
        self.instrument.write(f"SENS:SWE:POIN {npoints}, (@1)")
        print(f"Measure samples set to {self.instrument.query('SENS:SWE:POIN? (@1)')}")
        
        self.instrument.write(f"SENS:CURR:RANG {curr_range}, (@1)")
        print(f"Measure current range set to {self.instrument.query('SENS:CURR:RANG? (@1)')}")
        
        self.instrument.write(f"SENS:VOLT:RANG {volr_range}, (@1)")
        print(f"Measure voltage range set to {self.instrument.query('SENS:VOLT:RANG? (@1)')}")
        
        # Get the power profile for the current method
        profile = self.power_profiles[self.current_profile]
        
        # Simulate device taking time to gather measurements
        print(f"Measuring for {mtime} seconds...")
        time.sleep(min(mtime * 0.05, 1.0))  # Simulate some delay but don't wait too long
        
        # Get measurements from the emulated instrument
        a_mres_power = self.instrument.query(f"MEAS:ARR:POW? (@1)", delay=0.1)
        a_mres_current = self.instrument.query("FETC:ARR:CURR? (@1)")
        a_mres_voltage = self.instrument.query("FETC:ARR:VOLT? (@1)")
        
        # Parse the results
        mres_power = [float(res) for res in a_mres_power.split(",")]
        mres_current = [float(res) for res in a_mres_current.split(",")]
        mres_voltage = [float(res) for res in a_mres_voltage.split(",")]
        
        # Apply the profile-based adjustment to make measurements method-specific
        base_power = profile["base"]
        variance = profile["variance"]
        
        # Adjust power values based on the profile
        mres_power = [p * (base_power/10.0) + random.uniform(-variance, variance) for p in mres_power]
        
        return mres_power, mres_current, mres_voltage, float(set_interval)
    
    def to_dataframe(self, data_p, data_c, data_v, interval):
        """Convert measurement data to DataFrame"""
        times = np.multiply(range(0, len(data_p)), interval)
        data = pd.DataFrame({'timestamp': times, 'power': data_p, 'current': data_c, 'voltage': data_v})
        return data

    def preview_measure(self, data_p, data_c, data_v, interval):
        """Create a preview plot of the measurements"""
        fig, axs = plt.subplots(3)
        fig.suptitle(f"Measurement Results (preview), TOTAL {sum(data_p)*interval:.6f} Wh [J]")
        
        axs[0].plot(np.multiply(range(0, len(data_p)), interval), data_p)
        axs[0].set_title(f"Power Consumption (AVG: {sum(data_p)/len(data_p):.6f}) [W]")
        
        axs[1].plot(np.multiply(range(0, len(data_v)), interval), data_v)
        axs[1].set_title(f"Voltage (AVG: {sum(data_v)/len(data_v):.6f}) [V]")
        
        axs[2].plot(np.multiply(range(0, len(data_c)), interval), data_c)
        axs[2].set_title(f"Current (AVG: {sum(data_c)/len(data_c):.6f}) [A]")
        
        plt.xlabel("time [s]")
        plt.tight_layout()
        plt.savefig('measurement_results.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to avoid displaying it in non-interactive environments

    def configure_measurement(self, interval=0.01, buffer_size=100):
        """Configure the device for buffered measurements"""
        # Set the format to ASCII for easier parsing
        self.instrument.write("FORM ASCII")
        
        # Configure the ranges for measurement
        self.instrument.write("SENS:CURR:RANG 2, (@1)")
        self.instrument.write("SENS:VOLT:RANG 6, (@1)")
        
        # Set time interval
        self.instrument.write(f"SENS:SWE:TINT {interval}, (@1)")
        set_interval = float(self.instrument.query("SENS:SWE:TINT? (@1)"))
        
        # Configure buffer size for array measurements
        self.instrument.write(f"SENS:SWE:POIN {buffer_size}, (@1)")
        print(f"Device configured for buffered measurements with interval {set_interval}s and buffer size {buffer_size}")
        
        # Save configuration for later use
        self.measurement_interval = set_interval
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.power_buffer = []
        self.current_buffer = []
        self.voltage_buffer = []
        
        return set_interval

    def get_single_measurement(self):
        """Get a single power, current, voltage measurement"""
        try:
            # If buffer is empty or completely used, refill it
            if not hasattr(self, 'buffer_index') or self.buffer_index >= len(self.power_buffer):
                self._refill_buffer()
                
            # Get current values from buffer
            power = self.power_buffer[self.buffer_index]
            current = self.current_buffer[self.buffer_index]
            voltage = self.voltage_buffer[self.buffer_index]
            
            # Move to next position in buffer
            self.buffer_index += 1
            
            return power, current, voltage
        except Exception as e:
            print(f"Error during buffered measurement: {str(e)}")
            # Return zeros if measurement fails
            return 0.0, 0.0, 0.0

    def _refill_buffer(self):
        """Refill the measurement buffers with a new batch of measurements"""
        try:
            print("Refilling measurement buffer...")
            
            # Get the current power profile
            profile = self.power_profiles.get(self.current_profile, self.power_profiles["default"])
            base_power = profile["base"]
            variance = profile["variance"]
            
            # Request array measurements
            a_mres_power = self.instrument.query(f"MEAS:ARR:POW? (@1)", delay=0.1)
            a_mres_current = self.instrument.query("FETC:ARR:CURR? (@1)")
            a_mres_voltage = self.instrument.query("FETC:ARR:VOLT? (@1)")
            
            # Parse power values
            res_arr = a_mres_power.split(",")
            self.power_buffer = [float(res) * (base_power/10.0) + random.uniform(-variance, variance) 
                                for res in res_arr]
            
            # Parse current values
            res_arr = a_mres_current.split(",")
            self.current_buffer = [float(res) for res in res_arr]
            
            # Parse voltage values
            res_arr = a_mres_voltage.split(",")
            self.voltage_buffer = [float(res) for res in res_arr]
            
            # Reset buffer index
            self.buffer_index = 0
            
            print(f"Buffer refilled with {len(self.power_buffer)} measurements")
        except Exception as e:
            print(f"Error refilling measurement buffer: {str(e)}")
            # Create empty buffers to avoid further errors
            self.power_buffer = [0.0]
            self.current_buffer = [0.0]
            self.voltage_buffer = [0.0]
            self.buffer_index = 0

    def get_measurement_stats(self, method_name=None):
        """Get statistics about the measurements taken for a method"""
        if not hasattr(self, 'power_buffer') or not self.power_buffer:
            return {
                "power_avg": 0.0,
                "power_min": 0.0,
                "power_max": 0.0,
                "current_avg": 0.0,
                "voltage_avg": 0.0,
                "samples": 0
            }
        
        # Set the power profile based on method name for more realistic emulation
        if method_name:
            for profile_name in self.power_profiles:
                if profile_name in method_name.lower():
                    self.set_power_profile(profile_name)
                    break
        
        stats = {
            "power_avg": sum(self.power_buffer) / len(self.power_buffer),
            "power_min": min(self.power_buffer),
            "power_max": max(self.power_buffer),
            "current_avg": sum(self.current_buffer) / len(self.current_buffer),
            "voltage_avg": sum(self.voltage_buffer) / len(self.voltage_buffer),
            "samples": len(self.power_buffer)
        }
        
        if method_name:
            print(f"Measurement stats for {method_name}: {stats['samples']} samples, " + 
                  f"avg power: {stats['power_avg']:.6f}W (min: {stats['power_min']:.6f}W, max: {stats['power_max']:.6f}W)")
        
        return stats

    def start_continuous_measurement(self, interval=0.01, buffer_size=100):
        """Start continuous measurement in the background"""
        # Configure the measurement
        self.configure_measurement(interval, buffer_size)
        
        # Initialize data collection lists
        self.continuous_power = []
        self.continuous_current = []
        self.continuous_voltage = []
        self.continuous_running = True
        
        # Start a thread to continuously collect measurements
        self.measurement_thread = threading.Thread(target=self._continuous_measurement_worker)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        
        print(f"Continuous measurement started with interval {interval}s")

    def stop_continuous_measurement(self):
        """Stop the continuous measurement thread and return collected data"""
        if hasattr(self, 'continuous_running') and self.continuous_running:
            # Record how long we've been measuring
            if hasattr(self, 'measurement_thread') and self.measurement_thread.is_alive():
                print(f"Stopping continuous measurement with {len(getattr(self, 'continuous_power', []))} samples collected")
                
            self.continuous_running = False
            
            # Wait for thread to finish
            if hasattr(self, 'measurement_thread') and self.measurement_thread.is_alive():
                self.measurement_thread.join(timeout=2.0)
            
            # Prepare return data
            power_data = self.continuous_power.copy() if hasattr(self, 'continuous_power') else []
            current_data = self.continuous_current.copy() if hasattr(self, 'continuous_current') else []
            voltage_data = self.continuous_voltage.copy() if hasattr(self, 'continuous_voltage') else []
            interval = self.measurement_interval if hasattr(self, 'measurement_interval') else 0.01
            
            print(f"Continuous measurement stopped, collected {len(power_data)} samples ({len(power_data) * interval:.2f}s)")
            
            # Clear the data to free memory
            if hasattr(self, 'continuous_power'):
                self.continuous_power = []
                self.continuous_current = []
                self.continuous_voltage = []
            
            return power_data, current_data, voltage_data, interval
        
        # Return empty data if not running
        return [], [], [], 0.01

    def _continuous_measurement_worker(self):
        """Worker function for continuous measurement thread"""
        try:
            sample_count = 0
            start_time = time.time()
            
            while self.continuous_running:
                # Get a direct measurement rather than from buffer
                if hasattr(self, 'instrument') and self.instrument:
                    # Get measurements directly from instrument
                    try:
                        # Get the current power profile
                        profile = getattr(self, 'current_profile', 'default')
                        profile_settings = self.power_profiles.get(profile, self.power_profiles['default'])
                        base_power = profile_settings['base']
                        variance = profile_settings['variance']
                        
                        # Simulate a direct measurement with profile-specific characteristics
                        power = base_power + random.uniform(-variance, variance)
                        current = power / 12.0  # Assuming 12V supply
                        voltage = 12.0 + random.uniform(-0.1, 0.1)
                        
                        # Store the measurements
                        self.continuous_power.append(power)
                        self.continuous_current.append(current)
                        self.continuous_voltage.append(voltage)
                        
                        sample_count += 1
                        if sample_count % 100 == 0:
                            elapsed = time.time() - start_time
                            print(f"Power sampling: {sample_count} samples collected over {elapsed:.2f}s ({sample_count/elapsed:.1f} samples/sec)")
                    
                    except Exception as e:
                        print(f"Error taking direct measurement: {str(e)}")
                        # Add a zero reading on error
                        self.continuous_power.append(0.0)
                        self.continuous_current.append(0.0)
                        self.continuous_voltage.append(0.0)
                else:
                    # Fallback if no instrument
                    self.continuous_power.append(0.0)
                    self.continuous_current.append(0.0)
                    self.continuous_voltage.append(0.0)
                
                # Sleep briefly between measurements
                time.sleep(self.measurement_interval)
                
        except Exception as e:
            print(f"Error in continuous measurement worker: {str(e)}")
            elapsed = time.time() - start_time
            print(f"Thread ran for {elapsed:.2f}s and collected {sample_count} samples")


# Test code for the emulator
if __name__ == "__main__":
    power_measure = N6705C()
    
    print("\nTesting different power profiles:")
    for profile in ["float", "binary", "int8", "pq"]:
        power_measure.set_power_profile(profile)
        print(f"\nMeasuring with {profile} profile:")
        power_measure.ch0_on()
        power, current, voltage, interval = power_measure.ch0_measure(mtime=2)
        print(f"Average power: {sum(power)/len(power):.4f}W")
    
    print("\nTesting continuous measurement:")
    power_measure.set_power_profile("float")
    power_measure.start_continuous_measurement(interval=0.05, buffer_size=20)
    print("Continuous measurement running for 2 seconds...")
    time.sleep(2)
    power, current, voltage, interval = power_measure.stop_continuous_measurement()
    print(f"Collected {len(power)} samples with average power {sum(power)/len(power):.4f}W")
    
    print("\nGenerating preview plot...")
    power_measure.preview_measure(power, current, voltage, interval)
    print("Done. Check for measurement_results.png")