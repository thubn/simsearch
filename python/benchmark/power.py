import pyvisa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class N6705C:

    instrument=None

    def __init__(self):
        rm=pyvisa.ResourceManager()
        self.instrument=rm.open_resource("TCPIP::192.168.1.10::5025::SOCKET")
        self.instrument.read_termination='\n'
        self.instrument.write_termination='\n'
        print(self.instrument.query("*IDN?"))

    def ch0_off(self):
        self.instrument.write("OUTP OFF, (@1)")

    def ch0_on(self):
        self.instrument.write("OUTP ON, (@1)")

    def ch0_measure(self, interval=0.001, mtime=10, curr_range=2, volr_range=6):
        # self.instrument.write("FORM REAL")
        self.instrument.write("FORM ASCII")
        # print("Measure format set to "+self.instrument.query("FORM?"))
        self.instrument.write("SENS:SWE:TINT "+ str(interval)+", (@1)")
        set_interval=self.instrument.query("SENS:SWE:TINT? (@1)")
        # print("Measure timesteps set to "+set_interval)
        npoints=int(mtime/float(set_interval))
        self.instrument.write("SENS:SWE:POIN "+ str(npoints)+", (@1)")
        print("Measure samples set to "+self.instrument.query("SENS:SWE:POIN? (@1)"))
        self.instrument.write("SENS:CURR:RANG "+ str(curr_range)+", (@1)")
        print("Measure current range set to "+self.instrument.query("SENS:CURR:RANG? (@1)"))
        self.instrument.write("SENS:VOLT:RANG "+ str(volr_range)+", (@1)")
        print("Measure voltage range set to "+self.instrument.query("SENS:VOLT:RANG? (@1)"))
        

        a_mres_power=self.instrument.query("MEAS:ARR:POW? (@1)", delay=mtime)
        a_mres_current=self.instrument.query("FETC:ARR:CURR? (@1)")
        a_mres_voltage=self.instrument.query("FETC:ARR:VOLT? (@1)")
        res_arr=a_mres_power.split(",")
        mres_power=[]
        for res in res_arr:
            mres_power.append(float(res))

        res_arr=a_mres_current.split(",")
        mres_current=[]
        for res in res_arr:
            mres_current.append(float(res))

        res_arr=a_mres_voltage.split(",")
        mres_voltage=[]
        for res in res_arr:
            mres_voltage.append(float(res))

        # mres_power=self.instrument.query_binary_values("MEAS:ARR:POW? (@1)", datatype='f', delay=mtime)
        # mres_current=self.instrument.query_binary_values("FETC:ARR:CURR? (@1)", datatype='f')
        # mres_voltage=self.instrument.query_binary_values("FETC:ARR:VOLT? (@1)", datatype='f')
        return(mres_power, mres_current, mres_voltage, float(set_interval))
    
    def to_dataframe(self, data_p, data_c, data_v, interval):
        times = np.multiply(range(0,len(data_p)), interval)
        data = pd.DataFrame({'timestamp': times, 'power': data_p, 'current': data_c, 'voltage': data_v})

        return data

    def preview_measure(self, data_p, data_c, data_v, interval):
        fig, axs=plt.subplots(3)
        fig.suptitle("Measurement Results (preview), TOTAL "+str(sum(data_p)*interval)+" Wh [J]")
        axs[0].plot(np.multiply(range(0,len(data_p)), interval), data_p)
        axs[0].set_title("Power Consumption (AVG: "+str(sum(data_p)/len(data_p))+") [W]")
        # axs[0].set_ylim([0, 0.004])

        axs[1].plot(np.multiply(range(0,len(data_v)), interval), data_v)
        axs[1].set_title("Voltage (AVG: "+str(sum(data_v)/len(data_v))+") [V]")
        # axs[1].set_ylim([0, max(data_v)*1.1])
        axs[2].plot(np.multiply(range(0,len(data_c)), interval), data_c)
        axs[2].set_title("Current (AVG: "+str(sum(data_c)/len(data_c))+") [A]")
        # axs[2].set_ylim([0.0004, 0.0006])

        plt.xlabel("time [s]")
        plt.show()
        plt.savefig('measurement_results.png', dpi=300, bbox_inches='tight')

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
            
            # Request array measurements
            a_mres_power = self.instrument.query(f"MEAS:ARR:POW? (@1)", delay=self.measurement_interval*self.buffer_size*1.1)
            a_mres_current = self.instrument.query("FETC:ARR:CURR? (@1)")
            a_mres_voltage = self.instrument.query("FETC:ARR:VOLT? (@1)")
            
            # Parse power values
            res_arr = a_mres_power.split(",")
            self.power_buffer = [float(res) for res in res_arr]
            
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
        import threading
        self.measurement_thread = threading.Thread(target=self._continuous_measurement_worker)
        self.measurement_thread.daemon = True
        self.measurement_thread.start()
        
        print(f"Continuous measurement started with interval {interval}s")

    def stop_continuous_measurement(self):
        """Stop the continuous measurement thread and return collected data"""
        if hasattr(self, 'continuous_running') and self.continuous_running:
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
            
            duration = len(power_data) * interval
            print(f"Continuous measurement stopped, collected {len(power_data)} samples ({duration:.2f}s)")
            
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
                try:
                    # Take direct measurements from the instrument
                    power_val = float(self.instrument.query("MEAS:POW? (@1)"))
                    current_val = float(self.instrument.query("MEAS:CURR? (@1)"))
                    voltage_val = float(self.instrument.query("MEAS:VOLT? (@1)"))
                    
                    # Store the measurements
                    self.continuous_power.append(power_val)
                    self.continuous_current.append(current_val)
                    self.continuous_voltage.append(voltage_val)
                    
                    sample_count += 1
                    if sample_count % 100 == 0:
                        elapsed = time.time() - start_time
                        print(f"Power sampling: {sample_count} samples collected over {elapsed:.2f}s ({sample_count/elapsed:.1f} samples/sec)")
                    
                except Exception as e:
                    print(f"Error taking direct measurement: {str(e)}")
                    # Add a zero reading on error to maintain timing
                    self.continuous_power.append(0.0)
                    self.continuous_current.append(0.0)
                    self.continuous_voltage.append(0.0)
                
                # Sleep between measurements to avoid overwhelming the instrument
                # Adjust timing based on your instrument's capabilities
                import time
                time.sleep(self.measurement_interval)
                
        except Exception as e:
            print(f"Error in continuous measurement worker: {str(e)}")
            elapsed = time.time() - start_time
            print(f"Thread ran for {elapsed:.2f}s and collected {sample_count} samples")

power_measure = N6705C()


duration = 30
# power_measure.ch0_on()
power, current, voltage, interval = power_measure.ch0_measure(mtime=duration)
# power_measure.ch0_off()
power_measure.preview_measure(power, current, voltage, interval)


data = power_measure.to_dataframe(power, current, voltage, interval)
print(data)