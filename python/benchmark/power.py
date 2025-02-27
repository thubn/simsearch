#!/usr/bin/env python3

import pyvisa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class N6705C:
    """
    Interface for Keysight/Agilent N6705C DC Power Analyzer.
    
    This class provides methods to connect to and control an N6705C power analyzer
    over TCP/IP, allowing power measurement of connected devices. The implementation
    focuses on channel 2 (@2) for all operations.
    """

    instrument = None  # Holds the PyVISA instrument resource

    def __init__(self):
        """
        Initialize connection to the N6705C power analyzer.
        
        Establishes a TCP/IP socket connection to the analyzer at 192.168.1.10:5025,
        configures communication parameters, and verifies connection by querying
        device identification.
        """
        rm = pyvisa.ResourceManager()
        self.instrument = rm.open_resource("TCPIP::192.168.1.10::5025::SOCKET")
        self.instrument.read_termination = '\n'
        self.instrument.write_termination = '\n'
        print(self.instrument.query("*IDN?"))

    def ch0_off(self):
        """Turn off channel 2 of the power analyzer."""
        self.instrument.write("OUTP OFF, (@2)")

    def ch0_on(self):
        """Turn on channel 2 of the power analyzer."""
        self.instrument.write("OUTP ON, (@2)")

    def ch0_measure(self, interval=0.001, mtime=10, curr_range=0.001, volr_range=5):
        """
        Configure and perform power measurements on channel 2.
        
        Parameters:
            interval (float): Sample interval in seconds (default: 0.001s)
            mtime (float): Total measurement time in seconds (default: 10s)
            curr_range (float): Current measurement range in Amperes (default: 0.001A)
            volr_range (float): Voltage measurement range in Volts (default: 5V)
            
        Returns:
            tuple: (power_data, current_data, voltage_data, actual_interval)
                - Lists of measured values and the actual sampling interval used
        """
        # Configure ASCII data format for results
        self.instrument.write("FORM ASCII")
        
        # Set measurement time interval
        self.instrument.write("SENS:SWE:TINT " + str(interval) + ", (@2)")
        set_interval = self.instrument.query("SENS:SWE:TINT? (@2)")
        
        # Calculate and set number of measurement points
        npoints = int(mtime/float(set_interval))
        self.instrument.write("SENS:SWE:POIN " + str(npoints) + ", (@2)")
        
        # Configure measurement ranges
        self.instrument.write("SENS:CURR:RANG " + str(curr_range) + ", (@2)")
        self.instrument.write("SENS:VOLT:RANG " + str(volr_range) + ", (@2)")

        # Perform measurements and retrieve data
        a_mres_power = self.instrument.query("MEAS:ARR:POW? (@2)", delay=mtime)
        a_mres_current = self.instrument.query("FETC:ARR:CURR? (@2)")
        a_mres_voltage = self.instrument.query("FETC:ARR:VOLT? (@2)")
        
        # Process power measurement data
        res_arr = a_mres_power.split(",")
        mres_power = []
        for res in res_arr:
            mres_power.append(float(res))

        # Process current measurement data
        res_arr = a_mres_current.split(",")
        mres_current = []
        for res in res_arr:
            mres_current.append(float(res))

        # Process voltage measurement data
        res_arr = a_mres_voltage.split(",")
        mres_voltage = []
        for res in res_arr:
            mres_voltage.append(float(res))

        # mres_power=self.instrument.query_binary_values("MEAS:ARR:POW? (@2)", datatype='f', delay=mtime)
        # mres_current=self.instrument.query_binary_values("FETC:ARR:CURR? (@2)", datatype='f')
        # mres_voltage=self.instrument.query_binary_values("FETC:ARR:VOLT? (@2)", datatype='f')
        return(mres_power, mres_current, mres_voltage, float(set_interval))
    
    def to_dataframe(self, data_p, data_c, data_v, interval):
        """
        Convert measurement data to pandas DataFrame.
        
        Parameters:
            data_p (list): Power measurements in Watts
            data_c (list): Current measurements in Amperes
            data_v (list): Voltage measurements in Volts
            interval (float): Time interval between measurements
            
        Returns:
            pandas.DataFrame: DataFrame with timestamp and measurement columns
        """
        times = np.multiply(range(0, len(data_p)), interval)
        data = pd.DataFrame({'timestamp': times, 'power': data_p, 'current': data_c, 'voltage': data_v})
        return data

    def preview_measure(self, data_p, data_c, data_v, interval):
        """
        Visualize measurement data in a three-panel plot.
        
        Creates a figure with three subplots showing power, voltage, and current
        measurements over time, including average values and total energy consumed.
        
        Parameters:
            data_p (list): Power measurements in Watts
            data_c (list): Current measurements in Amperes
            data_v (list): Voltage measurements in Volts
            interval (float): Time interval between measurements
        """
        fig, axs = plt.subplots(3)
        fig.suptitle("Measurement Results (preview), TOTAL " + str(sum(data_p)*interval) + " Wh [J]")
        
        # Power plot
        axs[0].plot(np.multiply(range(0, len(data_p)), interval), data_p)
        axs[0].set_title("Power Consumption (AVG: " + str(sum(data_p)/len(data_p)) + ") [W]")
        axs[0].set_ylim([0, 0.004])
        
        # Voltage plot
        axs[1].plot(np.multiply(range(0, len(data_v)), interval), data_v)
        axs[1].set_title("Voltage (AVG: " + str(sum(data_v)/len(data_v)) + ") [V]")
        axs[1].set_ylim([0, max(data_v)*1.1])
        
        # Current plot
        axs[2].plot(np.multiply(range(0, len(data_c)), interval), data_c)
        axs[2].set_title("Current (AVG: " + str(sum(data_c)/len(data_c)) + ") [A]")
        axs[2].set_ylim([0.0004, 0.0006])

        plt.xlabel("time [s]")
        plt.show()

power_measure = N6705C()



power_measure.ch0_on()
power, current, voltage, interval = power_measure.ch0_measure(mtime=duration)
power_measure.ch0_off()
data = power_measure.to_dataframe(power, current, voltage, interval)