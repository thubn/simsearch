import pyvisa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class N6705C:

    instrument=None

    def __init__(self):
        rm=pyvisa.ResourceManager()
        self.instrument=rm.open_resource("TCPIP::129.217.34.156::5025::SOCKET")
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

power_measure = N6705C()


duration = 30
# power_measure.ch0_on()
power, current, voltage, interval = power_measure.ch0_measure(mtime=duration)
# power_measure.ch0_off()
power_measure.preview_measure(power, current, voltage, interval)


data = power_measure.to_dataframe(power, current, voltage, interval)
print(data)