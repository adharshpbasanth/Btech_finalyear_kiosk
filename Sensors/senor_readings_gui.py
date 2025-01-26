import os
import tkinter
import max30102
import hrcalc

import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
#GPIO.setup(18, GPIO.IN)

from smbus2 import SMBus
from mlx90614 import MLX90614

bus = SMBus(1)
sensor = MLX90614(bus, address=0x5A)
m = max30102.MAX30102()
hr2 = 0
sp2 = 0


class App:
    def __init__(self, window, window_title):
        self.window = window
        
        self.Temp1Lbl = tkinter.Label(window, text="[Body Temperature    : ]",font=("Arial", 20), fg = "orange",relief="ridge",borderwidth = 2)
        self.Temp1Lbl.pack(anchor=tkinter.CENTER, expand=True)

        self.Temp2Lbl = tkinter.Label(window, text="[Ambient Temperature : ]",font=("Arial", 20), fg = "dark green",relief="ridge",borderwidth = 2)
        self.Temp2Lbl.pack(anchor=tkinter.CENTER, expand=True)

        self.PulseLbl = tkinter.Label(window, text="[Heart Pulse Rate    : ]",font=("Arial", 20), fg = "red",relief="ridge",borderwidth = 2)
        self.PulseLbl.pack(anchor=tkinter.CENTER, expand=True)

        self.SPO2Lbl = tkinter.Label(window, text="[Oxygen Saturation   : ]",font=("Arial", 20), fg ="blue",relief="ridge",borderwidth = 2)
        self.SPO2Lbl.pack(anchor=tkinter.CENTER, expand=True)

        
        self.delay = 30
        self.update()

        self.window.mainloop()

 
    def update(self):
        celcius = sensor.get_object_1()
        faren = (celcius*1.8)+32
        ambient = sensor.get_ambient()
        self.Temp1Lbl['text'] = "[Body Temperature    : "+str(round(faren, 2))+u"\N{DEGREE SIGN}F]"
        self.Temp2Lbl['text'] = "[Ambient Temperature : "+str(round(ambient, 2))+u"\N{DEGREE SIGN}C]"
        red, ir = m.read_sequential()
        hr,hrb,sp,spb = hrcalc.calc_hr_and_spo2(ir, red)
        if(hrb == True and hr != -999 and hr < 105):
            hr2 = int(hr)
            #print("Heart Rate : ",hr2)
            self.PulseLbl['text'] = "[Heart Pulse Rate    : "+str(hr2)+"bpm]"
        if(spb == True and sp != -999 and sp < 100):
            sp2 = int(sp)
            #print("SPO2       : ",sp2)
            self.SPO2Lbl['text'] = "[Oxygen Saturation   : "+str(sp2)+"%]"
        self.window.after(self.delay, self.update)
            


# Create a window and pass it to the Application object
root = tkinter.Tk()
root.geometry("+{}+{}".format(250, 50))
App(root, "Sensor readings")
