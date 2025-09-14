import tkinter as tk
from functools import partial

class TkApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.drum_vol = self.bass_vol = self.melody_vol = self.vocal_vol = None
        self.button1 = self.button2 = self.button3 = self.button4 = self.button5 = None
        self.__init_ui()

    def __init_ui(self):
        self.title('InsideOut')
        self.geometry('800x600')
        self.drum_vol = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.bass_vol = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.melody_vol = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.vocal_vol = tk.Scale(self, from_=1, to=0, resolution=0.01)
        self.drum_vol.set(1)
        self.bass_vol.set(1)
        self.melody_vol.set(1)
        self.vocal_vol.set(1)

        self.drum_btn_1 = tk.Button(self, text='Drums (on)', width=25, pady=20)
        self.drum_btn_2 = tk.Button(self, text='Drums (on)', width=25, pady=20)
        self.drum_btn_3 = tk.Button(self, text='Drums (on)', width=25, pady=20)
        self.drum_btn_4 = tk.Button(self, text='Drums (on)', width=25, pady=20)

        self.vocal_btn_1 = tk.Button(self, text='Vocals (on)', width=25, pady=20)
        self.vocal_btn_2 = tk.Button(self, text='Vocals (on)', width=25, pady=20)
        self.vocal_btn_3 = tk.Button(self, text='Vocals (on)', width=25, pady=20)
        self.vocal_btn_4 = tk.Button(self, text='Vocals (on)', width=25, pady=20)

        self.melody_btn_1 = tk.Button(self, text='Melody (on)', width=25, pady=20)
        self.melody_btn_2 = tk.Button(self, text='Melody (on)', width=25, pady=20)
        self.melody_btn_3 = tk.Button(self, text='Melody (on)', width=25, pady=20)
        self.melody_btn_4 = tk.Button(self, text='Melody (on)', width=25, pady=20)

        self.bass_btn_1 = tk.Button(self, text='Bass (on)', width=25, pady=20)
        self.bass_btn_2 = tk.Button(self, text='Bass (on)', width=25, pady=20)
        self.bass_btn_3 = tk.Button(self, text='Bass (on)', width=25, pady=20)
        self.bass_btn_4 = tk.Button(self, text='Bass (on)', width=25, pady=20)

        self.drum_btn_1.grid(row=1, column=0)
        self.drum_btn_2.grid(row=2, column=0)
        self.drum_btn_3.grid(row=3, column=0)
        self.drum_btn_4.grid(row=4, column=0)

        self.bass_btn_1.grid(row=1, column=1)
        self.bass_btn_2.grid(row=2, column=1)
        self.bass_btn_3.grid(row=3, column=1)
        self.bass_btn_4.grid(row=4, column=1)

        self.melody_btn_1.grid(row=1, column=2)
        self.melody_btn_2.grid(row=2, column=2)
        self.melody_btn_3.grid(row=3, column=2)
        self.melody_btn_4.grid(row=4, column=2)

        self.vocal_btn_1.grid(row=1, column=3)
        self.vocal_btn_2.grid(row=2, column=3)
        self.vocal_btn_3.grid(row=3, column=3)
        self.vocal_btn_4.grid(row=4, column=3)

        self.drum_vol.grid(row=0, column=0)
        self.bass_vol.grid(row=0, column=1)
        self.melody_vol.grid(row=0, column=2)
        self.vocal_vol.grid(row=0, column=3)