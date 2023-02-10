# -*- coding: utf-8 -*-
"""
Thesis: Determining high order scale factor non-linearities with sled testing.

Author: Sean Abrahamson

This code houses all the functions needed for the Thesis Main code.
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import plotly.express as px
import numpy as np


def importEGIData(Headers):
    
    # Dialog Box to Select Data to import
    
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    filename = filedialog.askopenfilename(parent=root)
    root.destroy()
    
    if filename == '':
        print('No file selected')
    else: 
        D = pd.read_csv(filename , names = Headers) # Pull only first row from Excel File

    return D


def plotSimple(df, x = None, y = None):
    
    if x == None and y == None:
        fig = px.line(df)
        fig.show()
    elif y == None:
        fig = px.line(df, x = x)
        fig.show()
    else:
        fig = px.line(df, x = x, y = y)
        fig.show()    
    
    return

def lpf(x, omega_c, T):
    """Implement a first-order low-pass filter.
    
    The input data is x, the filter's cutoff frequency is omega_c 
    [rad/s] and the sample time is T [s].  The output is y.
    """
    N = np.size(x)
    y = x
    alpha = (2-T*omega_c)/(2+T*omega_c)
    beta = T*omega_c/(2+T*omega_c)
    for k in range(1, N):
        y[k] = alpha*y[k-1] + beta*(x[k]+x[k-1])
    return y

def aveDeriv(x,y,dt):
    
    
    
    return [xp,yp]
    
    
    
