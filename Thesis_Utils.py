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
from scipy import integrate


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


def generateReferenceTrajectory():
    
    EGI_accel = importEGIData(['Time', 'Ax','Ay','Az'])
    EGI_vel = importEGIData(['Time', 'Vx','Vy', 'Vz'])

    # EGI_accel =  pd.read_csv('EGI_accel.csv',names = ['Time', 'Ax','Ay','Az'])
    # EGI_vel = pd.read_csv('EGI_vel.csv',names = ['Time', 'Vx','Vy', 'Vz'])

    EGI_accel_vel = EGI_accel.join(EGI_vel[['Vx','Vy','Vz']])

    #%% Truth Gen Step 2 - Trim data to focus on actual sled run.
    
    print("Trimming data to start/stop time determined visually...")
    startTime = 399600
    stopTime = 399700

    EGI_accel_vel_trim = EGI_accel_vel[(EGI_accel_vel['Time'] > startTime) & (EGI_accel_vel['Time'] < stopTime) ] # trim accelerometer output

    # Creating new time array for data
    Tdur = EGI_accel_vel_trim['Time'].max() - EGI_accel_vel_trim['Time'].min()
    Tlen = len(EGI_accel_vel_trim['Time'])

    NewTimeSeries = np.linspace(0, Tdur, Tlen)

    EGI_accel_vel_trim.loc[:,'New Time'] = NewTimeSeries
                                                                         
    #%% Truth Gen Step 3 - Smooth Acceleration in X-axis
    # EGI_accel_smoothed_array = savgol_filter(EGI_accel_vel_trim['Ax'],25,3)

    EGI_accel_presmoothed = EGI_accel_vel_trim[['Ax']]

    EGI_accel_smoothed_array = lpf(EGI_accel_vel_trim[['Ax']].to_numpy(),50,Tdur/Tlen)

    EGI_accel_vel_trim['Ax'] = EGI_accel_presmoothed

    # EGI_accel_vel_trim['Ax_smooth'] = pd.Series(EGI_accel_smoothed_array)

    #%% Truth Gen Step 4 - Create a DataFrame to house all truth data

    referenceTrajectory = pd.DataFrame()
    
    print(EGI_accel_smoothed_array)
    
    referenceTrajectory['Time'] = EGI_accel_vel_trim['New Time']
    referenceTrajectory['refAccel_x'] = EGI_accel_smoothed_array
    referenceTrajectory['refEGIVel_x'] = EGI_accel_vel_trim['Vx']

    # Create New Time Series
    referenceTrajectory['Time'] = np.linspace(0, Tdur, Tlen)

    # Change initial acceleration in X to zero until launch. Determined visually
    print("Setting initial acceleration to 0 until launch...")
    referenceTrajectory['refAccel_x'][:1145] = 0

    # Change final acceleration after stop to zero. Determined visually
    print("Setting final acceleration at 0...")
    referenceTrajectory['refAccel_x'][4992:] = 0
    
    
    #%% Truth Gen Step 5 -  Integrate truth acceleration to get velocity and distance
    referenceTrajectory['refVel_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refAccel_x'],x = referenceTrajectory['Time'],initial = 0) 
    
    # Change final Velocity after stop to zero. Determined visually
    print("Setting final velocity at 0...")
    referenceTrajectory['refVel_x'][4992:] = 0
    
    referenceTrajectory['refDist_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refVel_x'],x = referenceTrajectory['Time'],initial = 0) 


    # Integrate EGI velocity to compare to double integrated acceleration
    referenceTrajectory['refEGIDist_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refEGIVel_x'],x = referenceTrajectory['Time'],initial = 0) 
    
    #%% Save trajectory to Pickle File
    
    referenceTrajectory.to_pickle("./referenceTrajectory.pkl")
    
    return 

#%% Generate Track RPV Function 

def generateTrackRPV(referenceTrajectory):
        
    trackRPV = pd.DataFrame()
    
    Interupter_delta = 4.5 * 0.3048 # ft converted to meters
    TrackLength = 10000   # Meters
    
    trackRPV['Interupters_DwnTrk_dist'] = np.arange(0, TrackLength, Interupter_delta)
    
    trackRPV['Time'] = np.interp(trackRPV['Interupters_DwnTrk_dist'],referenceTrajectory['refDist_x'],referenceTrajectory['Time'])
    

    
    trackRPV = trackRPV[trackRPV['Interupters_DwnTrk_dist'] <= referenceTrajectory['refDist_x'].max()]
    
    trackRPV = trackRPV.drop_duplicates(subset=['Time'])
    
    trackRPV = trackRPV[:-1]
    
    
    trackRPV_zeroVel= pd.DataFrame()
    trackRPV_zeroVel_start = pd.DataFrame() 
    trackRPV_zeroVel_end = pd.DataFrame()
    
    trackRPV_zeroVel_start['Time'] = referenceTrajectory['Time'][referenceTrajectory['Time']<trackRPV['Time'].min()]
    trackRPV_zeroVel_start['Interupters_DwnTrk_dist'] = 0
    
    trackRPV_zeroVel_end['Time'] = referenceTrajectory['Time'][referenceTrajectory['Time']>trackRPV['Time'].max()]
    trackRPV_zeroVel_end['Interupters_DwnTrk_dist'] = trackRPV['Interupters_DwnTrk_dist'].max()
    
    trackRPV_zeroVel = pd.concat((trackRPV_zeroVel_start,trackRPV_zeroVel_end), axis = 0)
    
    trackRPV = pd.concat((trackRPV, trackRPV_zeroVel), axis = 0)
    
    trackRPV = trackRPV.sort_values(by='Time').reset_index(drop=True)
    
    #%% Save track RPV to pickle file
    trackRPV.to_pickle("./trackRPV.pkl")
        
    return


    
    
    
