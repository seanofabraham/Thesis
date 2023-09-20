#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 10:22:10 2023

@author: seanabrahamson
"""

from Thesis_Main import *
import os.path
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'browser'

# from Thesis_Utils import *

#%% Initial Configuration Parameters
 
#Coefficients
g = 9.791807  

#Generate New Trajectory and RPV
generateNewTrajectory = False

#Generate New RPV (with configuration parameters)
generateNewRPVs = False

sigmaRPV = 0.006       # Meters (.006 is about a quarter of an inch)
tauRPV =  0            # Time Lag Error (seconds)
biasRPV = 0            # Bias error in RPV (meters)

MCn = 500


#%% Generate track reference position vectory

referenceTrajectory = pd.read_pickle("./referenceTrajectory.pkl")

if generateNewRPVs == True:
    for i in range(MCn):
        generateTrackRPV(referenceTrajectory, sigmaRPV, tauRPV, biasRPV, Overwrite=False)
            

#%% Import Track RPVs

# Create DF with all trackRPVs
trueRPV = pd.read_pickle(f"./RPVs/trackRPV_sig0_tau0_bias0.pkl")
RPVsErrDistLong = pd.DataFrame()
RPVsErrVelLong = pd.DataFrame()
RPVerror = pd.DataFrame()
RPVlist = []
RPVsDistErrList = []
RPVsVelErrList = []

#%%
for i in range(MCn):
    
    RPVsErrDist = pd.DataFrame()
    RPVsErrVel = pd.DataFrame()
    
    RPV = pd.read_pickle(f"./VarianceRPVs/trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}_{i}.pkl")
    
    RPVsErrDist['Time'] = RPV['Time']
    RPVsErrDist['Interupters_DwnTrk_dist'] = trueRPV['Interupters_DwnTrk_dist']
    RPVsErrDist['Dist_er'] = RPV['Interupters_DwnTrk_dist'] - trueRPV['Interupters_DwnTrk_dist']
    
    RPV_Derr = RPVsErrDist['Dist_er'].to_numpy()
    RPV_Ve = np.diff(RPV_Derr)/np.diff(trueRPV['Time'])
    RPV_Ve_t = (RPV['Time'].head(-1) + np.diff(RPV['Time'])/2).to_numpy() # UPDATE TIME TAG FOR DIFFERENTIATION.
    RPV_DwnTrkDist = (trueRPV['Interupters_DwnTrk_dist'].head(-1) + np.diff(trueRPV['Interupters_DwnTrk_dist'])/2).to_numpy() # UPDATE TIME TAG FOR
    
    RPVsErrVel['Time'] = RPV_Ve_t
    RPVsErrVel['Interupters_DwnTrk_dist'] = RPV_DwnTrkDist
    RPVsErrVel['VelErr_x'] = RPV_Ve
    
    RPVsErrDistLong = RPVsErrDistLong.append(RPVsErrDist, ignore_index=True)
    RPVsErrVelLong = RPVsErrVelLong.append(RPVsErrVel, ignore_index=True)
    RPVlist.append(RPV)
    RPVsDistErrList.append(RPVsErrDist)
    RPVsVelErrList.append(RPVsErrVel)
        
    del RPVsErrDist
    del RPVsErrVel


#%%    
# Create DF with all trackRPVs
RPVVelErr = RPVsVelErrList[0]

RPVsDF = pd.concat([trueRPV] + [RPV['Interupters_DwnTrk_dist'] for RPV in RPVlist], axis=1)  
RPVDistErr = pd.concat([trueRPV] + [RPV['Dist_er'] for RPV in RPVsDistErrList], axis=1)  
RPVVelErr = pd.concat([RPVVelErr] + [RPV['VelErr_x'] for RPV in RPVsVelErrList], axis=1) 

RPVVelErr['Delta_t'] = np.diff(RPVDistErr['Time'])
RPVVelErr['Sigma'] = np.sqrt(2)*sigmaRPV/RPVVelErr['Delta_t']

#%% Generate Results

# Calculate the standard deviation for each column
DistErr_std = RPVDistErr.iloc[:, 2:].std(axis=1)
# Add the standard deviation column to the dataframe
RPVDistErr['Std'] = DistErr_std


# Calculate the standard deviation for each column
VelErr_std = RPVVelErr.iloc[:, 2:].std(axis=1)
# Add the standard deviation column to the dataframe
RPVVelErr['Std'] = VelErr_std


#%%
Plots = False

if Plots == True: 

    #%% Density Contour Plot Distance
    DensityPlotDist = go.Figure(go.Histogram2dContour(
        x = RPVsErrDistLong['Interupters_DwnTrk_dist'],
        y = RPVsErrDistLong['Dist_er']
        ))
    
    # DensityPlot = px.density_contour(RPVsLong, x='Time', y='Dist_er', marginal_y='histogram')
    DensityPlotDist.update_traces(contours_coloring='fill', contours_showlines=False, colorscale = 'blues')
    DensityPlotDist.add_hline(y=sigmaRPV)
    DensityPlotDist.add_hline(y=-sigmaRPV)
    
    DensityPlotDist.add_trace(go.Scatter(x = trueRPV['Interupters_DwnTrk_dist'],
    y = RPVDistErr['Std']))
    
    DensityPlotDist.update_layout(xaxis_title="Down Track Distance", yaxis_title="Distance Error (m)")
    DensityPlotDist.show()
    
    #%% Density Contour Plot Velocity
    DensityPlotVel1= go.Figure(go.Histogram2dContour(
        x = RPVsErrVelLong['Time'],
        y = RPVsErrVelLong['VelErr_x']
        ))
    
    # DensityPlot = px.density_contour(RPVsLong, x='Time', y='Dist_er', marginal_y='histogram')
    DensityPlotVel1.update_traces(contours_coloring='fill',contours_showlines=False, colorscale = 'blues')
    DensityPlotVel1.update_layout(xaxis_title="Time", yaxis_title="Velocity Error (m/s)")

        
    DensityPlotVel1.add_trace(go.Scatter(x = RPVsErrVelLong['Time'], y = RPVVelErr['Std']))
    
    DensityPlotVel1.add_trace(go.Scatter(x = RPVsErrVelLong['Time'], y = RPVVelErr['Sigma']))


    DensityPlotVel1.show()
    
    #%%
    
    DensityPlotVel2= go.Figure(go.Histogram2dContour(
        x = RPVsErrVelLong['Interupters_DwnTrk_dist'],
        y = RPVsErrVelLong['VelErr_x']
        ))
    
    # DensityPlot = px.density_contour(RPVsLong, x='Time', y='Dist_er', marginal_y='histogram')
    DensityPlotVel2.update_traces(contours_coloring='fill',contours_showlines=False, colorscale = 'blues')
    DensityPlotVel2.update_layout(xaxis_title="Down Track Distance", yaxis_title="Velocity Error (m/s)")

    DensityPlotVel2.show()
    
    
    
    

    