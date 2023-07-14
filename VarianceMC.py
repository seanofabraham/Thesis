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

sigmaRPV = 0.006           # Meters (.006 is about a quarter of an inch)
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
RPVsErrDist = pd.DataFrame()
RPVsErrVel = pd.DataFrame()
RPVsErrDistLong = pd.DataFrame()
RPVsErrVelLong = pd.DataFrame()
RPVerror = pd.DataFrame()
RPVlist = []

#%%
for i in range(MCn):
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
    RPVsDistErrList.append(RPVsErrVel)
    RPVsVelErrList.append()
    
# Create DF with all trackRPVs
RPVs = pd.read_pickle(f"./RPVs/trackRPV_sig0_tau0_bias0.pkl")
RPVsDF = pd.concat([RPVs] + [RPV['Interupters_DwnTrk_dist'] for RPV in RPVlist], axis=1)  
    

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
    DensityPlotDist.show()
    
    #%% Density Contour Plot Velocity
    DensityPlotVel1= go.Figure(go.Histogram2dContour(
        x = RPVsErrVelLong['Time'],
        y = RPVsErrVelLong['VelErr_x']
        ))
    
    # DensityPlot = px.density_contour(RPVsLong, x='Time', y='Dist_er', marginal_y='histogram')
    DensityPlotVel1.update_traces(contours_coloring='fill',contours_showlines=False, colorscale = 'blues')

    DensityPlotVel1.show()
    
    
    DensityPlotVel2= go.Figure(go.Histogram2dContour(
        x = RPVsErrVelLong['Interupters_DwnTrk_dist'],
        y = RPVsErrVelLong['VelErr_x']
        ))
    
    # DensityPlot = px.density_contour(RPVsLong, x='Time', y='Dist_er', marginal_y='histogram')
    DensityPlotVel2.update_traces(contours_coloring='fill',contours_showlines=False, colorscale = 'blues')

    DensityPlotVel2.show()
    
    
    
    
    #%% Uncertainty in RPV plots
    RPV_UncertPlot1 = PlotlyPlot()
    
    RPV_UncertPlot1.setTitle('Reference Position Vector Error')
    RPV_UncertPlot1.setYaxisTitle('Distance (m)')
    RPV_UncertPlot1.setYaxis2Title('Distance (m)')
    RPV_UncertPlot1.setXaxisTitle('Time (s)')
    RPV_UncertPlot1.settwoAxisChoice([False, True])
    RPV_UncertPlot1.plotTwoAxis(UCert[['DistErr_x']], df_x = UCert[['Time']], Name = 'Distance Error', Mode = 'markers')
    RPV_UncertPlot1.addScatter(referenceTrajectory[['refDist_x']], df_x = referenceTrajectory[['Time']], Mode = 'markers', Name = 'Ref Trajectory Velocity',secondary_y = True)
     
    RPV_UncertPlot1.update_template()

    RPV_UncertPlot1.show()
    # RPV_PlotvsTraj1.write_image('ReferencePositionVector1',saveFigPath)
     
    
    RPV_UncertPlot2 = PlotlyPlot()
    
    RPV_UncertPlot2.setTitle('Reference Position Vector Error')
    RPV_UncertPlot2.setYaxisTitle('Velocity Error (m/s)')
    RPV_UncertPlot2.setYaxis2Title('Velocity (m/s)')
    RPV_UncertPlot2.setXaxisTitle('Time (s)')
    RPV_UncertPlot2.settwoAxisChoice([False, True])
    RPV_UncertPlot2.plotTwoAxis(UCert[['VelErr_x']], df_x = UCert[['Time']], Name = 'Velocity Error', Mode = 'markers')
    RPV_UncertPlot2.addScatter(referenceTrajectory[['refVel_x']], df_x = referenceTrajectory[['Time']], Mode = 'markers', Name = 'Ref Trajectory Velocity',secondary_y = True)
   
    
    RPV_UncertPlot2.update_template()

    RPV_UncertPlot2.show()
    # RPV_PlotvsTraj1.write_image('ReferencePositionVector1',saveFigPath)
    