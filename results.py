#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:46:04 2023

@author: seanabrahamson
"""

from Thesis_Main import *
# from Thesis_Utils import *

#%% Initial Configuration Parameters
 
#Coefficients
g = 9.791807  

#Generate New Trajectory and RPV
generateNewTrajectory = False

#Generate New RPV (with configuration parameters)
sigmaRPV = 5E-1     # Meters??
tauRPV =  0         # Time Lag Error (seconds)
biasRPV = 0         # Bias error in RPV (meters)

generateNewRPV = False

# Used to play around with coefficients
changeDefaultCoeff = False
CoeffDict = {'K_2': 5E-6}

# Used to determine how many coefficients to calculate
N_model_start = 0     #  0 =  K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 
N_model_end = 5      #  0 = K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 

N_model = [0,0]
# # Fix indexing numbers
# N_model[0] = N_model_start
N_model[1]= N_model_end + 1





#%% Generate or import trajectory
"""
Generates or creates reference trajectory from EGI data. 
"""

if generateNewTrajectory == True:      
    generateReferenceTrajectory()
    
# Import Reference Trajectory

referenceTrajectory = pd.read_pickle("./referenceTrajectory.pkl")


#%%
# Generate track reference position vectory

if generateNewRPV == True:    
    generateTrackRPV(referenceTrajectory, sigmaRPV, tauRPV, biasRPV)

trackRPV = pd.read_pickle(f"./trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}.pkl") 


#%% Generate Simulated Accelerometer

sensorSim, AccelObj = AccelSim(referenceTrajectory, N_model, changeDefaultCoeff, CoeffDict, g)

coefficientDF, Error = RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g)

results_list = [Error, AccelObj, sensorSim, coefficientDF]

Results = {}

Results[f"Coeff: {N_model[0]}-{N_model[1]-1}"] = results_list

for n in range(N_model[1]):
    
    N_model[0] = n
    N_model[1] = n+1
    
    sensorSim, AccelObj = AccelSim(referenceTrajectory, N_model, changeDefaultCoeff, CoeffDict, g)

    coefficientDF, Error = RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g)

    results_list = [Error, AccelObj, sensorSim, coefficientDF]

    Results[f"Coeff: {N_model[0]}-{N_model[1]-1}"] = results_list
 
#%% Results Invesigation

for key in Results:
    
    print(key)
    print(Results[key][3])
    print('\n')
    

#%% Plots scripts 
"""
PLOTS

"""

# Choose which results you want to look at:
N_model[0] = 0
N_model[1] = 5
    
Error = Results(f"Coeff: {N_model[0]}-{N_model[1]-1}")[0]
sensorSim = Results(f"Coeff: {N_model[0]}-{N_model[1]-1}")[2]

    
#%%
Plots = False

if Plots == True: 
    
    #%% Plot reference Trajectory Results
    refTrajectory_fig = PlotlyPlot()
    
    refTrajectory_fig.setTitle('EGI Acceleration and Velocity and Integrated Velocity')
    refTrajectory_fig.setYaxisTitle('Acceleration (m/s/s)')
    refTrajectory_fig.setYaxis2Title('Velocity (m/s)')
    refTrajectory_fig.setXaxisTitle('GPS Time (s)')
    refTrajectory_fig.settwoAxisChoice([False, True, True])
    refTrajectory_fig.plotTwoAxis(referenceTrajectory[['refAccel_x', 'refVel_x', 'refEGIVel_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    refTrajectory_fig.show()

    refTrajectory_fig2 = PlotlyPlot()
    
    refTrajectory_fig2.setTitle('EGI displacement from EGI velocity and from Integrated acceleration')
    refTrajectory_fig2.setYaxisTitle('Velocity (m/s)')
    refTrajectory_fig2.setYaxis2Title('Distance (m)')
    refTrajectory_fig2.settwoAxisChoice([False, False, True, True])
    refTrajectory_fig2.plotTwoAxis(referenceTrajectory[['IntVel_x', 'EGIVel_x', 'IntDist_x', 'EGIDist_x']], df_x = referenceTrajectory[['Time']])
    refTrajectory_fig2.show()
    
    #%% Plot integration for sensor simulation results vs Truth
    sensorSimVTruth_fig = PlotlyPlot()
    
    sensorSimVTruth_fig.setTitle('Sensor Accelerometer sim and integrated velocity vs Truth')
    sensorSimVTruth_fig.setYaxisTitle('Velocity (m/s)')
    sensorSimVTruth_fig.setYaxis2Title('Distance (m)')
    sensorSimVTruth_fig.settwoAxisChoice([False, True])
    sensorSimVTruth_fig.plotTwoAxis(referenceTrajectory[['refAccel_x', 'refVel_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    sensorSimVTruth_fig.addScatter(sensorSim[['SensorSim_Ax']], df_x = sensorSim[['Time']],secondary_y=False)
    sensorSimVTruth_fig.addScatter(sensorSim[['SensorSim_Vx']], df_x = sensorSim[['Time']],secondary_y=True)
    sensorSimVTruth_fig.show()

    sensorSimVTruth_fig2 = PlotlyPlot()
    
    sensorSimVTruth_fig2.setTitle('Sensor integrated velocity sim and distance velocity vs referenceTrajectory')
    sensorSimVTruth_fig2.setYaxisTitle('Velocity (m/s)')
    sensorSimVTruth_fig2.setYaxis2Title('Distance (m)')
    sensorSimVTruth_fig2.settwoAxisChoice([False, True])
    sensorSimVTruth_fig2.plotTwoAxis(referenceTrajectory[['refVel_x','refDist_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    sensorSimVTruth_fig2.addScatter(sensorSim[['SensorSim_Vx']], df_x = sensorSim[['Time']],secondary_y=False)
    sensorSimVTruth_fig2.addScatter(sensorSim[['SensorSim_Dx']], df_x = sensorSim[['Time']],secondary_y=True)
    sensorSimVTruth_fig2.show()
    
    #%% Plot Track RPV
    RPV_PlotvsTraj = PlotlyPlot()
    
    RPV_PlotvsTraj.setTitle('Track Rerence Position Vector vs Reference Trajectory')
    RPV_PlotvsTraj.setYaxisTitle('Distance (m)')
    RPV_PlotvsTraj.settwoAxisChoice([False, True])
    RPV_PlotvsTraj.plotTwoAxis(referenceTrajectory[['refDist_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    RPV_PlotvsTraj.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']])
    RPV_PlotvsTraj.show()   
    
    
    #%% Plot Coordinate Functions
    coordinatFunc_fig = PlotlyPlot()
    
    coordinatFunc_fig.setTitle('Coordinate Functions vs Error')
    coordinatFunc_fig.setYaxisTitle('Distance (m)')
    coordinatFunc_fig.setYaxis2Title('Accleration (m/s/s)')
    coordinatFunc_fig.settwoAxisChoice([False, True])
    coordinatFunc_fig.plotTwoAxis(Error[['VelErr_x']], df_x = Dist_Error[['Time']], mode = 'markers')
    coordinatFunc_fig.addScatter(AccelOne.AccelModelCoef['K_2'], df_x = referenceTrajectory[['Time']], secondary_y = False)
    # coordinatFunc_fig.addScatter(referenceTrajectory[['refAccel_x']], df_x = referenceTrajectory[['Time']], secondary_y = True)
    
    coordinatFunc_fig.show()


    #%% Plot Velcocity and Distance Errors
    distVelError_fig = PlotlyPlot()
    
    distVelError_fig.setTitle('Distance and Velocity Error')
    distVelError_fig.setYaxisTitle('Distance (m)')
    distVelError_fig.setYaxis2Title('Velocity (m/s)')
    distVelError_fig.settwoAxisChoice([False, True])
    distVelError_fig.plotTwoAxis(Dist_Error[['DistErr_x']], df_x = Dist_Error[['Time']], mode = 'markers')
    # distVelError_fig.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']], secondary_y = False)
    # distVelError_fig.addScatter(trackRPV[['SensorInterpDist']], df_x = trackRPV[['Time']], secondary_y = False)
    distVelError_fig.addScatter(Error[['VelErr_x']], df_x = Error[['Time']], secondary_y = True)
    # distVelError_fig.addScatter(sensorSim[['SensorSim_Dx']], df_x = sensorSim[['Time']])
    
    distVelError_fig.show()

#%% Plots Residuals Acceleration and Velocity
    
    ResidualsVsVel_fig = PlotlyPlot()
    
    ResidualsVsVel_fig.setTitle('Residuals')
    ResidualsVsVel_fig.setYaxisTitle('Velocity Error Residuals (m/s)))')
    ResidualsVsVel_fig.setYaxis2Title('Velocity (m/s)')
    ResidualsVsVel_fig.setXaxisTitle('GPS Time (s)')
    ResidualsVsVel_fig.settwoAxisChoice([False, True])
    ResidualsVsVel_fig.plotTwoAxis(Error[['Ve_x_Resid']], df_x = Error[['Time']])
    ResidualsVsVel_fig.addLine(Error[['SensorSim_Vx']], df_x = Error[['Time']],secondary_y=True)
    ResidualsVsVel_fig.show()
    
    ResidualsVsAccel_fig = PlotlyPlot()
    
    ResidualsVsAccel_fig.setTitle('Residuals')
    ResidualsVsAccel_fig.setYaxisTitle('Velocity Error Residuals (m/s)')
    ResidualsVsAccel_fig.setYaxis2Title('Acceleration (m/s/s)')
    ResidualsVsAccel_fig.setXaxisTitle('GPS Time (s)')
    ResidualsVsAccel_fig.settwoAxisChoice([False, True])
    ResidualsVsAccel_fig.plotTwoAxis(Error[['Ve_x_Resid']], df_x = Error[['Time']])
    ResidualsVsAccel_fig.addLine(Error[['SensorSim_Ax']], df_x = Error[['Time']],secondary_y=True)
    ResidualsVsAccel_fig.show()
        
#%% Plot Velocity Error versus velocity error model
    VelErrVsResid = PlotlyPlot()
    
    VelErrVsResid.setTitle('Velocity Error vs Estimated Error Model')
    VelErrVsResid.setYaxisTitle('Velocity Error (m/s)')
    VelErrVsResid.setXaxisTitle('GPS Time (s)')
    VelErrVsResid.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']], mode = 'markers')
    VelErrVsResid.addScatter(Error[['V_error_model']], df_x = Error[['Time']], secondary_y=False)
    # VelErrVsResid.addScatter(Error[['A_Bias']], df_x = Error[['Time']], secondary_y=False)
    VelErrVsResid.show()