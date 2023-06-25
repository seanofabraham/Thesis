#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:46:04 2023

@author: seanabrahamson
"""

from Thesis_Main import *
import os.path
import pandas as pd

# from Thesis_Utils import *

#%% Initial Configuration Parameters
 
#Coefficients
g = 9.791807  

#Generate New Trajectory and RPV
generateNewTrajectory = False

#Generate New RPV (with configuration parameters)
generateNewRPV = True

sigmaRPV = 0.006        # Meters (.006 is about a quarter of an inch)
tauRPV =  0            # Time Lag Error (seconds)
biasRPV = 0            # Bias error in RPV (meters)


# Used to play around with coefficients
changeDefaultCoeff = True
CoeffDict = {'K_1': 5E-6}

# Used to determine how many coefficients to calculate

N_model_start = 0     #  0 =  K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 
N_model_end = 5      #  0 = K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 

# Definition of which coefficients will be computed based on above selections. 
ModelDict = {'0': 'K_1',
             '1': 'K_0',
             '2': 'K_2',
             '3': 'K_3',
             '4': 'K_4',
             '5': 'K_5'}


N_model = [0,0]
# # Fix indexing numbers
# N_model[0] = N_model_start
N_model[1]= N_model_end + 1


# Perform only full model as defined above or look at each individual coefficient.
individualCoeffAnalysis = True

             

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

if generateNewRPV == False:   
    generateNewRPV = not os.path.isfile(f"./RPVs/trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}.pkl")

if generateNewRPV == True:    
    generateTrackRPV(referenceTrajectory, sigmaRPV, tauRPV, biasRPV)

trackRPV = pd.read_pickle(f"./RPVs/trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}.pkl")

#%%
# Analyze effects of measurement uncertainty 
trackPureRPV = pd.read_pickle(f"./RPVs/trackRPV_sig0_tau0_bias0.pkl")

UCert = pd.DataFrame()

RPV_Derr = trackRPV['Interupters_DwnTrk_dist'] - trackPureRPV['Interupters_DwnTrk_dist']

RPV_Ve = np.diff(RPV_Derr)/np.diff(trackPureRPV['Time'])
RPV_Ve_t = (trackRPV['Time'].head(-1) + np.diff(trackRPV['Time'])/2).to_numpy() # UPDATE TIME TAG FOR DIFFERENTIATION.

UCert['Time'] = RPV_Ve_t
UCert['DistErr_x'] = np.interp(RPV_Ve_t,trackRPV['Time'],RPV_Derr) 
UCert['VelErr_x'] = RPV_Ve


#%% Generate Simulated Accelerometer for full model

sensorSim, AccelObj = AccelSim(referenceTrajectory, N_model, changeDefaultCoeff, CoeffDict, g)

#%% Perform Regression Analysis for full model

coefficientDF, Error, cov_A = RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g)

results_list = [Error, AccelObj, sensorSim, coefficientDF, cov_A]

Results = {}

Results[f"Coeff: {ModelDict[str(N_model[0])]}-{ModelDict[str(N_model[1]-1)]}"] = results_list


#%% perform Regression Analysis for individual coefficients

if individualCoeffAnalysis == True:
    for n in range(N_model[1]):
        
        N_model[0] = n
        N_model[1] = n+1
        
        sensorSim, AccelObj = AccelSim(referenceTrajectory, N_model, changeDefaultCoeff, CoeffDict, g)
    
        coefficientDF, Error, cov_A = RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g)
    
        results_list = [Error, AccelObj, sensorSim, coefficientDF, cov_A]
    
        Results[f"Coeff: {ModelDict[str(N_model[0])]}-{ModelDict[str(N_model[1]-1)]}"] = results_list
 
#%% Results Invesigation

for key in Results:
    
    print(key)
    print(Results[key][3])
    print('\n')
   
    
print(Results['Coeff: K_1-K_5'][4])    

df = pd.DataFrame(Results['Coeff: K_1-K_5'][4]).T
# df.to_excel(excel_writer = "/Users/seanabrahamson/Library/CloudStorage/Box-Box/EE_Masters/Thesis/Results.xlsx")


#%% Plots scripts 
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PLOTS For THESIS



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
#Choose a Path to save figures to
# saveFigPath = '/Users/seanabrahamson/Box/EE_Masters/Thesis/Thesis_Figures' # Imac path
saveFigPath = '/Users/seanabrahamson/Library/CloudStorage/Box-Box/EE_Masters/Thesis/Thesis_Figures' #MacBook Pro Path

# Choose which results you want to look at:
N_model[0] = 0
N_model[1] = 5
    
Error = Results[f"Coeff: {ModelDict[str(N_model[0])]}-{ModelDict[str(N_model[1])]}"][0]
sensorSim = Results[f"Coeff: {ModelDict[str(N_model[0])]}-{ModelDict[str(N_model[1])]}"][2]

    
#%%
Plots = False

if Plots == True: 
    
    #%% PLOTS for Thesis
    #%% Reference Trajectory
    refTrajectory_fig = PlotlyPlot()
    
    # refTrajectory_fig.setTitle('Reference Trajectory')
    refTrajectory_fig.setYaxisTitle('Acceleration (m/s<sup>2</sup>)')
    refTrajectory_fig.setYaxis2Title('Velocity (m/s)')
    refTrajectory_fig.setXaxisTitle('Time (s)')
    refTrajectory_fig.settwoAxisChoice([False])
    refTrajectory_fig.plotTwoAxis(referenceTrajectory[['refAccel_x']], df_x = referenceTrajectory[['Time']], Name = '$\\text{Acceleration}$')
    refTrajectory_fig.addLine(referenceTrajectory[['refVel_x']], df_x = referenceTrajectory[['Time']], Name = '$\\text{Velocity}$', secondary_y=True, )
    refTrajectory_fig.legendTopRight()    
    refTrajectory_fig.update_template()
    # refTrajectory_fig.update_legend()
    refTrajectory_fig.show()

    
    refTrajectory_fig.write_image('ReferenceTrajectory',saveFigPath)
    
    #%% Reference Position Vector    
    
    RPV_PlotvsTraj1 = PlotlyPlot()
    
    RPV_PlotvsTraj1.setTitle('Reference Position Vector')
    RPV_PlotvsTraj1.setYaxisTitle('Distance (m)')
    RPV_PlotvsTraj1.setXaxisTitle('Time (s)')
    RPV_PlotvsTraj1.settwoAxisChoice([False, True])
    RPV_PlotvsTraj1.plotTwoAxis(referenceTrajectory[['refDist_x']], df_x = referenceTrajectory[['Time']], Name = 'Reference Trajectory')
    RPV_PlotvsTraj1.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']], Mode = 'markers', Name = 'Reference Position Vector', Opacity = .7)
    
    zoom_x = [17,20.5]
    zoom_y = [-40, 60]
    RPV_PlotvsTraj1.addShadedBox(zoom_x, zoom_y, scale_factor_y = 1.8)
       
    RPV_PlotvsTraj1.update_template()

    RPV_PlotvsTraj1.show()
    RPV_PlotvsTraj1.write_image('ReferencePositionVector1',saveFigPath)
    
    
    RPV_PlotvsTraj2 = PlotlyPlot()
    
    RPV_PlotvsTraj2.setTitle('Reference Position Vector')
    RPV_PlotvsTraj2.setYaxisTitle('Distance (m)')
    RPV_PlotvsTraj2.setXaxisTitle('Time (s)')
    RPV_PlotvsTraj2.settwoAxisChoice([False, True])
    RPV_PlotvsTraj2.plotTwoAxis(referenceTrajectory[['refDist_x']], df_x = referenceTrajectory[['Time']], Name = 'Reference Trajectory')
    RPV_PlotvsTraj2.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']], Mode = 'markers', Name = 'Reference Position Vector', Opacity = .8)
    
    RPV_PlotvsTraj2.zoom(zoom_x, zoom_y)
       
    RPV_PlotvsTraj2.update_template()
    RPV_PlotvsTraj2.show()
    RPV_PlotvsTraj2.write_image('ReferencePositionVector2',saveFigPath)

    #%% Error Contributions
    #%% Plot Distance Error as Caused by individual Error Coefficients
    
    zoom_x = [18,77]
    zoom_y = [-.05 ,.15]
    
    DistErrorCoeffs_fig = PlotlyPlot()
    
    DistErrorCoeffs_fig.setTitle('Distance Errors')
    DistErrorCoeffs_fig.setXaxisTitle('Time (s)')
    DistErrorCoeffs_fig.setYaxisTitle('Distance (m)')
    DistErrorCoeffs_fig.setYaxis2Title('Distance (m)')
    DistErrorCoeffs_fig.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            DistErrorCoeffs_fig.plotTwoAxis(-Error[['DistErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers', Opacity = .7, Size = 4)
            init = False
        else:
            DistErrorCoeffs_fig.addScatter(-Error[['DistErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key[:-4], Opacity = .7, Size = 4)
    
    DistErrorCoeffs_fig.addShadedBox(zoom_x, zoom_y,scale_factor_y=1.7)
    
    DistErrorCoeffs_fig.update_template()    
    DistErrorCoeffs_fig.show()
    DistErrorCoeffs_fig.write_image('DistanceErrorAllCoefficients',saveFigPath)
        
    
    DistErrorCoeffs_figZoom = PlotlyPlot()
    
    DistErrorCoeffs_figZoom.setTitle('Distance Errors')
    DistErrorCoeffs_figZoom.setXaxisTitle('Time (s)')
    DistErrorCoeffs_figZoom.setYaxisTitle('Distance (m)')
    DistErrorCoeffs_figZoom.setYaxis2Title('Distance (m)')
    DistErrorCoeffs_figZoom.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            DistErrorCoeffs_figZoom.plotTwoAxis(-Error[['DistErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers', Opacity = .9, Size = 4)
            init = False
        else:
            DistErrorCoeffs_figZoom.addScatter(-Error[['DistErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key[:-4],Opacity = .9, Size = 4)
    
    DistErrorCoeffs_figZoom.update_template()    
    DistErrorCoeffs_figZoom.zoom(zoom_x, zoom_y)
    DistErrorCoeffs_figZoom.show()
    DistErrorCoeffs_figZoom.write_image('DistanceErrorAllCoefficientsZoom',saveFigPath)
    
    #%% Plot Velocity Error as Caused by individual Error Coefficients
        
    zoom_x = [18,77]
    zoom_y = [-.002 ,.004]
    
    VelErrorCoeffs_fig = PlotlyPlot()
     
    VelErrorCoeffs_fig.setTitle('Velocity Errors')
    VelErrorCoeffs_fig.setXaxisTitle('Time (s)')
    VelErrorCoeffs_fig.setYaxisTitle('Velocity (m/s)')
    VelErrorCoeffs_fig.setYaxis2Title('Velocity (m/s)')
    VelErrorCoeffs_fig.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            VelErrorCoeffs_fig.plotTwoAxis(-Error[['VelErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers')
            init = False
        else:
            VelErrorCoeffs_fig.addScatter(-Error[['VelErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key[:-4])
    
    VelErrorCoeffs_fig.update_template()       
    VelErrorCoeffs_fig.addShadedBox(zoom_x, zoom_y,scale_factor_y=1.7)
    VelErrorCoeffs_fig.show()
    VelErrorCoeffs_fig.write_image('VelocityErrorAllCoefficients',saveFigPath)
    
    
    VelErrorCoeffs_figZoom = PlotlyPlot()
    
    VelErrorCoeffs_figZoom.setTitle('Velocity Errors')
    VelErrorCoeffs_figZoom.setXaxisTitle('Time (s)')
    VelErrorCoeffs_figZoom.setYaxisTitle('Velocity (m/s)')
    VelErrorCoeffs_figZoom.setYaxis2Title('Velocity (m/s)')
    VelErrorCoeffs_figZoom.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            VelErrorCoeffs_figZoom.plotTwoAxis(-Error[['VelErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers')
            init = False
        else:
            VelErrorCoeffs_figZoom.addScatter(-Error[['VelErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key[:-4])
    
    VelErrorCoeffs_figZoom.update_template()     
    VelErrorCoeffs_figZoom.zoom(zoom_x, zoom_y)
    VelErrorCoeffs_figZoom.show()
    VelErrorCoeffs_figZoom.write_image('VelocityErrorAllCoefficientsZoomed',saveFigPath)


    #%% Plot Sensor Simulation vs the Reference Trajectory

    sensorSimVTruth_fig = PlotlyPlot()
    
    sensorSimVTruth_fig.setTitle('Sensor Accelerometer sim and integrated velocity vs Truth')
    sensorSimVTruth_fig.setYaxisTitle('Velocity (m/s)')
    sensorSimVTruth_fig.setYaxis2Title('Distance (m)')
    sensorSimVTruth_fig.settwoAxisChoice([False, True])
    sensorSimVTruth_fig.plotTwoAxis(referenceTrajectory[['refAccel_x', 'refVel_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    sensorSimVTruth_fig.addScatter(sensorSim[['SensorSim_Ax']], df_x = sensorSim[['Time']],secondary_y=False)
    sensorSimVTruth_fig.addScatter(sensorSim[['SensorSim_Vx']], df_x = sensorSim[['Time']],secondary_y=True)
    sensorSimVTruth_fig.update_template() 
    sensorSimVTruth_fig.show()

    sensorSimVTruth_fig2 = PlotlyPlot()
    
    sensorSimVTruth_fig2.setTitle('Sensor integrated velocity sim and distance velocity vs referenceTrajectory')
    sensorSimVTruth_fig2.setYaxisTitle('Velocity (m/s)')
    sensorSimVTruth_fig2.setYaxis2Title('Distance (m)')
    sensorSimVTruth_fig2.settwoAxisChoice([False, True])
    sensorSimVTruth_fig2.plotTwoAxis(referenceTrajectory[['refVel_x','refDist_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    sensorSimVTruth_fig2.addScatter(sensorSim[['SensorSim_Vx']], df_x = sensorSim[['Time']],secondary_y=False)
    sensorSimVTruth_fig2.addScatter(sensorSim[['SensorSim_Dx']], df_x = sensorSim[['Time']],secondary_y=True)
    sensorSimVTruth_fig2.update_template() 
    sensorSimVTruth_fig2.show()


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
    
    

    #%% Other Plots    
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Other Plots     
    Used for developing code originally.
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    #%% Plot reference Trajectory Results
    refTrajectory_fig = PlotlyPlot()
    
    refTrajectory_fig.setTitle('EGI Acceleration and Velocity and Integrated Velocity')
    refTrajectory_fig.setYaxisTitle('Acceleration (m/s/s)')
    refTrajectory_fig.setYaxis2Title('Velocity (m/s)')
    refTrajectory_fig.setXaxisTitle('GPS Time (s)')
    refTrajectory_fig.settwoAxisChoice([False, True, True])
    refTrajectory_fig.plotTwoAxis(referenceTrajectory[['refAccel_x', 'refVel_x', 'refEGIVel_x']], df_x = referenceTrajectory[['Time']])
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
    
    
    if sigmaRPV != 0:
        plotSimple(np.diff(trackRPV['Interupters_DwnTrk_dist']))
    
    
    #%% Plot Coordinate Functions
    coordinatFunc_fig = PlotlyPlot()
    
    coordinatFunc_fig.setTitle('Coordinate Functions vs Error')
    coordinatFunc_fig.setYaxisTitle('Distance (m)')
    coordinatFunc_fig.setYaxis2Title('Accleration (m/s/s)')
    coordinatFunc_fig.settwoAxisChoice([False, True])
    coordinatFunc_fig.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']], mode = 'markers')
    coordinatFunc_fig.addScatter(AccelOne.AccelModelCoef['K_2'], df_x = referenceTrajectory[['Time']], secondary_y = False)
    # coordinatFunc_fig.addScatter(referenceTrajectory[['refAccel_x']], df_x = referenceTrajectory[['Time']], secondary_y = True)
    coordinatFunc_fig.show()



    #%% Plot Velocity Error as Caused by individual Error Coefficients
    
    DistErrorCoeffs_fig = PlotlyPlot()
    
    DistErrorCoeffs_fig.setTitle('Distance Errors')
    DistErrorCoeffs_fig.setYaxisTitle('Distance (m)')
    DistErrorCoeffs_fig.setYaxis2Title('Distance (m)')
    DistErrorCoeffs_fig.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            DistErrorCoeffs_fig.plotTwoAxis(Error[['DistErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers')
            init = False
        else:
            DistErrorCoeffs_fig.addScatter(Error[['DistErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key)
        
    DistErrorCoeffs_fig.show()
    

    VelErrorCoeffs_fig = PlotlyPlot()
    
    VelErrorCoeffs_fig.setTitle('Velocity Errors')
    VelErrorCoeffs_fig.setYaxisTitle('Velocity (m/s)')
    VelErrorCoeffs_fig.setYaxis2Title('Velocity (m/s)')
    VelErrorCoeffs_fig.settwoAxisChoice([False, False])
    init = True 
    for key in Results:
        Error = Results[key][0]
        if init == True:
            VelErrorCoeffs_fig.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']], Name = key, mode = 'markers')
            init = False
        else:
            VelErrorCoeffs_fig.addScatter(Error[['VelErr_x']], df_x = Error[['Time']], secondary_y = False, Name = key)
        
    VelErrorCoeffs_fig.show()
    


    #%% Plot Velcocity and Distance Errors
    distVelError_fig = PlotlyPlot()
    
    distVelError_fig.setTitle('Distance and Velocity Error')
    distVelError_fig.setYaxisTitle('Distance (m)')
    distVelError_fig.setYaxis2Title('Velocity (m/s)')
    distVelError_fig.settwoAxisChoice([False, True])
    distVelError_fig.plotTwoAxis(Error[['DistErr_x']], df_x = Error[['Time']], mode = 'markers')
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
    
    
    
    
    
    
    
    