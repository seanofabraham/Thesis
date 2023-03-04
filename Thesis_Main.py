
"""
Thesis: Determining high order scale factor non-linearities with sled testing.

Author: Sean Abrahamson

This is the main running code for the thesis. It calls the functions needed.
"""


#%% Import Utilities

from Thesis_Utils import *
# from scipy.signal import savgol_filter
from classes_x import *
import numpy as np
from scipy import integrate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

#%% Initial Configuration Parameters

#Coefficients
g = 9.791807  

#Generate New Trajectory and RPV
generateNewTrajectory = False

# NoZeroVel = No portions of zero velocity areas
# Start = Zero velocity area at beginning
# StartEnd = Zero velocity are at beginning and end
RPVType = 'NoZeroVel'

# Used to play around with coefficients
changeDefaultCoef = False
CoeffDict = {'K_2': 5E-6}

# Used to determine how many coefficients to calculate
N_model_start = 5     #  0 =  K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 
N_model_end = 5      #  0 = K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 

# Fix indexing numbers
N_model_start_idx = N_model_start
N_model_end_idx = N_model_end + 1

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

if generateNewTrajectory == True:    
    generateTrackRPV(referenceTrajectory)


if RPVType == 'Start':
    trackRPV = pd.read_pickle("./trackRPV_0Vel_Start.pkl")
elif RPVType == 'StartEnd':
    trackRPV = pd.read_pickle("./trackRPV_0Vel_StartEnd.pkl") 
elif RPVType == 'NoZeroVel':
    trackRPV = pd.read_pickle("./trackRPV_noZeroVel.pkl") 
else:
    print('No acceptable RPV type selected. Using RPV with no 0Vel areas...')
    trackRPV = pd.read_pickle("./trackRPV_noZeroVel.pkl") 

#%%

# trackRPV_zeroVel_end = pd.DataFrame()


# trackRPV_zeroVel_end['Time'] = referenceTrajectory['Time'][referenceTrajectory['refVel_x']==0]
# trackRPV_zeroVel_end['Time'] = trackRPV_zeroVel_end['Time'][trackRPV_zeroVel_end['Time']>trackRPV['Time'].max()]

# trackRPV_zeroVel_end['Interupters_DwnTrk_dist'] = referenceTrajectory['refDist_x'].max()

#%% ACCEL SIM Step 1 - Simulate a Acceleromter with Bias using Accelerometer class
"""
ACCEL SIM - Scripts used to generate simulated accelerometer output based on truth input

Using smoothed acceleration truth data to simulate
"""

AccelOne = Accelerometer()

if changeDefaultCoef == True:
        AccelOne.AccelModelCoef.update(CoeffDict)

# Create data frame to house data
sensorSim = pd.DataFrame()
sensorSim['Time'] = referenceTrajectory['Time']

# Change to array for us in simulation.
A_i_true = referenceTrajectory['refAccel_x'].to_numpy()  

# Simulate
A_x_sim = AccelOne.simulate(A_i_true, N_model_start_idx, N_model_end_idx)  

#Store data in data frame. 
sensorSim['SensorSim_Ax'] = A_x_sim

#%% Integrate Simulated accelerations to develop Velocity and Displacement.
sensorSim['SensorSim_Vx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Ax'],x = sensorSim['Time'],initial = 0) 
sensorSim['SensorSim_Dx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Vx'],x = sensorSim['Time'],initial = 0) 

#%% Error - Compare simulated acceleromter with track reference
"""
Error - Scripts used to compare accelerometer simulation versus track truth
"""
Dist_Error = pd.DataFrame()
Dist_Error['Time'] = trackRPV['Time']

# Interpolate Sensor Sim to Track

trackRPV['SensorInterpDist'] = np.interp(trackRPV['Time'],sensorSim['Time'],sensorSim['SensorSim_Dx'])

Dist_Error['DistErr_x'] = trackRPV['Interupters_DwnTrk_dist'] - trackRPV['SensorInterpDist']

# Compute Velocity Error
Ve_x = np.diff(Dist_Error['DistErr_x'])/np.diff(Dist_Error['Time'])
Ve_t = (trackRPV['Time'].head(-1) + np.diff(Dist_Error['Time'])/2).to_numpy() # UPDATE TIME TAG FOR DIFFERENTIATION.

Error = pd.DataFrame()

Error['Time'] = Ve_t
Error['SensorSim_Ax'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Ax']) 
Error['SensorSim_Vx'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Vx'])
Error['VelErr_x'] = Ve_x
 
#%% - Regression Analysis
"""
Regression Analysis - Scripts used to compute error model
"""

# # Compute coordinate functions
referenceTrajectory['Ax^2 (g)'] = (referenceTrajectory[['refAccel_x']]/g)**2
referenceTrajectory['Ax^3 (g)'] = (referenceTrajectory[['refAccel_x']]/g)**3
referenceTrajectory['Ax^4 (g)'] = (referenceTrajectory[['refAccel_x']]/g)**4
referenceTrajectory['Ax^5 (g)'] = (referenceTrajectory[['refAccel_x']]/g)**5

referenceTrajectory['intAx^2 (g)'] = -integrate.cumulative_trapezoid(y = referenceTrajectory['Ax^2 (g)'],x = referenceTrajectory['Time'],initial = 0) 
referenceTrajectory['intAx^3 (g)'] = -integrate.cumulative_trapezoid(y = referenceTrajectory['Ax^3 (g)'],x = referenceTrajectory['Time'],initial = 0) 
referenceTrajectory['intAx^4 (g)'] = -integrate.cumulative_trapezoid(y = referenceTrajectory['Ax^4 (g)'],x = referenceTrajectory['Time'],initial = 0)
referenceTrajectory['intAx^5 (g)'] = -integrate.cumulative_trapezoid(y = referenceTrajectory['Ax^5 (g)'],x = referenceTrajectory['Time'],initial = 0) 

# Computer Jerk Term
# sensorSim['Jx'] = 

Vx = np.interp(Ve_t, referenceTrajectory['Time'],referenceTrajectory['refVel_x'])
intAx_2 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^2 (g)']) 
intAx_3 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^3 (g)']) 
intAx_4 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^4 (g)']) 
intAx_5 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^5 (g)'])

coordinateFunctionDF = pd.DataFrame()
coordinateFunctionDF['Time'] = Ve_t
 

coeff_dict = {'Est_V_0': 0, 'Est_K_1': 0, 'Est_K_0': 0, 'Est_K_2': 0, 'Est_K_3': 0, 'Est_K_4': 0, 'Est_K_5': 0}

# Create Complete A Matrix
complete_A = np.array([np.ones(len(Ve_t)), Vx, Ve_t, intAx_2, intAx_3, intAx_4, intAx_5])
complete_A = complete_A.T

complete_A_DF = pd.DataFrame(np.fliplr(complete_A), columns=['IntAx_5', 'IntAx_4', 'IntAx_3', 'IntAx_2', 'Ve_t', 'Vx', 'Ones'])

trimmed_A_filt = np.zeros(complete_A.shape[1], dtype = bool)
trimmed_A_filt[0] = 1

trimmed_A_filt[N_model_start_idx+1:N_model_end_idx+1] = 1

trimmed_A = complete_A[:,trimmed_A_filt]

# trimmed_A = np.fliplr(trimmed_A)

# Linear Regression
coeff_list = tuple(None for _ in range(trimmed_A.shape[1]))

coeff_list = np.linalg.lstsq(trimmed_A*g, Ve_x, rcond=None)[0]

print_List = np.array(list(coeff_dict.keys()))

n = 0
for coef in print_List[trimmed_A_filt]:
    coeff_dict[coef] = coeff_list[n]
    n += 1

## UPDATE COEFFICIENT VALUES TO RIGHT UNITS.

#%% Save results to DataFrame

coefficientDF = pd.DataFrame()

coefficientDF = pd.concat((coefficientDF, pd.DataFrame.from_dict(AccelOne.AccelModelCoef, orient = 'index', columns= ['Accel Model'])))
coefficientDF = coefficientDF.append(pd.Series([0], index=coefficientDF.columns, name="V_0"))

# Build Estimated Coefficient DF
estimatedCoefficients = pd.DataFrame.from_dict(coeff_dict, orient = 'index', columns= ['Estimated Coefficients'])

renameDict = {}
for coeff in print_List:
    renameDict[coeff] = coeff[4:]
    
estimatedCoefficients = estimatedCoefficients.rename(index = renameDict) 
estimatedCoefficients.replace(0, np.nan, inplace=True)


coefficientDF = pd.merge(coefficientDF,estimatedCoefficients,left_index=True, right_index=True)

coefficientDF['Coefficient Estimate Error'] = coefficientDF['Accel Model'] - coefficientDF['Estimated Coefficients']

coefficientDF.to_pickle(f"./coefficientDF_{RPVType}.pkl")

#%% Display estimated error coefficient values

print(coefficientDF)
        
#%%  Plot the residual

V_error_model_terms = [coeff_dict['Est_V_0'], 
                       coeff_dict['Est_K_1']*Vx,  
                       coeff_dict['Est_K_0']*Ve_t, 
                       coeff_dict['Est_K_2']*intAx_2, 
                       coeff_dict['Est_K_3']*intAx_3,  
                       coeff_dict['Est_K_4']*intAx_4,  
                       coeff_dict['Est_K_5']*intAx_5]

Error['V_error_model'] = sum(V_error_model_terms)*g 

Error['Ve_x_Resid'] = Error['VelErr_x'] - Error['V_error_model'] 

#%% Plots scripts 
"""
PLOTS

"""
Plots = False

if Plots == True: 
    #%% Plots Acceleration and Velocity
    
    Figure1 = PlotlyPlot()
    
    Figure1.setTitle('EGI Acceleration, Velocity and Smoothed acceleration')
    Figure1.setYaxisTitle('Acceleration (m/s/s)')
    Figure1.setYaxis2Title('Velocity (m/s)')
    Figure1.setXaxisTitle('GPS Time (s)')
    Figure1.settwoAxisChoice([False, True])
    Figure1.plotTwoAxis(EGI_accel_vel_trim[['Ax','Vx']], df_x= EGI_accel_vel_trim[['New Time']])
    Figure1.addLine(referenceTrajectory[['refAccel_x']], df_x = referenceTrajectory[['Time']],secondary_y=False)
    Figure1.show()
    
    
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

