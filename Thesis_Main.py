
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

#%% Initial Configuration
generateNewTrajectory = False
plotcheck = False

changeDefaultCoef = False
CoeffDict = {'K_0': .05}

N_model_start = 5  #  0 =  K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 
N_model_end = 5   #  0 = K_1 (Scale Factor), 1 = K_0 (Bias), 2 = K_2, etc. 

g = 9.791807  
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

# Generate track reference position vectory

if generateNewTrajectory == True:    
    generateTrackRPV(referenceTrajectory)
    
trackRPV = pd.read_pickle("./trackRPV.pkl") 

#%% Plotcheck 
if plotcheck == True:
    figRefTraj = px.scatter(x = referenceTrajectory['Time'],y = referenceTrajectory['Accel_x'])
    figRefTraj.show()   

    figRPV = px.scatter(x = trackRPV['Time'],y = trackRPV['Interupters_DwnTrk_dist'])
    figRPV.add_trace(go.Scatter(x = referenceTrajectory['Time'],y = referenceTrajectory['IntDist_x']))
    figRPV.show()   
        
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
A_i_true = referenceTrajectory['Accel_x'].to_numpy()  

# Simulate
A_x_sim = AccelOne.simulate(A_i_true, N_model_start_idx, N_model_end_idx)  

#Store data in data frame. 
sensorSim['Ax'] = A_x_sim

#%% Integrate Simulated accelerations to develop Velocity and Displacement.
sensorSim['Vx'] = integrate.cumulative_trapezoid(y = sensorSim['Ax'],x = sensorSim['Time'],initial = 0) 
sensorSim['Dx'] = integrate.cumulative_trapezoid(y = sensorSim['Vx'],x = sensorSim['Time'],initial = 0) 

#%% Error - Compare simulated acceleromter with track reference
"""
Error - Scripts used to compare accelerometer simulation versus track truth
"""
Dist_Error = pd.DataFrame()
Dist_Error['Time'] = trackRPV['Time']

# Interpolate Sensor Sim to Track

trackRPV['SensorInterpDist'] = np.interp(trackRPV['Time'],sensorSim['Time'],sensorSim['Dx'])

px.scatter(trackRPV, x = 'Time', y = ['SensorInterpDist','Interupters_DwnTrk_dist'])

Dist_Error['De_x'] = trackRPV['Interupters_DwnTrk_dist'] - trackRPV['SensorInterpDist']

# Compute Velocity Error
Ve_x = np.diff(Dist_Error['De_x'])/np.diff(Dist_Error['Time'])
Ve_t = (trackRPV['Time'].head(-1) + np.diff(Dist_Error['Time'])/2).to_numpy() # UPDATE TIME TAG FOR DIFFERENTIATION.

Error = pd.DataFrame()

Error['Time'] = Ve_t
Error['Ax'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['Ax']) 
Error['Vx'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['Vx'])
Error['VelErr_x'] = Ve_x
 
#%% - Regression Analysis
"""
Regression Analysis - Scripts used to compute error model
"""

# # Compute coordinate functions
sensorSim['Ax^2'] = sensorSim[['Ax']]**2
sensorSim['Ax^3'] = sensorSim[['Ax']]**3
sensorSim['Ax^4'] = sensorSim[['Ax']]**4
sensorSim['Ax^5'] = sensorSim[['Ax']]**5

sensorSim['intAx^2'] = integrate.cumulative_trapezoid(y = sensorSim['Ax^2'],x = sensorSim['Time'],initial = 0) 
sensorSim['intAx^3'] = integrate.cumulative_trapezoid(y = sensorSim['Ax^3'],x = sensorSim['Time'],initial = 0) 
sensorSim['intAx^4'] = integrate.cumulative_trapezoid(y = sensorSim['Ax^4'],x = sensorSim['Time'],initial = 0) 
sensorSim['intAx^5'] = integrate.cumulative_trapezoid(y = sensorSim['Ax^5'],x = sensorSim['Time'],initial = 0) 

# Computer Jerk Term
# sensorSim['Jx'] = 

Vx = np.interp(Ve_t, sensorSim['Time'],sensorSim['Vx'])
intAx_2 = np.interp(Ve_t,sensorSim['Time'],sensorSim['intAx^2']) 
intAx_3 = np.interp(Ve_t,sensorSim['Time'],sensorSim['intAx^3']) 
intAx_4 = np.interp(Ve_t,sensorSim['Time'],sensorSim['intAx^4']) 
intAx_5 = np.interp(Ve_t,sensorSim['Time'],sensorSim['intAx^5']) 

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

## UPDATE COEFFICIENT VALUES TO RIGHT UNITS.

#%% Display estimated error coefficient values

print_List = np.array(list(coeff_dict.keys()))

print('\nEstimated Error Coefficients')

n = 0
for coef in print_List[trimmed_A_filt]:
    coeff_dict[coef] = coeff_list[n]
    n += 1 

# coeff_dict['Est_K_1'] = coeff_dict['Est_K_1'] + 1

for coeff, value in coeff_dict.items():
    print(f"{coeff}: {value}")
    
print('\nAcclerometer Simulation Error Coefficients')    
i = 0
for coeff, value in AccelOne.AccelModelCoef.items():
    print(f"{coeff}: {value}")
    if i == N_model_end:
        break
    else:
        i += 1
        
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
    Figure1.addLine(referenceTrajectory[['Accel_x']], df_x = referenceTrajectory[['Time']],secondary_y=False)
    Figure1.show()
    
    
    #%% Plot Integration Velocity Results
    Figure2 = PlotlyPlot()
    
    Figure2.setTitle('EGI Acceleration and Velocity and Integrated Velocity')
    Figure2.setYaxisTitle('Acceleration (m/s/s)')
    Figure2.setYaxis2Title('Velocity (m/s)')
    Figure2.setXaxisTitle('GPS Time (s)')
    Figure2.settwoAxisChoice([False, True, True])
    Figure2.plotTwoAxis(referenceTrajectory[['Accel_x', 'IntVel_x', 'EGIVel_x']], df_x = referenceTrajectory[['Time']], mode = 'markers')
    Figure2.show()
    
    #%% Plot integration results
    Figure3 = PlotlyPlot()
    
    Figure3.setTitle('EGI displacement from EGI velocity and from Integrated acceleration')
    Figure3.setYaxisTitle('Velocity (m/s)')
    Figure3.setYaxis2Title('Distance (m)')
    Figure3.settwoAxisChoice([False, False, True, True])
    Figure3.plotTwoAxis(referenceTrajectory[['IntVel_x', 'EGIVel_x', 'IntDist_x', 'EGIDist_x']], df_x = referenceTrajectory[['Time']])
    Figure3.show()
    
    #%% Plot integration for sensor simulation results
    Figure4 = PlotlyPlot()
    
    Figure4.setTitle('Sensor Accelerometer sim and integrated velocity vs Truth')
    Figure4.setYaxisTitle('Velocity (m/s)')
    Figure4.setYaxis2Title('Distance (m)')
    Figure4.settwoAxisChoice([False, True])
    Figure4.plotTwoAxis(referenceTrajectory[['Accel_x', 'IntVel_x']], df_x = referenceTrajectory[['Time']])
    Figure4.addScatter(sensorSim[['Ax']], df_x = sensorSim[['Time']],secondary_y=False)
    Figure4.addScatter(sensorSim[['Vx']], df_x = sensorSim[['Time']],secondary_y=True)
    Figure4.show()
    
    
    #%% Plot integration for sensor simulation results
    Figure5 = PlotlyPlot()
    
    Figure5.setTitle('Sensor integrated velocity sim and distance velocity vs referenceTrajectory')
    Figure5.setYaxisTitle('Velocity (m/s)')
    Figure5.setYaxis2Title('Distance (m)')
    Figure5.settwoAxisChoice([False, True])
    Figure5.plotTwoAxis(referenceTrajectory[['IntVel_x','IntDist_x']], df_x = referenceTrajectory[['Time']])
    Figure5.addScatter(sensorSim[['Vx']], df_x = sensorSim[['Time']],secondary_y=False)
    Figure5.addScatter(sensorSim[['Dx']], df_x = sensorSim[['Time']],secondary_y=True)
    Figure5.show()
    
    #%% Plot Errors
    Figure6 = PlotlyPlot()
    
    Figure6.setTitle('Distance and Velocity Error')
    Figure6.setYaxisTitle('Distance Error (m)')
    Figure6.setYaxis2Title('Velocity Error (m/s)')
    Figure6.settwoAxisChoice([False, True])
    Figure6.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']])
    Figure6.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']])
    
    Figure6.show()
    
    #%% Plot Error
    Figure7 = PlotlyPlot()
    
    Figure7.setTitle('Velocity Residuals')
    Figure7.setYaxisTitle('Velocity Error (m/s)')
    Figure7.setYaxis2Title('Velocity (m/s)')
    Figure7.settwoAxisChoice([False])
    Figure7.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']])
    # Figure7.addScatter(referenceTrajectory[['IntVel_x']], df_x = referenceTrajectory[['Time']],secondary_y=True)
    Figure7.addScatter(trackRPV[['Interupters_DwnTrk_dist']], df_x = trackRPV[['Time']],secondary_y=True)


    Figure7.show()
    
    #%% Plot Error vs Velocity and Acceleration

    Figure_VelErrorVsVel = PlotlyPlot()

    Figure_VelErrorVsVel.setTitle('Velocity Error')
    Figure_VelErrorVsVel.setYaxisTitle('Velocity Error (m/s)')
    Figure_VelErrorVsVel.setYaxis2Title('Velocity (m/s)')
    Figure_VelErrorVsVel.setXaxisTitle('GPS Time (s)')
    Figure_VelErrorVsVel.settwoAxisChoice([False, True])
    Figure_VelErrorVsVel.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']])
    Figure_VelErrorVsVel.addLine(Error[['Vx']], df_x = Error[['Time']],secondary_y=True)
    Figure_VelErrorVsVel.show()

    Figure_VelErrorVsAccel = PlotlyPlot()

    Figure_VelErrorVsAccel.setTitle('Velocity Error')
    Figure_VelErrorVsAccel.setYaxisTitle('Velocity Error (m/s)')
    Figure_VelErrorVsAccel.setYaxis2Title('Velocity (m/s)')
    Figure_VelErrorVsAccel.setXaxisTitle('GPS Time (s)')
    Figure_VelErrorVsAccel.settwoAxisChoice([False, True])
    Figure_VelErrorVsAccel.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']])
    Figure_VelErrorVsAccel.addLine(Error[['Ax']], df_x = Error[['Time']],secondary_y=True)
    Figure_VelErrorVsAccel.show()

#%% Plots Residuals Acceleration and Velocity
    
    Figure = PlotlyPlot()
    
    Figure.setTitle('Residuals')
    Figure.setYaxisTitle('Velocity Error Residuals (m/s)))')
    Figure.setYaxis2Title('Velocity (m/s)')
    Figure.setXaxisTitle('GPS Time (s)')
    Figure.settwoAxisChoice([False, True])
    Figure.plotTwoAxis(Error[['Ve_x_Resid']], df_x = Error[['Time']])
    Figure.addLine(Error[['Vx']], df_x = Error[['Time']],secondary_y=True)
    Figure.show()
    
    Figure_x = PlotlyPlot()
    
    Figure_x.setTitle('Residuals')
    Figure_x.setYaxisTitle('Velocity Error Residuals (m/s)')
    Figure_x.setYaxis2Title('Acceleration (m/s/s)')
    Figure_x.setXaxisTitle('GPS Time (s)')
    Figure_x.settwoAxisChoice([False, True])
    Figure_x.plotTwoAxis(Error[['Ve_x_Resid']], df_x = Error[['Time']])
    Figure_x.addLine(Error[['Ax']], df_x = Error[['Time']],secondary_y=True)
    Figure_x.show()
        
#%% Plot Velocity Error versus velocity error model
    VelErrVsResid = PlotlyPlot()
    
    VelErrVsResid.setTitle('Velocity Error vs Estimated Error Model')
    VelErrVsResid.setYaxisTitle('Velocity Error (m/s)')
    VelErrVsResid.setXaxisTitle('GPS Time (s)')
    VelErrVsResid.plotTwoAxis(Error[['VelErr_x']], df_x = Error[['Time']], mode = 'markers')
    VelErrVsResid.addScatter(Error[['V_error_model']], df_x = Error[['Time']], secondary_y=False)
    # VelErrVsResid.addScatter(Error[['A_Bias']], df_x = Error[['Time']], secondary_y=False)
    VelErrVsResid.show()

