
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

def AccelSim(referenceTrajectory, N_model, changeDefaultCoeff, CoeffDict, g):
    
    #%% ACCEL SIM Step 1 - Simulate a Acceleromter with Bias using Accelerometer class
    """
    ACCEL SIM - Scripts used to generate simulated accelerometer output based on truth input
    
    Using smoothed acceleration truth data to simulate
    """
    
    AccelOne = Accelerometer()
    
    if changeDefaultCoeff == True:
            AccelOne.AccelModelCoef.update(CoeffDict)
    
    AccelOne.g = g
    
    # Create data frame to house data
    sensorSim = pd.DataFrame()
    sensorSim['Time'] = referenceTrajectory['Time']
    
    # Change to array for us in simulation.
    A_i_true = referenceTrajectory['refAccel_x'].to_numpy()  
    
    # Simulate
    A_x_sim = AccelOne.simulate(A_i_true, N_model[0], N_model[1])  
    
    #Store data in data frame. 
    sensorSim['SensorSim_Ax'] = A_x_sim
    
    #%% Integrate Simulated accelerations to develop Velocity and Displacement.
    sensorSim['SensorSim_Vx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Ax'],x = sensorSim['Time'],initial = 0) 
    sensorSim['SensorSim_Dx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Vx'],x = sensorSim['Time'],initial = 0) 
    
    AccelObj = AccelOne
    
    return sensorSim, AccelObj

def RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g, saveToPickel = False):
    
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
    Error['SensorSim_Dx'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Dx'])
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
    
    trimmed_A_filt[N_model[0]+1:N_model[1]] = 1
    
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
    
    #%% Save results to DataFrame
    
    coefficientDF = pd.DataFrame()
    
    coefficientDF = pd.concat((coefficientDF, pd.DataFrame.from_dict(AccelObj.AccelModelCoef, orient = 'index', columns= ['Accel Model'])))
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
    
                
    #%% Compute Velocity Error Residuals
    
    V_error_model_terms = [coeff_dict['Est_V_0'], 
                           coeff_dict['Est_K_1']*Vx,  
                           coeff_dict['Est_K_0']*Ve_t, 
                           coeff_dict['Est_K_2']*intAx_2, 
                           coeff_dict['Est_K_3']*intAx_3,  
                           coeff_dict['Est_K_4']*intAx_4,  
                           coeff_dict['Est_K_5']*intAx_5]
    
    Error['V_error_model'] = sum(V_error_model_terms)*g 
    Error['Ve_x_Resid'] = Error['VelErr_x'] - Error['V_error_model'] 
   
        
    #%% Save off results:
    if saveToPickel == True:
        Error.to_pickle(f"./ErrorDF_{N_model[0]}-{N_model[1]}.pkl")
        coefficientDF.to_pickle(f"./coefficientDF_{N_model[0]}-{N_model[1]}.pkl")

    return coefficientDF, Error

