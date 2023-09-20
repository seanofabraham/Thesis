
"""
Thesis: Determining high order scale factor non-linearities with sled testing.

Author: Sean Abrahamson

This is the main running code for the thesis. It calls the functions needed.
"""


#%% Import Utilities

# from Thesis_Utils import *
# from scipy.signal import savgol_filter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from classes_x import *
import numpy as np
from scipy import integrate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sigfig import round
import os
from sklearn.linear_model import LinearRegression
# import tkinter as tk
# from tkinter import filedialog


def importEGIData(Headers,filepath):
    
    # Dialog Box to Select Data to import
    
    # root = tk.Tk()
    # root.withdraw()
    # root.wm_attributes('-topmost', 1)
    # filename = filedialog.askopenfilename(parent=root)
    # root.destroy()
    
    
    if filepath == '':
        print('No file selected')
    else: 
        D = pd.read_csv(filepath , names = Headers) # Pull only first row from Excel File

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


def generateReferenceTrajectory(plotcheck = False):
    
    Accel_filepath = './EGI_data/EGI_accel.csv'
    Vel_filepath = './EGI_data/EGI_accel.csv'
    
    EGI_accel = importEGIData(['Time', 'Ax','Ay','Az'],Accel_filepath)
    EGI_vel = importEGIData(['Time', 'Vx','Vy', 'Vz'],Vel_filepath)

    # EGI_accel =  pd.read_csv('EGI_accel.csv',names = ['Time', 'Ax','Ay','Az'])
    # EGI_vel = pd.read_csv('EGI_vel.csv',names = ['Time', 'Vx','Vy', 'Vz'])

    EGI_accel_vel = EGI_accel.join(EGI_vel[['Vx','Vy','Vz']])

    #%% Truth Gen Step 2 - Trim data to focus on actual sled run.
    print('Developing Reference Trajectory')
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
    referenceTrajectory['refAccel_x'][4968:] = 0
    
    
    #%% Truth Gen Step 5 -  Integrate truth acceleration to get velocity and distance
    referenceTrajectory['refVel_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refAccel_x'],x = referenceTrajectory['Time'],initial = 0) 
    
    # Change final Velocity after stop to zero. Determined visually
    print("Setting final velocity at 0...")
    referenceTrajectory['refVel_x'][4968:] = 0
    
    referenceTrajectory['refDist_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refVel_x'],x = referenceTrajectory['Time'],initial = 0) 


    # Integrate EGI velocity to compare to double integrated acceleration
    referenceTrajectory['refEGIDist_x'] = integrate.cumulative_trapezoid(y = referenceTrajectory['refEGIVel_x'],x = referenceTrajectory['Time'],initial = 0) 
    
    # Compute start motion time.
    startMotionTime = referenceTrajectory['Time'][referenceTrajectory['refAccel_x']>0.001].iloc[0]
    
    referenceTrajectory['Time'] = referenceTrajectory['Time']-startMotionTime
    
    
    #%% Save trajectory to Pickle File
    
    referenceTrajectory.to_pickle("./referenceTrajectory.pkl")
    
    #%% Plots Acceleration and Velocity
    if plotcheck == True:
        Figure1 = PlotlyPlot()
        
        Figure1.setTitle('EGI Acceleration, Velocity and Smoothed acceleration')
        Figure1.setYaxisTitle('Acceleration (m/s/s)')
        Figure1.setYaxis2Title('Velocity (m/s)')
        Figure1.setXaxisTitle('GPS Time (s)')
        Figure1.settwoAxisChoice([False, True])
        Figure1.plotTwoAxis(referenceTrajectory[['Ax','Vx']], df_x= EGI_accel_vel_trim[['New Time']])
        Figure1.addLine(referenceTrajectory[['refAccel_x']], df_x = referenceTrajectory[['Time']],secondary_y=False)
        Figure1.show()
    
    
    return 

#%% Generate Track RPV Function 

def generateTrackRPV(referenceTrajectory, sigmaRPV, tauRPV, biasRPV, Overwrite=True):
    
    print("\n Generating RPV")
    trackRPV = pd.DataFrame()
    
    # trackRPVzeroVel = "NoZeroVel"
    trackRPVzeroVel = 'NoZeroVel'
    
    
    if trackRPVzeroVel == "NoZeroVel":
        print("No zero velocity portions of test selected")
    
    Interupter_delta = 4.5 * 0.3048 # ft converted to meters
    TrackLength = 10000   # Meters
    
    trackRPV['Interupters_DwnTrk_dist'] = np.arange(0, TrackLength, Interupter_delta)
    
    trackRPV['Time'] = np.interp(trackRPV['Interupters_DwnTrk_dist'],referenceTrajectory['refDist_x'],referenceTrajectory['Time'])
    
    trackRPV = trackRPV[trackRPV['Interupters_DwnTrk_dist'] <= referenceTrajectory['refDist_x'].max()]
    
    trackRPV = trackRPV.drop_duplicates(subset=['Time'])
    
    trackRPV = trackRPV[:-1]
    
    if trackRPVzeroVel != "NoZeroVel":
        
        trackRPV_zeroVel_start = pd.DataFrame() 
        trackRPV_zeroVel_start['Time'] = referenceTrajectory['Time'][referenceTrajectory['Time']<trackRPV['Time'].min()]
        trackRPV_zeroVel_start['Interupters_DwnTrk_dist'] = 0
        trackRPV_zeroVel_start = trackRPV_zeroVel_start.tail(2)
    
        trackRPV_zeroVel_end = pd.DataFrame()
        
        trackRPV_zeroVel_end['Time'] = referenceTrajectory['Time'][referenceTrajectory['refVel_x']==0]
        trackRPV_zeroVel_end['Time'] = trackRPV_zeroVel_end['Time'][trackRPV_zeroVel_end['Time']>trackRPV['Time'].max()]
        trackRPV_zeroVel_end['Interupters_DwnTrk_dist'] = referenceTrajectory['refDist_x'].max()
        trackRPV_zeroVel_end = trackRPV_zeroVel_end.dropna()
        
        trackRPV_zeroVel_StartEnd = pd.DataFrame()
        trackRPV_zeroVel_StartMidEnd = pd.DataFrame()
        trackRPV_zeroVel_StartMid = pd.DataFrame()
    
        if trackRPVzeroVel == 'StartEnd':
            trackRPV_zeroVel_StartEnd = pd.concat((trackRPV_zeroVel_start,trackRPV_zeroVel_end), axis = 0)
            trackRPV_zeroVel_StartMidEnd = pd.concat((trackRPV, trackRPV_zeroVel_StartEnd), axis = 0)
            trackRPV_zeroVel_StartMidEnd = trackRPV_zeroVel_StartMidEnd.sort_values(by='Time').reset_index(drop=True)
            # trackRPV_zeroVel_StartMidEnd.to_pickle("./trackRPV_0Vel_StartEnd.pkl")
        elif trackRPVzeroVel == 'Start':
            trackRPV_zeroVel_StartMid = pd.concat((trackRPV, trackRPV_zeroVel_start), axis = 0)
            trackRPV_zeroVel_StartMid = trackRPV_zeroVel_StartMid.sort_values(by='Time').reset_index(drop=True)    
            # trackRPV_zeroVel_StartMid.to_pickle("./trackRPV_0Vel_Start.pkl")
            trackRPV = trackRPV_zeroVel_StartMid
    
    
    trackRPV = trackRPV.sort_values(by='Time').reset_index(drop=True)
    
    # trackRPV['Time'] = trackRPV['Time']-trackRPV['Time'][0]
    
    # Add error to Track RPV
    if sigmaRPV != 0:
        noise = np.random.normal(0,sigmaRPV,len(trackRPV)) # Add random noise to RPV
        trackRPV['Interupters_DwnTrk_dist'] = trackRPV['Interupters_DwnTrk_dist'] + noise
    
    if tauRPV != 0:
        trackRPV['Time'] = trackRPV['Time'] - tauRPV
        
    if biasRPV != 0:
        trackRPV['Interupters_DwnTrk_dist'] = trackRPV['Interupters_DwnTrk_dist'] + biasRPV


    #%% Save track RPV to pickle file
    if Overwrite == True:
        trackRPV.to_pickle(f"./RPVs/trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}.pkl")
    else:
       filepath = incrementFileName(f"./VarianceRPVs/trackRPV_sig{sigmaRPV}_tau{tauRPV}_bias{biasRPV}.pkl")
       trackRPV.to_pickle(filepath)
    return


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
    
    sensorSim['SensorSim_Ax'][referenceTrajectory['refAccel_x'] == 0] = 0
    
    #%% Integrate Simulated accelerations to develop Velocity and Displacement.
    sensorSim['SensorSim_Vx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Ax'],x = sensorSim['Time'],initial = 0) 
    sensorSim['SensorSim_Dx'] = integrate.cumulative_trapezoid(y = sensorSim['SensorSim_Vx'],x = sensorSim['Time'],initial = 0) 
    
    AccelObj = AccelOne
    
    return [sensorSim, AccelObj]

def RegressionAnalysis(referenceTrajectory, trackRPV, AccelObj, sensorSim, N_model, g,sigmaRPV, saveToPickel = False, WLSoption = True, LeastSquaresMethod = 'LongHand'):
    
    #%% Error - Compare simulated acceleromter with track reference
    """
    Error - Scripts used to compare accelerometer simulation versus track truth
    """
    Dist_Error = pd.DataFrame()
    Dist_Error['Time'] = trackRPV['Time']
    
    # Interpolate Sensor Sim to Track
    
    trackRPV['SensorDwnTrkDist'] = np.interp(trackRPV['Time'],sensorSim['Time'],sensorSim['SensorSim_Dx'])
    
    Dist_Error['DistErr_x'] = trackRPV['Interupters_DwnTrk_dist'] - trackRPV['SensorDwnTrkDist']
    

    # Compute Velocity Error
    Ve_x = (np.diff(Dist_Error['DistErr_x'])/np.diff(Dist_Error['Time']))
    Ve_t = (Dist_Error['Time'].head(-1) + np.diff(Dist_Error['Time'])/2).to_numpy()
    
    Error = pd.DataFrame()
    
    Error['Time'] = Ve_t
    Error['SensorSim_Ax'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Ax']) 
    Error['SensorSim_Vx'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Vx'])
    Error['SensorSim_Dx'] = np.interp(Ve_t,sensorSim['Time'],sensorSim['SensorSim_Dx'])
    Error['DistErr_x'] = np.interp(Ve_t,Dist_Error['Time'],Dist_Error['DistErr_x']) 
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
    
    
    Vx = np.interp(Ve_t, referenceTrajectory['Time'],referenceTrajectory['refVel_x'])
    intAx_2 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^2 (g)']) 
    intAx_3 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^3 (g)']) 
    intAx_4 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^4 (g)']) 
    intAx_5 = np.interp(Ve_t,referenceTrajectory['Time'],referenceTrajectory['intAx^5 (g)'])
    
    coordinateFunctionDF = pd.DataFrame()
    coordinateFunctionDF['Time'] = Ve_t
     
    
    coeff_dict = {'Est_V_0': 0, 'Est_K_1': 0, 'Est_K_0': 0, 'Est_K_2': 0, 'Est_K_3': 0, 'Est_K_4': 0, 'Est_K_5': 0}
    
    # Create Complete A Matrix
    complete_A = np.array([np.ones(len(Ve_t))/g, -Vx/g, -Ve_t, intAx_2, intAx_3, intAx_4, intAx_5])*g
    complete_A = complete_A.T
    
    complete_A_DF = pd.DataFrame(np.fliplr(complete_A), columns=['IntAx_5', 'IntAx_4', 'IntAx_3', 'IntAx_2', 'Ve_t', 'Vx', 'Ones'])
    
    trimmed_A_filt = np.zeros(complete_A.shape[1], dtype = bool)
    trimmed_A_filt[0] = 1
    
    trimmed_A_filt[N_model[0]+1:N_model[1]+1] = 1

    trimmed_A = complete_A[:,trimmed_A_filt]
    
    '''
    COMPUTE COVARIANCE
    '''

    #%% Compute Covariance    
    
    #%% 
    # Linear Regression
    coeff_list = tuple(None for _ in range(trimmed_A.shape[1]))

    
    if sigmaRPV == 0 or WLSoption == False: 
        size = trimmed_A.shape[0]
        W = np.identity(size)
    
    else: 
        
        # Develop weighted matrix
        delta_t = np.diff(trackRPV['Time'])
        vel_sig = np.sqrt(2)*sigmaRPV/delta_t
        
        W = np.diag(vel_sig,0) - np.diag((.5*vel_sig[1:]),-1) - np.diag((.5*vel_sig[1:]),1)
    
        W = np.linalg.inv(W)
        
        # W = np.diag(1/vel_sig) 
        
    A = trimmed_A
    
    AW = np.transpose(trimmed_A).dot(W)
    Ve_xW = W.dot(Ve_x)
    

   
    if LeastSquaresMethod == 'Numpy':

        coeff_list = np.linalg.lstsq(np.transpose(AW), Ve_xW, rcond=None)[0] # This has just been used for debugging to check if "Long" least squares leads to same results.
    
    elif LeastSquaresMethod == 'SciKit':
        testSKlearn = LinearRegression()
        testSKlearn.fit(trimmed_A, Ve_x, sample_weight=(np.diag(W)))
        coeff_list = testSKlearn.coef_
        coeff_list[0] = testSKlearn.intercept_
    
    elif LeastSquaresMethod == 'LongHand':
        At = np.transpose(trimmed_A)
        coeff_list = np.linalg.inv(At.dot(W).dot(trimmed_A)).dot(At).dot(W).dot(Ve_x)
        
    else: 
        print("Did not select an applicable Least Squares Method")


    covariance_A = np.linalg.inv(np.dot(AW,trimmed_A))

    print_List = np.array(list(coeff_dict.keys()))
    
    n = 0
    for coef in print_List[trimmed_A_filt]:
        coeff_dict[coef] = coeff_list[n]
        n += 1
    
    #%% Save results to DataFrame
    
    coefficientDF = pd.DataFrame()
    
    coefficientDF = pd.concat((coefficientDF, pd.DataFrame.from_dict(AccelObj.AccelModelCoef, orient = 'index', columns= ['Accel Model'])))
    
    coefficientDF.loc['V_0'] = 0
    
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
        

    return [coefficientDF, Error, covariance_A, A, Ve_x, W, LeastSquaresMethod]

def figText(text):

    LaTeXText = '$\\text{' + text + ' }$'

    return LaTeXText

def round_array_to_sigfigs(array, sigfigs):
    rounded_array = np.zeros_like(array)  # Create an array of zeros with the same shape as the input array
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == 0:
                rounded_array[i, j] = 0
            else:
                rounded_array[i, j] = round(array[i, j], sigfigs-1-int(np.floor(np.log10(np.abs(array[i, j])))))  # Calculate the number of decimals based on significant figures
    
    return rounded_array


def incrementFileName(base_path):
    
    # initialize the increment variable
    increment = 0
    
    # loop until we find a file name that doesn't exist

    while True:
        # create the file path with the increment
        file_path = f"{os.path.splitext(base_path)[0]}_{increment}{os.path.splitext(base_path)[1]}"
    
        # check if the file exists
        if os.path.isfile(file_path):
            # if it does, increment the counter and try again
            increment += 1
        else:
            # if it doesn't, break out of the loop
            break

    return file_path
    