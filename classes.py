import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

class Accelerometer:
    
    def __init__(self):  # Default accelerometer characteristics
     
        self.g = 9.791807                     # Definition of g
        
        self.AccelModelCoef = {'K_1': 1,                         # Scale Factor (g/g) NEEDS UPDATED
                               'K_0': 0,                          # Bias (g)
                               'K_2': 60.14440651  * 10**-6,      # is second-order coefficient (g/g^2)
                               'K_3': 0.0151975    * 10**-6,      # is third-order coefficient  (g/g^3)
                               'K_4': 0.00578331   * 10**-6,      # is fourth-order coefficient (g/g^4)
                               'K_5': 0.002277013  * 10**-6      # is fifth-order coefficient  (g/g^5)
                               }
        
        # self.K_1 = 1                          
        # self.K_0 = 0                          
        # self.K_2 = 60.14440651  * 10**-6      # is second-order coefficient (g/g^2)
        # self.K_3 = 0.0151975    * 10**-6      # is third-order coefficient  (g/g^3)
        # self.K_4 = 0.00578331   * 10**-6      # is fourth-order coefficient (g/g^4)
        # self.K_5 = 0.002277013  * 10**-6      # is fifth-order coefficient  (g/g^5)
        
        ## NEED TO BE UPDATED WITH NEW COEFFICIENTS
        # self.K_0_asym = 0                   # Bias Asymmetry 
        # self.K_1_asym = 0                   # Scale Factor Asymmetry
        # self.K_oq = 0                       # Odd Quadratic Coefficient
        # self.omeg_o = 0                    # is misalignmet of the IA with respect to the OA
        # self.omeg_p = 0                    # is misalignmen of the IA with respect to the PA
        # self.K_ip = 0                      # is crosscoupling coefficient 
        # self.K_io = 0                      # is crosscoupling coefficient
        # self.K_po = 0                      # is crosscoupling coefficient
        # self.K_pp = 1.32E-4 * 10**-6       # is cross-axis nonlinearity coefficients
        # self.K_ppp = 2.10E-7 * 10**-6
        # self.K_pppp = 2.3E-10 * 10**-6
        # self.K_oo = 0                      # is cros-axis nonlinearity coefficients
        # self.K_spin = 0                    # is spin correction coefficient, equal to 
        # self.K_ang_accel = 0               # is angular acceleration coefficient
        

    def simulate(self,a_i,n_start_idx, n_stop_idx):
                 
                 # a_p,a_o):
        """
        Starting with one dimensional error model. Outputs acceleration given
        true input acceleration.
        """
        #Convert acceleration into g
        g_i = a_i / self.g
        
        accel_model = [self.AccelModelCoef['K_0'], 
                       self.AccelModelCoef['K_1'] * (g_i), 
                       self.AccelModelCoef['K_2'] * (g_i**2), 
                       self.AccelModelCoef['K_3'] * (g_i**3), 
                       self.AccelModelCoef['K_4'] * (g_i**4), 
                       self.AccelModelCoef['K_5'] * (g_i**5)] 
        
        print('Start Index: ' + n_start_idx)
        print('End Index: ' + n_stop_idx)
        
        a_x_Sim = self.g * sum(accel_model[n_start_idx:n_stop_idx])
        
             # self.K_0 + self.K_1 * (a_i) + self.K_2 * (a_i**2) + self.K_3 * (a_i**3) + self.K_4 * (a_i**4) + self.K_5 * (a_i**5) 
             # self.omeg_o * a_p +
             # # self.omeg_p * a_o +          
             # # self.K_ip * a_i * a_p +
             # # self.K_io * a_i * a_o +
             # # self.K_po * a_p * a_o +
             # # self.K_pp * (a_p**2) + 
             # # self.K_oo * (a_o**2)
        
        return a_x_Sim

    
class PlotlyPlot:
        
    def __init__(self):
        
        self.title = ''
        self.x_axis = ''
        self.y_axis = ''
        self.y_axis_2 = ''
        self.twoAxisChoice = [False,True]
        
    def setXaxisTitle(self,title):
        self.x_axis = title
        return
    
    def setYaxisTitle(self,title):
        self.y_axis = title
        return
    
    def setYaxis2Title(self,title):
        self.y_axis_2 = title
        return
    
    def setTitle(self,title):
        self.title = title
        return
    
    def settwoAxisChoice(self,twoAxisChoice):
        self.twoAxisChoice = twoAxisChoice
        return
    
    def plotSimple(self,df, x = None, y = None):
        
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

    def plotTwoAxis(self, df, df_x, mode = 'lines'):
        
        #df is a dataframe
        #LeftRight is a list of booleans that determine which y data gets plotted on second axis
        
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

        
        count = 0     
       
        for col in df:
            # Add Traces
            self.fig.add_trace(
                go.Scatter(x = df_x.iloc[:,0], y = df[col], name = col, mode = mode),
                secondary_y = self.twoAxisChoice[count],)
            count += 1
            
        # Add Title 
        self.fig.update_layout(
            title_text = self.title)
        
        # Add Axis Labels
        self.fig.update_xaxes(title_text = self.x_axis)
        self.fig.update_yaxes(title_text = self.y_axis, secondary_y = False)
        self.fig.update_yaxes(title_text = self.y_axis_2, secondary_y = True)

        return
    
    def addScatter(self,df, df_x, secondary_y = None):
        
        name = df.columns.values[0]
        
        if secondary_y != None:
            self.twoAxisChoice.append(secondary_y)
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = name, mode = 'markers'),secondary_y = secondary_y)
        else:
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = name, mode = 'markers'))
    
    def addLine(self,df, df_x, secondary_y = None):
        
        name = df.columns.values[0]
        
        if secondary_y != None:
            self.twoAxisChoice.append(secondary_y)
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = name),secondary_y = secondary_y)
        else:
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = name))
    
    
    def show(self):
        self.fig.show()
        return