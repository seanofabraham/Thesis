import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'

class Accelerometer:
    
    def __init__(self):  # Default accelerometer characteristics
     
        self.g = 9.791807                     # Definition of g
        
        self.AccelModelCoef = {'K_1': 0,                          # Scale Factor (g/g) NEEDS UPDATED
                               'K_0': 5            * 10**-6,      # Bias (g)
                               'K_2': 60.14440651  * 10**-6,      # is second-order coefficient (g/g^2)
                               'K_3': 0.0151975    * 10**-6,      # is third-order coefficient  (g/g^3)
                               'K_4': 0.00578331   * 10**-6,      # is fourth-order coefficient (g/g^4)
                               'K_5': 0.002277013  * 10**-6       # is fifth-order coefficient  (g/g^5)
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
        
        accel_model = [self.AccelModelCoef['K_1'] * (g_i),
                       self.AccelModelCoef['K_0'] * np.ones(len(g_i)),  
                       self.AccelModelCoef['K_2'] * (g_i**2), 
                       self.AccelModelCoef['K_3'] * (g_i**3), 
                       self.AccelModelCoef['K_4'] * (g_i**4), 
                       self.AccelModelCoef['K_5'] * (g_i**5)]
        
        # print(accel_model[n_start_idx:n_stop_idx])
        
        a_x_Sim = self.g * sum(accel_model[n_start_idx:n_stop_idx]) + a_i
        
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
        self.template = 'simple_white'
        
    def setXaxisTitle(self,title):
        if title[0] != '$':
            self.x_axis = self.figText(title)
        else:        
            self.x_axis = title
        return
    
    def setYaxisTitle(self,title):
        if title[0] != '$':
            self.y_axis = self.figText(title)
        else:        
            self.y_axis = title
        return
    
    def setYaxis2Title(self,title):
        if title[0] != '$':
            self.y_axis_2 = self.figText(title)
        else:        
            self.y_axis_2 = title
        return
    
    def setTitle(self,title):
        if title[0] != '$':
            self.title = self.figText(title)
        else:        
            self.title = title
        return
    
    def settwoAxisChoice(self,twoAxisChoice):
        self.twoAxisChoice = twoAxisChoice
        return
    
    def plotSimple(self,df, x = None, y = None):
        
        if x == None and y == None:
            self.fig = px.line(df)
            self.fig.show()
        elif y == None:
            self.fig = px.line(df, x = x)
            self.fig.show()
        else:
            self.fig = px.line(df, x = x, y = y)
            self.fig.show()    
            
        return

    def plotTwoAxis(self, df, df_x, mode = 'lines', Name = None):
        
        #df is a dataframe
        #LeftRight is a list of booleans that determine which y data gets plotted on second axis
        
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])

        
        count = 0     
       
        for col in df:
            # Add Traces
            if Name != None: 
                self.fig.add_trace(
                    go.Scatter(x = df_x.iloc[:,0], y = df[col], name = Name, mode = mode),
                    secondary_y = self.twoAxisChoice[count],)
            else:
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
    
    def addScatter(self,df, df_x, secondary_y = None, Name = None, Mode = 'markers'):
        
        if Name == None:
            Name = df.columns.values[0]
            
        if secondary_y != None:
            self.twoAxisChoice.append(secondary_y)
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = Name, mode = Mode),secondary_y = secondary_y)
        else:
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = Name, mode = Mode))
    
    def addLine(self,df, df_x, secondary_y = None, Name = None):
        
        name = df.columns.values[0]
        
        if secondary_y != None:
            self.twoAxisChoice.append(secondary_y)
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = Name),secondary_y = secondary_y)
        else:
            self.fig.add_trace(go.Scatter(x = df_x.iloc[:,0],y = df.iloc[:,0], name = Name))
    
    def legendTopRight(self):
        self.fig.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
            ))
        
    def update_template(self, Template = 'simple_white'):
        self.fig.update_layout(template = Template)
    
    def write_image(self, figName, path):
        
        self.fig.write_image(f"{path}/{figName}.svg")  
        
    def addZoomSubPlot(self, zoom_x, zoom_y):
        # zoom_x and zoom_y are the x and y coordinates of the new zoom window.
        # zoom_x = [x1, x2]
        # zoom_y = [y1, y2]
        
        #Initialize Subplot 
        traces = self.fig.data #Take all traces from first figure
        
        self.fig = make_subplots(rows=1, cols=2)
        
        colorsG10 = ['#3366CC', '#DC3912', '#FF9900', '#109618', '#990099', '#0099C6', '#DD4477', '#66AA00', '#B82E2E', '#316395']
        color_i = 0
        # Add all traces back into both figures
        
        for trace in traces:
            if trace.line != None:
                trace.update(line=dict(color = colorsG10[color_i]))
                self.fig.add_trace(trace, row=1, col=1)
                self.fig.add_trace(trace, row=1, col=2)
                self.fig.data[-1].showlegend = False
                
            if color_i <= len(colorsG10):   
                color_i += 1
            else: 
                color_i = 0
                
        
        # Update zoom of subplot
        self.fig.update_xaxes(range=zoom_x, row=1, col=2)
        self.fig.update_yaxes(range=zoom_y, row=1, col=2)
        
        # Add box around zoom area
        self.addShadedBox(zoom_x, zoom_y, Row = 1, Col = 1)

   
    def addBox(self, box_x, box_y, Row=1, Col=1, scale_factor_x=1, scale_factor_y=1):
        
        box_X_scaled, box_Y_scaled = self.scale_rectangle(box_x, box_y, scale_factor_x,scale_factor_y)
        
        # Create the lines connecting the corners of the box to the corners of the second figure
        box_trace = go.Scatter(
            x=[box_X_scaled[0], box_X_scaled[0], box_X_scaled[1], box_X_scaled[1],box_X_scaled[0]],  # X coordinates for the lines
            y=[box_Y_scaled[0], box_Y_scaled[1], box_Y_scaled[1], box_Y_scaled[0],box_Y_scaled[0]], # Y coordinates for the lines
            mode='lines',
            line=dict(color='black', width=1),
            name = 'Zoomed Area' # Customize the line color, width, and style
        )
        self.fig.add_trace(box_trace, row = Row, col = Col) # Add box trace to original figure.
        
    def addShadedBox(self, box_x, box_y, Row=1, Col=1, scale_factor_x=1, scale_factor_y=1):
    
        box_X_scaled, box_Y_scaled = self.scale_rectangle(box_x, box_y, scale_factor_x,scale_factor_y)
        
        shape = go.layout.Shape(
            type="rect",
            xref="x",
            yref="y",
            x0=box_X_scaled[0],
            y0=box_Y_scaled[0],
            x1=box_X_scaled[1],
            y1=box_Y_scaled[1],
            fillcolor="lightblue",
            opacity=0.3,
            line=dict(color='black', width=1)
            
        )    
        
        self.fig.add_shape(shape,layer='below')
        
        self.addBox(box_x,box_y, Row=Row, Col=Col, scale_factor_x=scale_factor_x, scale_factor_y=scale_factor_y)
     
    def addLineShape(self, line_x, line_y, Row=1, Col=1):
        
        
        self.fig.add_shape(type="line",
                      x0=line_x[0], y0=line_y[0], x1=line_x[1], y1=line_y[1],
                      row=1, col=2,
                      line=dict(color="red", width=2))    
    
    def zoom(self, zoom_x, zoom_y, Row = 1, Col = 1):
        self.fig.update_xaxes(range=zoom_x, row=Row, col=Col)
        self.fig.update_yaxes(range=zoom_y, row=Row, col=Col)
    
    
    def show(self):
        self.fig.show()
        return
    
    def figText(self, text):

        LaTeXText = '$\\text{' + text + ' }$'

        return LaTeXText
    
    def scale_rectangle(self, x_coords, y_coords, scale_factor_x, scale_factor_y):
        x1 = x_coords[0]
        x2 = x_coords[1]
        y1 = y_coords[0]
        y2 = y_coords[1]
        
        # Calculating the center of the rectangle
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
    
        # Calculating the width and height of the rectangle
        width = abs(x2 - x1)
        height = abs(y2 - y1)
    
        # Scaling up the rectangle
        new_width = width * scale_factor_x
        new_height = height * scale_factor_y
    
        # Calculating the new coordinates of the rectangle
        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2
        
        new_x_coords = [new_x1, new_x2]
        new_y_coords = [new_y1, new_y2]
    
        return new_x_coords, new_y_coords   