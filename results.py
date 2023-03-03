#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 12:46:04 2023

@author: seanabrahamson
"""

import pandas as pd

coefficientDF_NoZeroVel = pd.read_pickle('./coefficientDF_NoZeroVel.pkl')
                                         
coefficientDF_Start = pd.read_pickle('./coefficientDF_Start.pkl')
                                         
coefficientDF_StartEnd = pd.read_pickle('./coefficientDF_StartEnd.pkl')
                                                  