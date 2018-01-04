# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:19:43 2017

"""
count=0

def get_runrate(team_deliveries):
  team_sum=0
  for ball in team_deliveries:
    #print(ball)
  
    for delivery in ball:
      team_sum+=ball[delivery]['runs']['total']
  print(team_sum)
  if delivery - int(delivery) > 0.6:
    delivery=int(delivery)+1
  else:
    delivery= int(delivery) + (delivery - int(delivery))/0.6
  print(delivery )
  rr=team_sum/delivery
  return rr

def get_runs(team_deliveries):
  team_sum=0
  for ball in team_deliveries:
    #print(ball)
  
    for delivery in ball:
      team_sum+=ball[delivery]['runs']['total']
  print(team_sum)

  return team_sum
  
def extract_runrates(filename,team1):
  
  Dict = yaml.load(open(filename))
  team2=""
  #check team
  flag=False
  teams=Dict['info']['teams']
  
  if team1  not in teams or 'winner' not in Dict['info']['outcome']:
    return 0,0,flag
  flag=True
  Dict = yaml.load(open(filename))
  if Dict['innings'][0]['1st innings']['team']==team1:
    team1_deliveries=Dict['innings'][0]['1st innings']['deliveries']
    team2_deliveries=Dict['innings'][1]['2nd innings']['deliveries']
    rr1=get_runs(team1_deliveries)/50.0
    rr2=get_runrate(team2_deliveries)
  else:
    team2_deliveries=Dict['innings'][0]['1st innings']['deliveries']
    team1_deliveries=Dict['innings'][1]['2nd innings']['deliveries'] 
    rr1=get_runrate(team1_deliveries)
    rr2=get_runs(team2_deliveries)/50.0    

  
  if  'winner' in Dict['info']['outcome']:
    print(Dict['info']['outcome']['winner'])
   
  return rr1-rr2,Dict['info']['dates'],flag


import yaml
import os
from matplotlib import pyplot

import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

team1='England'
result = pd.DataFrame(columns=['Difference','Date'])
for filename in os.listdir('odis_male'): #["1027317.yaml"]:#
  #print(filename)
  if filename.endswith(".yaml"):
    print(filename)
    rrdiff,date,flag=extract_runrates('odis_male/'+filename,team1)
    print(rrdiff)
    if flag:
      result.loc[len(result)] = [rrdiff,date]
      
result = result.sort_values(['Date'], ascending=[1])
del result["Date"]
      
result.to_csv("runrate_"+team1+".csv")    
    
