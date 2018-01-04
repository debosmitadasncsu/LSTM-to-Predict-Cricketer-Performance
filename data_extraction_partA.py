# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:19:43 2017

"""
import pandas as pd


def extract_runs(filename,team,player):
  
  Dict = yaml.load(open(filename))
  #check team
  flag=False
  teams=Dict['info']['teams']
  if  'winner' not in Dict['info']['outcome']:
    return 0,False,0,0,0,0
  if team not in teams:
    return 0,False,0,0,0,0
  
  if (Dict['innings'][0]['1st innings']['team']==team): 
    a=Dict['innings'][0]['1st innings']['deliveries']
  else:
    a=Dict['innings'][1]['2nd innings']['deliveries']

  fours=0
  sixes=0
  runs=0
  balls=0
  sr=0
  for ball in a:
    #print(ball)
    for delivery in ball:
      if (ball[delivery]['batsman']==player):
        flag=True
        balls+=1
        #print(ball[delivery]['runs']['batsman'])
        runs+=int(ball[delivery]['runs']['batsman'])
        if ball[delivery]['runs']['batsman']==4:
          fours+=1
        if ball[delivery]['runs']['batsman']==6:
          sixes+=1
          
  if(flag):          
    fours=fours*4
    sixes=sixes*6
    if runs>0:
      fours = fours/runs
      sixes = sixes/runs
    sr = runs/balls
  return runs,flag,Dict['info']['dates'],fours,sixes ,sr 

import yaml
import os
from matplotlib import pyplot
result = pd.DataFrame(columns=['Runs','SR','Fours','Sixes','Date'])
team='Sri Lanka'
player='KC Sangakkara'

for filename in os.listdir('odis_male'):
  print(filename)
  if filename.endswith(".yaml"):
    runs,flag,date,fours,sixes,sr=extract_runs('odis_male/'+filename,team,player)
    if (flag):
      print(runs)

      result.loc[len(result)] = [runs,sr,fours,sixes,date,]
    #print(flag)

result = result.sort_values(['Date'], ascending=[1])
del result["Date"]
#Dict = yaml.load(open('odis_male/1059713.yaml'))


result.to_csv("score_"+player+".csv")
