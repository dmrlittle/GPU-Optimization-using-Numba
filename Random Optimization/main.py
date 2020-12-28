#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:18:41 2020

@author: mrlittle
"""

import random
import math
import Farm_Evaluator_Vec
import numpy as np
import pandas as pd
#from tqdm import tqdm 
import time
import optfarm
import multiprocessing
from numba import jit, cuda ,vectorize,int32, float64

powerCurve = pd.read_csv('power_curve.csv', sep=',', dtype = np.float32)
powerCurve = powerCurve.to_numpy(dtype = np.float32)


def gen2(seed):
        random.seed(seed)
        turb_coord=np.empty((50,2))
        turb_coord[0,:]=[random.triangular(50,3950),random.triangular(50,3950)]
        cnt=1
        while(cnt<50):
            p2=[random.triangular(50,3950),random.triangular(50,3950)]
            if(np.sqrt(np.sum(np.square(turb_coord[:cnt,:]-p2),axis=1)).min()>=400):
                turb_coord[cnt,:]=p2
                cnt+=1
        return turb_coord
    
def gen(seed):
        random.seed(seed)
        turb_coord=[]
        turb_coord.append([random.triangular(50,3950),random.triangular(50,3950)])
        while(len(turb_coord)<50):
            p2=[random.triangular(50,3950),random.triangular(50,3950)]
            for p1 in turb_coord:
                if(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) ) < 400):
                    break
            else:
                turb_coord.append(p2)
            #print(len(turb_coord),cnt)
        return np.asarray(turb_coord)
    
def spreader():
    windlist=[#'Wind_Data/wind_data_2007.csv',
              'Wind_Data/wind_data_2008.csv',
              'Wind_Data/wind_data_2009.csv',
              'Wind_Data/wind_data_2013.csv',
              'Wind_Data/wind_data_2014.csv',
              'Wind_Data/wind_data_2015.csv',
              'Wind_Data/wind_data_2017.csv']
    for windy in windlist:
        df = pd.read_csv(windy)
        wind_resource = df[['drct', 'sped']].to_numpy(dtype = np.float32)
        yield optfarm.mrlittle(powerCurve,wind_resource,windy)
        
def obj(i):
    return optfarm.getAEP(gen(i),turb_rad, power_curve, wind_inst_freq,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t),i
#@vectorize([float64(int32)], target='cuda')
def obj2(i):
    return optfarm.getAEP(gen(i),turb_rad, power_curve, wind_inst_freq,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)

@cuda.jit
def obj3(i):
    return optfarm.getAEP(gen(i),turb_rad, power_curve, wind_inst_freq,n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)


Farm_Evaluator_Vec.checkConstraints(gen(1), 100)

turb_rad, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,windy=0,0,0,0,0,0,0,0,0

def run():
    start=time.time()
    global turb_rad, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,windy
    for turb_rad, power_curve, wind_inst_freq, n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t,windy in spreader():
        for i,j in zip(range(500000,10000000,1000),range(501000,10000001,1000)):
            pool = multiprocessing.Pool(processes=8)
            output=np.array(pool.map(obj, range(i,j)))
            pool.close()
            np.save(windy[:-4]+'/'+str(j),output)
            end=time.time()
            print(windy[:-4]+" : Saved Outputs = > ",j," Took : ",end-start)

def timer():
    start=time.time()
    
    pool = multiprocessing.Pool(processes=8)
    output=np.array(pool.map(gen2, range(1000)))
    pool.close()
    
    end=time.time()
    print(end-start)
    
def timer():
    start=time.time()
    
    [gen2(i) for i in range(1000)]
    
    end=time.time()
    print(end-start)
    
#timer()


if __name__ == '__main__' :
    run()




"""
start=time.time()

for i in range(0,1000):
    obj(i)
    
end=time.time()
print(end-start)
"""
"""
processes=[]
start=time.time()
for i in range(1000):
    p=multiprocessing.Process(target=obj,args=(i,))
    processes.append(p)
    p.start()

for process in processes:
    process.join()
end=time.time()
print(end-start)
"""


"""
start=time.time()

for i in range(0,1000):
    obj2(i)
    
end=time.time()
print(end-start)
"""
"""
start=time.time()

for i in range(0,1000):
    obj3(i)
    
end=time.time()
print(end-start)
"""
