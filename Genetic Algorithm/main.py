#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:26:43 2020

@author: mrlittle
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:01:45 2020

@author: mrlittle
"""


from numba import cuda
from numba.cuda.random import *
import numpy as np
import pandas as pd
#import timeit
#import random
#import cupy as cp
import math
import time
#import cuda_farm_eval as cudaeval
#import optfarm
from tqdm import tqdm 


import cudaevalv2

windyr=['Wind_Data/wind_data_2007.csv',
        'Wind_Data/wind_data_2008.csv',
        'Wind_Data/wind_data_2009.csv',
        'Wind_Data/wind_data_2013.csv',
        'Wind_Data/wind_data_2014.csv',
        'Wind_Data/wind_data_2015.csv',
        'Wind_Data/wind_data_2017.csv']
turb_diam,power_curve,wind_inst_freq = cudaevalv2.mrlittle('power_curve.csv',windyr)

@cuda.jit(device=True)
def cudarand(rng_states,ind):
    return (xoroshiro128p_uniform_float32(rng_states,ind)*10000%3901)+50,(xoroshiro128p_uniform_float32(rng_states,ind)*10000)%3901+50

@cuda.jit(device=True)
def testing():
    p=0

@cuda.jit
def cudagen(rng_states, array, turb_diam, power_curve, wind_inst_freq,mul,aep):
    i,j,k= cuda.grid(3)
    index=k*pow(mul,2)+j*pow(mul,1)+i
    turb_coords=array[index,:,:2]
    cnt=1
    
    turb_coords[0,:]=cudarand(rng_states,index)
    while(cnt<50):
        p21,p22=cudarand(rng_states,index)
        for p11,p12 in turb_coords[:cnt,:]:
            if(math.sqrt( ((p11-p21)**2)+((p12-p22)**2) ) < 400):
                break
        else:
            turb_coords[cnt,:]=p21,p22
            cnt+=1
            
    aep[index,0]=cudaevalv2.totalAEP(array[index,:,:], turb_diam, power_curve, wind_inst_freq)
    aep[index,1]=index


@cuda.jit(device=True)
def checker(array):
    for i,j in array:
        if(50<i<3950 and 50<j<3950):
            pass
        else:
            return False
    p11,p12=array[0,0],array[0,1]
    for i in range(1,50):
        if(math.sqrt( ((p11-array[i,0])**2)+((p12-array[i,1])**2) ) <= 400):
            return False
    return True

def checkeru(array):    
    for i,j in array:
        if(50<i,j<3950):
            continue
        else:
            return False
    p11,p12=array[0,0],array[0,1]
    for i in range(1,50):
        if(math.sqrt( ((p11-array[i,0])**2)+((p12-array[i,1])**2) ) < 400):
            return False
    return True

@cuda.jit
def cudagenv2(rng_states, array_, turb_diam, power_curve, wind_inst_freq,mul,aep):
    i,j,k= cuda.grid(3)
    ind=k*pow(mul,2)+j*pow(mul,1)+i
    array=array_[ind,:,:2]
    
    crsovrpnt1=int(xoroshiro128p_uniform_float32(rng_states,ind)*100)%50
    crsovrpnt2=int(xoroshiro128p_uniform_float32(rng_states,ind)*100)%50
    
    if(crsovrpnt1>crsovrpnt2):
        crsovrpnt1,crsovrpnt2=crsovrpnt2,crsovrpnt1
    
    for i in range(crsovrpnt1,crsovrpnt2):
        array[i,0]+=(xoroshiro128p_uniform_float32(rng_states,ind)-0.5)
        array[i,1]+=(xoroshiro128p_uniform_float32(rng_states,ind)-0.5)        
    if(checker(array)):
        for i in range(0,7,2):
            aep[ind,0]+=cudaevalv2.totalAEP(array_[ind,:,:], turb_diam, power_curve, wind_inst_freq[i,:,:])
        aep[ind,0]/=4
        aep[ind,1]=ind
    else:
        aep[ind,0]=0
        aep[ind,1]=ind

def indexgen(bpgx,bpgy,bpgz,tpbx,tpby,tpbz,mul):    
    array=np.zeros((10000000))
    @cuda.jit
    def indexgen_(array):
        i,j,k=cuda.grid(3)
        index=k*pow(mul,2)+j*pow(mul,1)+i
        array[index]=1
    indexgen_[(bpgx,bpgy,bpgz),(tpbx,tpby,tpbz)](array)
    x=np.array(np.where(array == 1))
    return x

def savegen(seed,index):
    rng_states=create_xoroshiro128p_states(100000, seed=seed)
    array=np.zeros((50,2),dtype=np.float32)
    @cuda.jit()
    def savegen_(array,rng_states,index):    
        turb_coords=array[:,:2]
        cnt=1
    
        turb_coords[0,:]=cudarand(rng_states,index)
        while(cnt<50):
            p21,p22=cudarand(rng_states,index)
            for p11,p12 in turb_coords[:cnt,:]:
                if(math.sqrt( ((p11-p21)**2)+((p12-p22)**2) ) < 400):
                    break
            else:
                turb_coords[cnt,:]=p21,p22
                cnt+=1
    savegen_[1,1](array,rng_states,index)
    pd.DataFrame(array).to_csv('submit.csv',index=False,header=['x','y'])
    return array



def adamgen():
    aep=np.load('adamaep.npy')
    array=np.zeros((120000,50,2),dtype=np.float32)
    for idx,(seed,ind) in enumerate(aep):
            if(idx == 12000):
                break
            array[idx,:,:]=savegen(seed,ind)
    return array

if(__name__ != '__main__'):
        gencnt = 100000
        array=np.zeros((gencnt,50,5),dtype=np.float32)
        aep=np.zeros((gencnt,2))
    
        mul = 100
        tpbx,tpby,tpbz=10,10,1
        bpgx,bpgy,bpgz=10,10,10
        tt=(tpbx*tpby*tpbz)*(bpgx*bpgy*bpgz)
    
        assert tpbz*bpgz <= tpby*bpgy <= tpbx*bpgx <= mul , 'Incorrect Multiplicative !'
        assert tt == gencnt , 'Unwanted Threads initiated !'
        assert np.any(indexgen(tpbx, tpby, tpbz, bpgx, bpgy, bpgz,mul) == range(gencnt)) , 'Incorrect Indexing!' 
    
        print('Kernel launch: cudagen ({} gen/launch) / Estimated ({} sec/launch)'.format(gencnt,(gencnt/10000)*33))
    
        startseed = int(input("Starting seed => "))
        endseed = int(input("Ending seed => "))

        print('Selected wind year :',windyr)
        choice = input("Takes {} seconds to run, Do you want continue? ".format((endseed-startseed)*330))
        assert choice == 'y' or choice == 'Y' , 'Operation Cancelled !'
    
        for i in range(startseed,endseed):
        
            start=time.time()
        
            rng_states=create_xoroshiro128p_states(tt, seed=i)
            cudagen[(bpgx,bpgy,bpgz),(tpbx,tpby,tpbz)](rng_states, array, turb_diam, power_curve, wind_inst_freq,mul,aep)
        
            end=time.time()
            print('Updated {} array in {} seconds'.format(tt*i,end-start))
            randy=np.random.randint(gencnt)
            print('Random Constrain Check with number ',randy)
            cudaevalv2.checkConstraints(array[randy,:,:2], 100 )
            
            np.save('genbackup{}/seed_{}_gencnt_{}'.format(windyr[22:24],i,gencnt),aep)
            
if(__name__ == '__main__'):
        gencnt = 4000
        aep=np.zeros((gencnt,2))
        aepp=np.zeros((12000,2))
        array=np.zeros((12000,50,5))
        
        mul = 100
        tpbx,tpby,tpbz=10,10,1
        bpgx,bpgy,bpgz=10,12,1
        tt=(tpbx*tpby*tpbz)*(bpgx*bpgy*bpgz)

        wind_inst_freq_=np.zeros((7,36,15),dtype=np.float32)
        for i in range(7):
            wind_inst_freq_[i,:,:]=wind_inst_freq[i]
        #assert tpbz*bpgz <= tpby*bpgy <= tpbx*bpgx <= mul , 'Incorrect Multiplicative !'
        #assert tt == gencnt , 'Unwanted Threads initiated !'
        #assert np.any(indexgen(tpbx, tpby, tpbz, bpgx, bpgy, bpgz,mul) == range(gencnt)) , 'Incorrect Indexing!'
        
        assert input("Continue? ")=='y' , 'Bye'
        
        startseed = int(input("Starting year => "))
        endseed = int(input("Ending year => "))
        print("Sending adam to earth......")
        #rng_states=create_xoroshiro128p_states(tt, seed=0)
        #cudagen[(bpgx,bpgy,bpgz),(tpbx,tpby,tpbz)](rng_states, array, turb_diam, power_curve, wind_inst_freq,mul,aepp)
        array[:12000,:,:2]=np.load('adam.npy')[:12000,:,:]
        mul = 100
        tpbx,tpby,tpbz=10,10,1
        bpgx,bpgy,bpgz=10,4,1
        tt=(tpbx*tpby*tpbz)*(bpgx*bpgy*bpgz)
        
        
        array=array[aepp[:,0].argsort()]
        aepp=aepp[aepp[:,0].argsort()]
        
        print("Lucifer's mission completed")
        print(f'Year 0 Max - ',aepp[-1,0])
        for i in range(startseed,endseed):
            arraycpy=array[-4000:,:,:].copy()
            rng_states=create_xoroshiro128p_states(tt, seed=i)
            cudagenv2[(bpgx,bpgy,bpgz),(tpbx,tpby,tpbz)](rng_states, arraycpy, turb_diam, power_curve, wind_inst_freq_,mul,aep)
            array[:4000,:,:]=arraycpy
            aepp[:4000,:]=aep
            array=array[aepp[:,0].argsort()]
            aepp=aepp[aepp[:,0].argsort()]
            print(f'Year {i+1} Max - {aepp[-1,0]} Dead People - ',sum(aepp[:,0]==0.0))
            
        
        
        
    

