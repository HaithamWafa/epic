#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 21:32:41 2023

@author: Haitham S. Fawzi
"""
import numpy as np
#import random
def getRewards(power_budget, IPS_ref):
    
    N_act = 5
    N_freq = 8
    N_util = 16
    
    Fmax = 8 
    Umax = 16
    Amax = 8
    
   # IPS_ref = 0.5 #defining this reward function IPS_ref @ 50%
    IPS_max = 1.0 #normalized max IPS
    
   # power_budget = 0.8 #defining this reward function power budget @80%
    
    space = []  ### S
    actions = [] ### A
    next_space = [] ### S'
    rewards = [] ### R
    
    
    def fgen(N_freq):
        if N_freq <= 0:
            raise ValueError("cannot be less than 0")
      #  samples = [random.randint(1, Fmax) for i in range(N_freq)] ### previously randomized
        samples = np.linspace(1, Fmax, N_freq)
        return samples
    def ugen(N_util):
        if N_util <= 0:
            raise ValueError("cannot be less than 0")
    
       # samples =  [random.randint(1, Umax) for i in range(N_util)] ### previously randomized
        samples = np.linspace(1, Umax, N_util)
        return samples
    
    def agen(N_act):
        if N_act <= 0:
            raise ValueError("cannot be less than 0")
       # samples = [random.randint(-2, 2) for i in range(N_act)] ### previously randomized
        samples = np.linspace(-2, 2, N_act)
        return samples
    
    ### actions vector ###
    action_levels = np.array(agen(N_act))/Amax
    
    ### the observation vectors ###
    freq_levels = np.array(fgen(N_freq),dtype=np.float32)/Fmax
    util_levels = np.array( ugen(N_util),dtype=np.float32)/Umax
    
    
    # Generate all possible states in the observation space
    for freq in freq_levels:
        for util in util_levels:
            for action in action_levels:
                old_freq = freq
                new_freq = freq + action
                #clipping both sides
                if(new_freq>1):
                    new_freq=1.0
                elif(new_freq<=0):
                    new_freq=0.125
                    
                old_ips = old_freq * util
                old_power = util * (old_freq ** 2.5)
                new_ips = new_freq * util
                new_power = util * (new_freq ** 2.5)
                
                #F, UTIL, POWER, IPS
                old_state = [old_freq, util, old_power, old_ips]
                new_state = [new_freq, util, new_power, new_ips]
                
                #reward computation
                
                delta=abs(new_ips-IPS_ref)/IPS_max
                if(new_power>power_budget):
                    R=0
                else:
                    R=1-delta
                #bookkkeeping
                space.append(old_state)
                actions.append(action)
                next_space.append(new_state)
                rewards.append(R)
               
    space = np.array(space)
    next_space = np.array(next_space)
    actions = np.array(actions)
    rewards = np.array(rewards)
    return rewards




    