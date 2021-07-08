#!/usr/bin/python3
#-*- coding: utf-8 -*-

import numpy as np_
from scipy.signal import medfilt
from analyse import ComputeHistStats
import filters as flt_
import pattern_matching as ptmt

    

def Predict_Cell_Events ( pol_traj, div_entro_thr, div_var_thr,death_entro_thr, death_var_thr ):
    
    # compute stats
    var,entro = ComputeHistStats(pol_traj)
    
   
    # smooth 
    smth_entro = medfilt(medfilt(entro, 15),15)
    smth_var = medfilt(medfilt(var, 15),15)
    
    # Get divison frames
    div_frms = Predict_Division_Times(smth_entro, smth_var, div_entro_thr, div_var_thr)
    
    if len(div_frms) > 0:
        
        last_div_frm = div_frms [-1]
        
    else :
        
        last_div_frm = None
    
    # Get death frame
    death_frm = Predict_Death_Time(smth_entro, smth_var, death_entro_thr, death_var_thr, last_div_frm)
    
    return div_frms, death_frm




def Predict_Division_Times ( smth_entro, smth_var,div_entro_thr, div_var_thr, track_div ) :
    
   
    div_frms = []
    
    ### Divison filters
    
    # create filters 
    
    trg_inv = - flt_.Triangle(0, 1, 60, 1.5/2.0, 0.5)
    trg = flt_.Triangle(0, 1, 60, 1.5/2.0, 0.5)
    
    # match fiter to stat
    
    mt_entro = ptmt.match_template(smth_entro, trg_inv, normalized= False, pad_input=True, mode="edge")
    mt_var = ptmt.match_template(smth_var, trg, normalized= False, pad_input=True, mode="edge")
    
    # search for the max matching 
    
    max_mt_entro = max(mt_entro)
    idx_max_mt_entro = np_.where(mt_entro == max(mt_entro))
    
    max_mt_var = max(mt_var)
    idx_max_mt_var = np_.where(mt_entro == max(mt_var))
    
    # division time 
    if max_mt_entro > div_entro_thr and max_mt_var > div_var_thr : 
        
   
    
        div_frm = idx_max_mt_entro
    
        div_frms.append(div_frm)
        
    return div_frms

    



def Predict_Death_Time (feature_name, feature, first_frm, last_frm, thr , last_div_frm ):

       

    smth_feature = medfilt(medfilt(feature, 15),15)
   
    # Analyse the trajectory after the last division event detected 
    # if no divison event, last_frm = 0  
    
    if last_div_frm != 0 : 
        
        smth_feature = smth_feature[last_div_frm : ]
        
        
    # create filters
    if feature_name == "entropy" :
        
        filter_ = flt_.Sigmoid_Inv(-100, 100, 25, 1.5)
    
    elif feature_name == "variance" :
        
        smth_feature = smth_feature / 1e10 # /1e10 to have lower values 
        filter_ = flt_.Sigmoid(-100, 100, 25, 1.5) 
    

    
    # Matching
    
    match = ptmt.match_template(smth_feature, filter_, normalized= False, pad_input=True, mode="edge")
    
    
    # search for the max matching 
    
    max_match= max(match)
    idx_max_mt = np_.where(match == max_match)[0][0]
    
    
    # death time 
    
    if max_match > thr : 
        
    
        dth_frm = last_div_frm + idx_max_mt
        
       # remove incomplete events occuring at the start or at the end of the experiment
        if dth_frm > first_frm and dth_frm < last_frm  : 
                
              return dth_frm
          
        else :
             
             return  None
         
    else : 
        
        return None
   
    
    
    
    
    
    
    
    