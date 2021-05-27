#!/usr/bin/python3
#-*- coding: utf-8 -*-


from type.sequence import sequence_t
from run_parameters import *
from model import train_new_model as tm_
import os
import numpy as np_
from task.get_feauture import GetFeatureMatrix
from analysis.death_time import CellDeath_t
from analysis.process_trajectories import process_traj
#%%

############# train a new segmentation model #############
   
if new_model.lower() == "cell_model" :
    
    tm_.Train_model(gray_rgb, cell_model)

elif new_model.lower() == "nucl_model" :
    
    tm_.Train_model(gray_rgb, nucl_model)
    
elif new_model.lower() == "cell_nucl_model" :
    
    tm_.Train_model(gray_rgb, cell_model)
    
    tm_.Train_model(gray_rgb, nucl_model)
    
           
############# sequence redaing #############
    
print("\nReading Sequence")

files= os.listdir(sequence_path)

class_prd=[]

for file in files :
    
    print(f"\nLoad ==> {file}\n")
    file_name=file.split('.')[0]
    if not os.path.exists("output/"+str(file_name)):
        os.makedirs("output/"+str(file_name))
        
    if not os.path.exists("output/"+str(file_name)+"/segmentation"):
        os.makedirs("output/"+str(file_name)+"/segmentation")
    
    if not os.path.exists("output/"+str(file_name)+"/tracking"):
        os.makedirs("output/"+str(file_name)+"/tracking")
    
    if not os.path.exists("output/"+str(file_name)+"/signal"):
        os.makedirs("output/"+str(file_name)+"/signal")
    
    if not os.path.exists("output/"+str(file_name)+"/features"):
        os.makedirs("output/"+str(file_name)+"/features")
    
    if sequence_path[-1] != "/" :
        sequence_path=sequence_path+"/"
        
    file_path= sequence_path+str(file)
    
    sequence = sequence_t.FromTiffFile(
        file_path,
        acq_seq,
        from_frame=from_frame,
        to_frame=to_frame, 
        n_channels=n_channels)
   
    

# organize channels
    
    sequence.OrganizeFrames(acq_seq,numerator, denominator)
    
#%%
############# channel alignement #############

    if n_channels >1:
        print("\nRegistration")
        sequence.RegisterChannelsOn(ref_channel.upper()) 
    
#%%    

   
##############   SEGMENTATION   #############
    
    print("\nSegmentation")
    
    sequence.SegmentCellsOnChannel( cell_channel, nucl_channel, cyto_seg, file_name)
    
    print(sequence)

#%%    

##############   TRAKING   #############
    
    print("Tracking") 
   
    
    dist_matrix=sequence.TrackCells(track_channel, bidirectional=bidirectional) # track cells

    sequence.PlotTracking(file_name) # plot the tracking graph
    
  
#%%       
        
##############   Feature Computation   #################
 
    print("\nFeature Computation")
    
    # fonction à simplifier, donner moins de choix mais propre
    
    intensities,features=sequence.ComputeCellFeatures(segm_protocol.lower(),same_channels,feature_channel,cyto_seg,
                                           cell_sig, nucl_sig, cyto_sig) # get all cell features







#%%
# statistical operations 
    
    signal=sequence.ComputeSignal(file_name,intensities,numerator, denominator, num_struct,denom_struct,
                              mean, median,var, ratio) 
    
       
   
#%%
    print("\nSave Features\n")
     

##############   SAVE FEATURES   #################

    
    for channel, types in intensities.items():
        for type_, frames in types.items():
            for idx,frame in enumerate (frames):
                if type(frame) != float :
                    sequence.WriteCellSignalEvolution(frame,file_name,channel,type_,idx)
                    
                
#%%   
    
    GetFeatureMatrix(file_name, features, to_frame)


#%%
# cell labeling with the corresponding unique identifier 
    
    print("\nCell Labeling") 
    
    sequence.CellLabeling(cell_channel.upper(),intensities,file_name,features,feature_channel.upper())  
        
    

    
                      
                        
                        
                        
                        
                        