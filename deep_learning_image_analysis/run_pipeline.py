#!/usr/bin/python3
#-*- coding: utf-8 -*-


from type.sequence import sequence_t
from run_parameters import *
from model import train_new_model as tm_
import os
import numpy as np_
import pickle

#model_name="C:/Users/ch_95/Desktop/live-cell_image_processing/deep_learning_image_analysis/cell_death_RF_3classes_weighted_model.sav"
#cell_death_model = pickle.load(open(model_name, 'rb'))

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
   
    
#%%
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

    #sequence.PlotTracking(file_name) # plot the tracking graph
    
  
#%%       
        
##############   Feature Computation   #################
 
    print("\nFeature Computation")
    
    intensities,features=sequence.ComputeCellFeatures(segm_protocol.lower(),same_channels,feature_channel,cyto_seg,
                                           cell_sig, nucl_sig, cyto_sig) # get all cell features







#%%
# statistic operations 
    
    signal=sequence.ComputeSignal(file_name,intensities,numerator, denominator, num_struct,denom_struct,
                              mean, median,var, ratio)
    
    
#%%
# cell labeling with the corresponding unique identifier 
    
    print("\nCell Labeling") # à revoir (rendre moins spécifique)
    
    sequence.CellLabeling(cell_channel.upper(),intensities,uid,file_name,features,feature_channel.upper())   
    
       
   
#%%
    print("\nSave features")
     

##############   SAVE FEATURES   #################

    
    for channel, types in intensities.items():
        for type_, frames in types.items():
            for idx,frame in enumerate (frames):
                if type(frame) != float :
                    sequence.WriteCellSignalEvolution(frame,file_name,channel,type_,idx)
                    
                
#%%   
    
    # sequence.WriteFeatureEvolution(features,file_name,type_,to_frame)
    

#%%
# get training matrix for cell phenotype classification (soit généraliser / soit mettre dans helpers)
    
    training_matrices,feature_names=sequence.GetTrainingMatrix(from_frame, to_frame,file_name,features,signal)
    
#%%  
    
import matplotlib.pyplot as plt

plt.plot(signal["RATIO"]["cell"]["mean"])    
    
    
    
# Cell phenotype prediction
    
#    pred=np_.empty((training_matrices[0].shape[0],1)) 
#    for idx,matrix_ in enumerate (training_matrices):
#        
#        if idx != 0 :
#            mask = np_.all(np_.isnan(matrix_) , axis=1) 
#            id_nan=np_.where(np_.all(np_.isnan(matrix_) , axis=1))
#            pred[id_nan]=np_.NaN 
#            matrix_=matrix_[~mask] 
#            #matrix_[id_nan]=0
#            prediction=cell_death_model.predict(matrix_)
#            
#            for id_, val in enumerate(prediction):
#                if pred[id_] != np_.NaN :
#                    pred[id_]=val
#
#    class_prd.append(pred)

        
    

    
                      
                        
                        
                        
                        
                        