#!/usr/bin/python3
#-*- coding: utf-8 -*-

from __future__ import annotations
from task.bg_subtraction import ContrastNormalized, BlockBasedCroppedImage
from type.cell import cell_t
from task import feature as ft_, segmentation as sg_
from type.frame import frame_t
from type.tracks import tracks_t
from run_parameters import segm_protocol, cyto_seg, track_channel
import imageio as io_
import itk
import matplotlib.pyplot as pl_
import networkx as nx_
import numpy as np_
import scipy as sc_
import scipy.spatial.distance as dt_
from typing import Callable, Optional, Sequence, Tuple
from run_parameters import *
import os
import csv
from skimage import filters
from numpy import ones,vstack
from numpy.linalg import lstsq




class sequence_t:

    def __init__(self) -> None:
        #
        self.frames = {}  # Dictionary "channel" -> list of frames
        self.channel_content={} # Dictionnary to reorganize the frames of each channel
        self.tracking = None  # tracks


    @classmethod

    def FromTiffFile(
        cls,
        path,
        channel_names: Sequence[str],
        from_frame: int = 0,
        to_frame: int = 999999,
        n_channels: int=1,

    ) -> sequence_t:

        """
        FromTiffFile function
        Reads the frames and split channels.
        Gets the frames properties : channel name, time point, content and shape
        """
        #
        # Channel name == '___' => discarded
        #
        instance = cls()

        for name in channel_names:

            if name != "___":
                name= name.upper()
                instance.frames[name] = [] # create an empty list for each channel

        ch_idx = n_channels - 1 # channel index
        time_point = -1

        #get file path
        #frame_reader = io_.get_reader(path) # read the frame

        frame_reader=itk.imread(path)
        frame_reader=itk.array_from_image(frame_reader)#.astype(np_.uint8)

#        assert n_channels * (to_frame + 1) <= len(frame_reader) , \
#            f"Last frame {to_frame} outside sequence ({frame_reader.get_length()} frame(s))"

        for raw_frame in frame_reader:
            ch_idx += 1
            if ch_idx == n_channels: # if n_channels are read
                ch_idx = 0 # channel index is reset to 0
                time_point += 1 # time point increments by 1

            if time_point < from_frame:
                continue
            elif time_point > to_frame:
                break

            name = channel_names[ch_idx].upper() # get channel name

            if name != "___":

                print(f"Frame {name}.{time_point}")
                frame = frame_t.WithProperties(name, time_point, raw_frame) # get the frame properties using "frame" script

                instance.frames[name].append(frame) # save the frame properties

        return instance # return an instance of the frames properties


    def ComputeImageRatio (self, numerator, denominator ):
        
       list_im_ratio=[]
       numerator=numerator.upper()
       denominator= denominator.upper()
       for tp,num_frame in enumerate(self.channel_content[numerator]):
          
           denom_frame=self.channel_content[denominator][tp]
           
#           Ibg_num= ContrastNormalized(num_frame, (127, 127))
#           Ibg_denom= ContrastNormalized(denom_frame, (127, 127))
#           
#           list_im_ratio.append(Ibg_num/Ibg_denom)
           
           list_im_ratio.append(num_frame/denom_frame)
           
       self.channel_content["RATIO"]=list_im_ratio
       
       return self.channel_content
   
   
   
    def OrganizeFrames(self,channel_names: Sequence[str],numerator, denominator):

        for channel in channel_names:
            if channel != "___" :
                channel=channel.upper()

                list_frames=[]
                for frames_ in self.frames[channel]:
                    list_frames.append(frames_.contents)

                self.channel_content[channel]=list_frames
                
        channel_content= self.ComputeImageRatio(numerator, denominator)

        return channel_content
    
   
           

    def __str__(self) -> str:

        """
        __str__ method
        Makes a string containing informations about the segmented frames
        (channel name, image size, time points, number of cells per frame)
        to print at the end of segmentation
        """

        self_as_str = ""

        for channel, frames in self.frames.items():
            initial_time_point = frames[0].time_point
            self_as_str += (
                f"[{channel}]\n" # channel name
                f"    {frames[0].size[1]}x{frames[0].size[0]}x" # image size
                f"[{initial_time_point}..{frames[-1].time_point}]\n" # the first and the last time points
            )


            if segm_protocol == "cell" :
                for f_idx, f_cell in enumerate (self.cells): #(self.nuclei):
                    if f_cell is not None:
                        self_as_str += (
                            f"    {initial_time_point+f_idx}: c {f_cell.__len__()}\n" # time point and the number of cells in the frame
                        )

            elif segm_protocol == "nucl" or segm_protocol == " cell and nucl ":

                for f_idx, f_cell in enumerate (self.nuclei):
                    if f_cell is not None:
                        self_as_str += (
                            f"    {initial_time_point+f_idx}: c {f_cell.__len__()}\n" # time point and the number of cells in the frame
                        )

        return self_as_str # returns a string containing all the informations



    def RegisterChannelsOn(self, ref_channel: str) -> None:

        """
        RegisterChannelsOn function
        Aligns channels by calculating and correcting the between channel shift
        Calls the "RegisterOn" method from "frame" script.
        """

        channel_names = tuple(self.frames.keys()) # get channel names
        ref_channel_idx = channel_names.index(ref_channel) # get the reference channel index
        other_channel_idc = set(range(channel_names.__len__())) - {ref_channel_idx} # indices of the other channels

        ref_frames = self.frames[ref_channel] # get the reference frames

        for c_idx in other_channel_idc:

            floating_frames = self.frames[channel_names[c_idx]] # target channel
            for f_idx in range(ref_frames.__len__()):
                floating_frames[f_idx].RegisterOn(ref_frames[f_idx].contents) # get the between channel shift (see "frame" script)



    def BackgroundNormalization (self, channels,post_processing_fct: Callable = None):

        """
        BackgroundNormalization function
        Normalizes the background using a post processing function (defiened on the run script)
        """

        if post_processing_fct is not None:

            for channel_name in channels:
                #if channel_name != "___" :
                if channel_name == "cherry":
                    channel_name=channel_name.upper()

                    for frames_ in self.frames[channel_name]:
                        frames_.contents= post_processing_fct(frames_.contents)





#############   SEGMENTATION   #############


    def SegmentCellsOnChannel(self, cell_channel: None, nucl_channel: None, cyto_seg: bool, file_name:str) -> None:

        """
        SegmentCellsOnChannel function
        Segment cells on each channel
        by calling the "SegmentCells" method from the "frame" script.
        """
        
        frames=self.channel_content
        
        if cell_channel != None :
            self.cell_channel = cell_channel.upper()
        else: 
            self.cell_channel=None
        
        
        if nucl_channel != None :
            self.nucl_channel=nucl_channel.upper()
        else: 
            self.nucl_channel=None
        


        frame_t.SegmentObject(self,frames,cell_channel, nucl_channel, cyto_seg, file_name)





    def RootCells(self):

        """
        RootCells function
        Gets the segmented cells with all the cell properties
        """
        if segm_protocol == "cell" :
            if self.cell_channel is None:
                raise RuntimeError(
                    "Segmentation-related function called before segmentation"
                )

            root_cell= self.cells[0]

        elif segm_protocol== "nucl" or  segm_protocol == "cell and nucl" :

            if self.nucl_channel is None:
                raise RuntimeError(
                    "Segmentation-related function called before segmentation"
                )

            root_cell= self.nuclei[0]

        #return self.frames[self.cell_channel][0].cells # returns the segmented root cells properties (at time_point=0)
        #return self.nuclei[0]


        return root_cell



#############   TRACKING   #################

    def TrackCells(
        self,channel_class: str, bidirectional: bool = True, max_dist: float = np_.inf
    ) -> None:

        """
        TrackCells method
        The tracking could be bidirectional or not.
        If bidirectional: Track cells by finding the nearest neighbor of each current cell.
        When the track is found, it's added to the tracking graph
        """
        if channel_class == "cell":
            track_channel= self.cell_channel
            type_= self.cells

        elif channel_class == "nucl":
            track_channel= self.nucl_channel
            type_= self.nuclei

        elif track_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        self.tracking = tracks_t()  # call the track_t class from "tracks" script
        #frames = self.frames[track_channel]  # get the frames of the track channel

        for f_idx in range(1, type_.__len__()):
            prev_frame = type_[f_idx - 1] # previous frame
            curr_frame = type_[f_idx] # current frame

            # call the "CellPosition" method from "frame" script


            prev_pos = frame_t.CellPositions(self,prev_frame) # get the previous positions
            
            

            curr_pos = frame_t.CellPositions(self,curr_frame) # get the current position

            # calculate the spatial distance ( the shortest distance between prev_pos and curr_pos)
            all_dist = dt_.cdist(prev_pos, curr_pos) # calculate all distances between the previous and the current positions
                                                     # all_dist : is a matrix :all_dist [prev,curr] = distance between prev and curr


            for curr_cell in type_[f_idx]: # for current cell in the list of segmented cells

                curr_uid = curr_cell.uid # get the current cell unique identifier

                if len(all_dist[:, curr_uid]) != 0 :

                    prev_uid = np_.argmin(all_dist[:, curr_uid]) # get the previous cell unique identifier
                                                                 # coresponding to the identifier of the lowest distance over all the forward distances

                    dist= all_dist[prev_uid, curr_uid]


                    if dist <= max_dist:

                        if bidirectional:
                            #sym=symmetric
                            sym_curr_uid = np_.argmin(all_dist[prev_uid, :]) # get the identifier of lowest distance over all the backward distances

                            if sym_curr_uid == curr_uid:

                                # Note: a cell has a unique next cell due to bidirectional constraint
                                prev_cell = type_[f_idx-1][prev_uid] # get the previous cell
                                # add an edge on the graph between the previous and the current cell
                                self.tracking.add_edge(prev_cell, curr_cell)



                        else: # if not bidirectional
                            # previous cell
                            prev_cell = type_[f_idx-1][prev_uid]

                            # current cell
                            curr_pos=np_.array([[curr_cell.position[0]],[curr_cell.position[1]]])

                            self.tracking.add_edge(prev_cell, curr_cell)

        return all_dist

    def PlotTracking(self,file:str, show_figure: bool = True) -> None:
        """
        PlotTracking method
        Plots the tracking 3D graphs, using the "Plot" function of "tracks" script
        """
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")

        self.tracking.Plot(file ,show_figure=show_figure) # plot tracking



    def CellFeatureNames(self) -> tuple:
        """
        CellFeatureNames
        Get the cell feature names from the "features" dictionary
        """
        # get the cell feature names from features dictionary (use "cell" script)
        #return tuple(self.frames[self.segm_channel][0].cells[0].features.keys())
        return tuple(self.cells[0][0].features.keys())



    def OrganizeFeatures(self,features:dict)->None:

        #for type_, channels in features.items():
        
        for channel, types in features.items():
            
            for type_,root_cells in types.items():
                
                for idx, root_cell  in enumerate(root_cells):
                   
                    if len(root_cell)>0:
                        cells_pix=[]
                        cells=[]
                        for cell in root_cell:
                           
                            cells_pix.append(len(cell))
                            cells.append(cell)
    
                        signal=np_.empty((max(cells_pix),len(root_cell)))*np_.NaN
                        
                        
                        for col,cell_ in enumerate(cells) :
                            
                            signal[0:len(cell_),col]=cell_
                            
    
                        root_cells[idx]=signal
    
                    elif len(root_cell)==0:
                        root_cells[idx]= 1*np_.NaN

        return features



    def ComputeCellFeatures(self, protocol: str,same_channels:None, feature_channel:str,cyto_seg:bool,
                            cell_sig: None,
                            nucl_sig: None,
                            cyto_sig:None)-> None:

        """
        This fuction computes feature using (GetUniSignal) and (GetMultiSignals)
        functions.
        """

        if segm_protocol=="cell" and self.cell_channel is None :

            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
            
        elif segm_protocol=="cell and nucl" and self.cell_channel is None and self.nucl_channel is None :
            
             raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
            
            

        intensities={}

        features={}

        protocol=protocol.lower()

        if protocol == "cell" or protocol == "nucl" :  # if cellular or nuclear segmentation only >>> OK NO BUG           
            
            print("protocol cell")
            for channel in cell_sig:
                channel= channel.upper()
                feature_channel=feature_channel.upper()
               
                if channel== feature_channel :
                    signal, props= GetUniSignal(self,channel,feature_channel,protocol)
                    
                    intensities[channel]=signal
                    features= props
                    
                else : 
                    signal= GetUniSignal(self,channel,feature_channel,protocol)
                    intensities[channel]=signal


        elif protocol == "cell and nucl": # if cellular and nuclear segmentation >>>>>> OK NO BUG

            if same_channels != None:

                for channel in same_channels:
                    channel= channel.upper()
                    
                    if cyto_seg == True :
                        signal,props= GetMultiSignals(self,channel,channel,channel,feature_channel, # >>>> OK NO BUG
                                               "cell and nucl")                    
                    
                    if cyto_seg == False :
                        signal,props= GetMultiSignals(self,channel,channel,None,feature_channel,  # >>>> OK NO BUG
                                               "cell and nucl")
                    
                    intensities=signal
                    features = props


            elif same_channels == None:
                
                if len(cell_sig)==1 and len(nucl_sig)==1 : # if the signal to extract is from different channels (one channel per structure)
                                                                               
                    cell_sig= cell_sig[0].upper()
                    nucl_sig= nucl_sig[0].upper()
                    
                    if cyto_seg ==False:  # and the cyto segmentation is not performed
                        
                        if cyto_sig != None :
                            raise RuntimeError( "Cyto segmentation is not performed, cyto_sig must be None" )
                            
                  
                        signal,props= GetMultiSignals(self,cell_sig,nucl_sig,cyto_sig,feature_channel, # >>>> OK NO BUG
                                                      "cell and nucl")

                        intensities=signal
                        features = props                  


                    if cyto_seg == True and len(cyto_sig)==1 : # if cyto segmentation
                        
                        if cyto_sig is None :
                            raise RuntimeError( "Cyto signal channel is empty, see cyto_sig" )
                            
                            
                        cyto_sig= cyto_sig[0].upper()
                        
                        
                        signal,props= GetMultiSignals(self,cell_sig,nucl_sig,cyto_sig,feature_channel, # >>>> OK NO BUG
                                                   "cell and nucl")
                        intensities=signal
                        features = props
                       

#======================== 3EME PARTIE =========================================
                        
                elif len(cell_sig) >1 or len(nucl_sig)>1:

                    print ("len(cell_sig) >1 or len(nucl_sig)>1 ")

                    for channel in cell_sig:
                        
                        if cyto_sig == None:         # >>>>>>>>>>> OK !!
                        
                            if channel in nucl_sig:                            
    
                                print(channel)
                                channel= channel.upper()
                                signal,props= GetMultiSignals(self,channel,channel,None,feature_channel,
                                               "cell and nucl")

                                intensities[channel]= signal
                                #intensities[channel]=signal
                                features = props
                    
#==============================================================================

#======================== 4EME PARTIE =========================================
                                    
                        elif cyto_sig != None :          # >>>>>>>>>> OK !!!!!

                            print("cyto_sig != None")
                            if channel in cyto_sig and channel in cell_sig and channel not in nucl_sig:
                                print("cyto_sig != None")
                                print(channel)
                                channel= channel.upper()

                                signal= GetMultiSignals(self,channel,None,channel,feature_channel, # >>>> OK !
                                                              "cell and nucl")
                                #intensities[channel]=self.OrganizeFeatures(signal)
                                intensities=signal
                                #features = props
                                
                            elif channel in cyto_sig  and channel in nucl_sig and channel not in cell_sig:
                                print(channel)
                                channel= channel.upper()

                                signal= GetMultiSignals(self,None,channel,channel,None, # >>>> OK !
                                                              "cell and nucl")
                                #intensities[channel]=self.OrganizeFeatures(signal)
                                intensities=signal
                                #features = props
                                    
                            else:
                                print("else cyto_sig")
                                channel= channel.upper()
                                signal,props= GetMultiSignals(self,channel,channel,channel,feature_channel,
                                               "cell and nucl")
                                intensities[channel]=signal
                                #intensities=signal
                                features = props
                                    
#==============================================================================
                                                             
                        else:                        # >>>>>>>>>> OK !!!!
                            print ( "ELSE ")

                            if channel in cell_sig and channel not in nucl_sig:
                                
#                                channel= channel.upper()
#                                
#                                signal= GetMultiSignals(self,channel,channel, None, None,"cell and nucl")
#                                intensities[channel]=self.OrganizeFeatures(signal)

#                            elif channel in cell_sig:
                                
                                cell_channel=channel.upper()
                                cell_signal, props= GetUniSignal(self,cell_channel,feature_channel,"cell")

                                intensities[cell_channel]=self.OrganizeFeatures(cell_signal)
                                features = props

                            elif channel in nucl_sig and channel not in cell_sig: 
                                
                                nucl_channel=channel.upper()
                                nucl_signal, props= GetUniSignal(self,nucl_channel,feature_channel,"nucl")

                                intensities[channel]=nucl_signal
                            #intensities[channel]=cell_signal
                                features = props



        return self.OrganizeFeatures(intensities),features #props
        #return intensities,features




    def ComputeSignal(self,file:str, intensity: dict,
                     numerator:str, denominator: str, num_struct:str, denom_struct:str,
                     mean: bool, median: bool, variance: bool, ratio: bool):

        """
        This fuction computes statistics operations on the extracted signal
        """

        calc_sig={}



        for channel, types in intensity.items():
            for type_,frames in types.items():
                cell_num=[]

                for idx, frame in enumerate(frames):
                    if type(frame) != float :
                        cell_num.append(frame.shape[1])


        cell_num=max(cell_num)
        frame_num=len(frames)


        for channel, types in intensity.items():
            calc_sig[channel]={}
            for type_,frames in types.items():
                calc_sig[channel][type_]={}
                #med=np_.empty((frame_num,cell_num))*np_.NaN
                #avg=np_.empty((frame_num,cell_num))*np_.NaN
                
                med=np_.empty((cell_num,frame_num))*np_.NaN
                avg=np_.empty((cell_num,frame_num))*np_.NaN
                var=np_.empty((cell_num,frame_num))*np_.NaN

                #if median== True and mean ==False or mean== True and median == False:
                if median==True and mean ==False and variance ==False:
                    operation= "median"
                    calc_sig[channel][type_][operation]={}


                    for col, frame in enumerate(frames):
                        for idx in range(0,frame.shape[1]):

                            signal=frame[:,idx]
                            signal=signal[~ np_.isnan(signal)]

                            med[idx,col]= np_.median(signal)
                    sig=med

                elif mean==True and median==False and variance ==False:
                    operation="mean"
                    calc_sig[channel][type_][operation]={}


                    for col, frame in enumerate(frames):
                        for idx in range(0,frame.shape[1]):

                            signal=frame[:,idx]
                            signal=signal[~ np_.isnan(signal)]

                            avg[idx,col]= np_.mean(signal)

                    sig=avg


                elif variance==True and median==False and mean==False:
                    operation="variance"
                    calc_sig[channel][type_][operation]={}


                    for col, frame in enumerate(frames):
                        for idx in range(0,frame.shape[1]):

                            signal=frame[:,idx]
                            signal=signal[~ np_.isnan(signal)]

                            var[idx,col]= np_.var(signal)

                    sig=var

                    calc_sig[channel][type_][operation]=sig

                    if not os.path.exists("output/"+str(file)+"/signal/"+str(operation)+"_"+str(channel)):
                        os.makedirs("output/"+str(file)+"/signal/"+str(operation)+"_"+str(channel))

                    fname= "output/"+str(file)+"/signal/"+str(operation)+str(channel)+"/"+str(type_)+"_"+str(operation)+".csv"

                    np_.savetxt(fname, sig, delimiter=",")

                elif median == True and mean == True and variance== True:

                    if not os.path.exists("output/"+str(file)+"/signal/median_intensity"):
                        os.makedirs("output/"+str(file)+"/signal/median_intensity")

                    if not os.path.exists("output/"+str(file)+"/signal/mean_intensity"):
                        os.makedirs("output/"+str(file)+"/signal/mean_intensity")
                        
                    if not os.path.exists("output/"+str(file)+"/signal/var_intensity"):
                        os.makedirs("output/"+str(file)+"/signal/var_intensity")

                    calc_sig[channel][type_]["median"]={}
                    calc_sig[channel][type_]["mean"]={}
                    calc_sig[channel][type_]["variance"]={}


                    for col, frame in enumerate(frames):
                        if type(frame) != float :
                            for idx in range(0,frame.shape[1]):

                                signal=frame[:,idx]
                                signal=signal[~ np_.isnan(signal)]

                                med[idx,col]=np_.median(signal)
                                avg[idx,col]= np_.mean(signal)
                                var[idx,col]= np_.var(signal)
                                
                            calc_sig[channel][type_]["median"]=med
                            calc_sig[channel][type_]["mean"]=avg
                            calc_sig[channel][type_]["variance"]=var


                            fname1= "output/"+str(file)+"/signal/median_intensity/"+str(type_)+"_"+str(channel)+"_median.csv"
                            np_.savetxt(fname1, med, delimiter=",")

                            fname2= "output/"+str(file)+"/signal/mean_intensity/"+str(type_)+"_"+str(channel)+"_mean.csv"
                            np_.savetxt(fname2, avg, delimiter=",")

                            fname3= "output/"+str(file)+"/signal/var_intensity/"+str(type_)+"_"+str(channel)+"_var.csv"
                            np_.savetxt(fname3, avg, delimiter=",")


        if ratio== True :
            
            
            med_ratio=np_.empty((cell_num,frame_num))*np_.NaN
            mean_ratio=np_.empty((cell_num,frame_num))*np_.NaN
            
            if median==True and mean==False :

                ratio_mat=calc_sig[numerator.upper()][num_struct.lower()]["median"]/calc_sig[denominator.upper()][denom_struct.lower()]["median"]
                calc_sig["median_ratio"]=ratio_mat
                fname= "output/"+str(file)+"/signal/median_ratio_.csv"
                np_.savetxt(fname, ratio_mat, delimiter=",")

            elif mean == True and median==False:
                ratio_mat=calc_sig[numerator.upper()][num_struct.lower()]["mean"]/calc_sig[denominator.upper()][denom_struct.lower()]["mean"]
                calc_sig["mean_ratio"]=ratio_mat
                fname= "output/"+str(file)+"/signal/mean_ratio_.csv"
                np_.savetxt(fname, ratio_mat, delimiter=",")
                
                
            elif median== True and mean == True :
                ratio_med=calc_sig[numerator.upper()][num_struct.lower()]["median"]/calc_sig[denominator.upper()][denom_struct.lower()]["median"]
                calc_sig["median_ratio"]=ratio_med
                fname= "output/"+str(file)+"/signal/median_ratio_.csv"
                np_.savetxt(fname, ratio_med, delimiter=",")

                ratio_avg=calc_sig[numerator.upper()][num_struct.lower()]["mean"]/calc_sig[denominator.upper()][denom_struct.lower()]["mean"]
                calc_sig["mean_ratio"]=ratio_avg
                fname1= "output/"+str(file)+"/signal/mean_ratio_.csv"
                np_.savetxt(fname1, ratio_avg, delimiter=",")

        return calc_sig






    def CellFeatureEvolution(self, root_cell: cell_t, feature: str) -> list:

        """
        CellFeatureEvolution function
        Calculates the feature evolution of each cell between the previous and the current position.
        Use the "TrackContainingCell" method of the "tarcks" script.
        """

        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")
        # self.segm_channel is necessarily not None

        evolution = []


        track = self.tracking.TrackContainingCell(root_cell) # call the tracking method from "tracks" script
        if track is not None:
            current_piece = []
            for from_cell, to_cell in nx_.dfs_edges(track, source=root_cell): # get the current and the next cell positions idx

                if current_piece.__len__() == 0:
                    if feature in from_cell.features:
                        current_piece.append(
                            (from_cell.time_point, from_cell.features[feature]) # save the current cell position time point and features
                        )
                    else:
                        current_piece.append(
                            (from_cell.time_point, 0) # save the current cell position time point and features
                        )

                if feature not in to_cell.features:
                    if feature in from_cell.features:
                        current_piece.append((to_cell.time_point, from_cell.features[feature]))
                    elif feature not in from_cell.features:
                        current_piece.append((to_cell.time_point, 0))
                else:
                    current_piece.append((to_cell.time_point, to_cell.features[feature])) # save the next cell position time point and features

                if track.out_degree(to_cell) != 1: # if the number of edges pointing out of the node is # 1
                    evolution.append(current_piece) # save the current piece (time_point,feature)
                    current_piece = []

        return evolution



    def PlotCellFeatureEvolutions(
        self,
        cell_list: Sequence,
        feature: str,
        file: str,
        show_figure: bool = True,
    ) -> None:

        """
        This function plots the cell feature evolutions
        calculated by the "CellFeatureEvolution" function
        """

        figure = pl_.figure()
        axes = figure.gca() # get the current axes
        axes.set_title(feature) # set a title
        axes.set_xlabel("time points") # set the x label
        axes.set_ylabel("feature value") # set the y label
        plots = []
        labels = []
        colors = "bgrcmyk"


        for root_cell in cell_list:
            color_idx = root_cell.uid % colors.__len__()
            subplot = None

            for piece in self.CellFeatureEvolution(root_cell, feature): # get the feature evolution
                if not (isinstance(piece[0][1], int) or isinstance(piece[0][1], float)): # if the feature value isn't "int" or "float" type
                    break

                time_points = []
                feature_values = []
                for time_point, feature_value in piece: # get the time point and the feature value
                    time_points.append(time_point) # save the time point
                    feature_values.append(feature_value) # save the feature value

                subplot = axes.plot(
                    time_points, feature_values, colors[color_idx] + "-x"
                )[0] # plot f(time_points, feature_values)

            if subplot is not None:
                plots.append(subplot)
                labels.append(f"root_cell {root_cell.uid}") # get uid labels

        if plots.__len__() > 0: # if "plots" list isn't empty
            axes.legend(handles=plots, labels=labels, loc='lower right') # place the legend on the axes
            # save plot
            pl_.savefig(str(file)+"/plot_features/"+feature)
        else:
            pl_.close(figure)

        if show_figure:
            pl_.show()





    def WriteCellSignalEvolution(self, signal_list,file,channel,type_,idx) :


        # save features in csv file

        if not os.path.exists("output/"+str(file)+"/signal/total_intencity/"+str(type_)+"/"+str(channel)):
            os.makedirs("output/"+str(file)+"/signal/total_intencity/"+str(type_)+"/"+str(channel))

        fname= "output/"+str(file)+"/signal/total_intencity/"+str(type_)+"/"+str(channel)+"/Track_Signal_"+str(idx+1)+".csv"


        np_.savetxt(fname, signal_list, delimiter=",", fmt='%s')
        
        



    def WriteFeatureEvolution(self, features_dict,file,type_, to_frame):
            
      
        header1=["time_points", "uid", "position_X", "position_Y","area", "shift", "edge", "convex_area",
                "maj_axis_len","min_axis_len","perimeter" ]

        header2= ["time_points", "uid", "position_X", "position_Y","area"]
        
        for type_, features in features_dict.items():
            
            if not os.path.exists("output/"+str(file)+"/features/"+str(type_)):
                os.makedirs("output/"+str(file)+"/features/"+str(type_))
            
            for feature_name, feature_lists in features.items():
                
                
                
                for cell_idx, list_values in enumerate(feature_lists):
                   
                    len_list= len(list_values)
                    fname= "output/"+str(file)+"/features/"+str(type_)+"/Cell_"+str(cell_idx+1)+".csv"

                    with open(fname,"w", newline='') as f :
                       if type_ == "cell":
                           writer = csv.DictWriter(f, fieldnames=header1) 
                           writer.writeheader()
                       
                       else:
                           writer = csv.DictWriter(f, fieldnames=header2) 
                           writer.writeheader()
                       
                       if len_list>1 :
                           
                           for frame in range(0,len_list)  :
                               if frame <= to_frame:  # solution provisoire il faut revoir la seg et le tracking
                                   if type_ == "cell":
                                        if frame < len(list_values):
                                            
                                            line={"time_points":features["time_points"][cell_idx][frame], "uid":features["uid"][cell_idx][frame],
                                                   "position_X":features["position"][cell_idx][frame][0],
                                                   "position_Y":features["position"][cell_idx][frame][1],"area":features["area"][cell_idx][frame], 
                                                   "shift":features["shift"][cell_idx][frame], "edge":features["edge"][cell_idx][frame],
                                                   "convex_area":features["convex_area"][cell_idx][frame],
                                                   "maj_axis_len":features["major_axis_length"][cell_idx][frame],
                                                   "min_axis_len":features["minor_axis_length"][cell_idx][frame],
                                                   "perimeter":features["perimeter"][cell_idx][frame]}
                                         
                                            writer.writerow(line)
                                        
                                   else: 
                                        if frame < len(list_values):
                                            line={"time_points":features["time_points"][cell_idx][frame], "uid":features["uid"][cell_idx][frame],
                                                   "position_X":features["position"][cell_idx][frame][0],
                                                   "position_Y":features["position"][cell_idx][frame][1],"area":features["area"][cell_idx][frame]} 
                                                   
                                            
                                            writer.writerow(line)





    def CellLabeling(self,channel,singnal,uid,file, features,feature_channel):

        """
        CellLabeling function allows to identify cells in the original image,
        by labeling them with their cell unique identifier.
        """

        if self.cell_channel != None :

            name="segmentation_map"

            if not os.path.exists("output/"+str(file)+"/"+name):
                os.makedirs("output/"+str(file)+"/"+name)


        #        for idx,frame in enumerate (self.channel_content[channel]):
        #
        #            if idx % 10 == 0 :
        #                pl_.figure(figsize=(30,30))
        #                pl_.imshow(frame, cmap="gray")
        #
        #                for cell in range(0,len(features["nucl"]["uid"])):
        #
        #                    if uid=="root_cell_uid": # the cell is labeled with the same uid over the diferent frames
        #                        if len(features["nucl"]["uid"][cell])>0 and idx < len(features["nucl"]["position"][cell]):
        #                            pl_.text(features["nucl"]["position"][cell][idx][0],features["nucl"]["position"][cell][idx][1],
        #                                         str(features["nucl"]["uid"][cell][0]),color="red", fontsize=40)
        #
        #                    elif uid=="cell_uid": # the cell is labeled with the specfic frame uid, if the uid isn't available,
        #                                       # the cell is labeled with "x"
        #                       if len(features["nucl"]["position"][cell])-1>= idx:
        #                           pl_.text(features["nucl"]["position"][cell][0][0],features["nucl"]["position"][cell][0][1],
        #                                     str(features["nucl"]["uid"][cell][idx]),color="red", fontsize=40)
        #                       else:
        #
        #                           pl_.text(features["nucl"]["position"][cell][0][0],features["nucl"]["position"][cell][0][1],
        #                                 "x" ,color="red", fontsize= 40 )

        #idx=0
           
        for idx in range (len(self.channel_content[channel])):
            
            frame= self.channel_content[channel][idx]
            pl_.figure(figsize=(30,30))
            pl_.imshow(frame, cmap="gray")
    
            if track_channel == "nucl" :
                for cell in range(0,len(features["cell"]["uid"])):
                    if len(features["cell"]["uid"][cell])>idx :
#                        pl_.text(features["cell"]["position"][cell][idx][0],features["cell"]["position"][cell][idx][1],
#                                     str(features["cell"]["uid"][cell][0]),color="red", fontsize=50)
                        
                        pl_.text(features["cell"]["position"][cell][idx][0],features["cell"]["position"][cell][idx][1],
                                     str(cell),color="red", fontsize=50)
    
            
            
            elif track_channel == "cell" :
    
                for cell in range(0,len(features["cell"]["uid"])):
                    #print(cell)
                    if len(features["cell"]["uid"][cell])> idx: #0 :
                        pl_.text(features["cell"]["position"][cell][idx][0],features["cell"]["position"][cell][idx][1],
                                     str(features["cell"]["uid"][cell][0]),color="red", fontsize=50)


            # save figures
    
            path= "output/"+str(file)+"/"+name+"/frame"+str(idx)+".jpg"
            pl_.savefig(path)
            pl_.close()
            
        
        
        # ========= plot cell phenotype ==================================
        
#        for idx in range (len(self.channel_content[channel])):
#            
#            frame= self.channel_content[channel][idx]
#            pl_.figure(figsize=(30,30))
#            pl_.imshow(frame, cmap="gray")
#    
#            if track_channel == "nucl" :
#                for cell in range(0,len(features["cell"]["uid"])):
#                    if len(features["cell"]["uid"][cell])>idx :
##                        pl_.text(features["cell"]["position"][cell][idx][0],features["cell"]["position"][cell][idx][1],
##                                     str(features["cell"]["uid"][cell][0]),color="red", fontsize=50)
#                        
#                        pl_.text(features["cell"]["position"][cell][idx][0],features["cell"]["position"][cell][idx][1],
#                                     str(cell),color="red", fontsize=50)
    
        
        
        
        
        
        #=================================================================

        self.channel_content[channel]=None
        self.channel_content["CHERRY"]=None



    def GetTrainingMatrix(self,from_frame, to_frame,file,features,signal):
    
        if not os.path.exists("output/"+str(file)+"/features/cell_death_training"):
            os.makedirs("output/"+str(file)+"/features/cell_death_training")
            
        frame_number= to_frame-from_frame 
        
        feature_names=[]
        matrices=[]
        
        
        row=len(features["cell"]["area"])
        col=len(features["cell"])+4
        
        
        for frame_idx in range(0,frame_number+1):
            
            frame_vec=[]
            training_matrix=np_.empty((row,col))*np_.NaN 
            
            for feature_name, feature_list in features["cell"].items():
                
                if feature_name != "time_points" and feature_name != "pixels" and feature_name != "uid" and feature_name != "position":
                    
                    feature_names.append(feature_name)
                    vec=np_.empty((row))*np_.NaN
                 
                    for cell_idx, cell_features in enumerate(feature_list):
                            
                        if len(cell_features) >= 0 and len(cell_features) > frame_idx: # >=10 quand j'analyse >100 frames
                            
                            vec[cell_idx]=cell_features[frame_idx]
                                                        
                    frame_vec.append(vec)
                  
        
                    for col_idx, vector in enumerate(frame_vec):
                        
                        training_matrix[:,col_idx]=vector 
                        
                    mean_cfp=signal["CFP"]["cell"]["mean"]
                    mean_cfp=mean_cfp.T
                    training_matrix[:,-8]=mean_cfp[:,frame_idx]
                    
                    median_cfp=signal["CFP"]["cell"]["median"]
                    median_cfp=median_cfp.T
                    training_matrix[:,-7]=median_cfp[:,frame_idx]
                    
                    var_cfp=signal["CFP"]["cell"]["variance"]
                    var_cfp=var_cfp.T
                    training_matrix[:,-6]=var_cfp[:,frame_idx]
                    
                    mean_yfp=signal["YFP"]["cell"]["mean"]
                    mean_yfp=mean_yfp.T
                    training_matrix[:,-5]=mean_yfp[:,frame_idx]
                    
                    median_yfp=signal["YFP"]["cell"]["median"]
                    median_yfp=median_yfp.T
                    training_matrix[:,-4]=median_yfp[:,frame_idx]
                    
                    var_yfp=signal["YFP"]["cell"]["variance"]
                    var_yfp=var_yfp.T
                    training_matrix[:,-3]=var_yfp[:,frame_idx]
                    
                    mean_ratio=signal["mean_ratio"]
                    mean_ratio=mean_ratio.T
                    training_matrix[:,-2]=mean_ratio[:,frame_idx]
                    
                    median_ratio=signal["median_ratio"]
                    median_ratio=median_ratio.T
                    training_matrix[:,-1]=median_ratio[:,frame_idx]
                    
                    
                    
                    
                    
                    
#            mask = np_.all(np_.isnan(training_matrix) , axis=1) 
#            training_matrix=training_matrix[~mask]         
            matrices.append(training_matrix)
            fname= "output/"+str(file)+"/features/cell_death_training/cells_"+str(frame_idx)+".csv"
            
            np_.savetxt(fname,training_matrix, delimiter=",")
        
        feature_names.append("mean_intensity")
        feature_names.append("median_intensity")
        feature_names.append("var_intensity")
        feature_names.append("ratio")
               
            
        return matrices,feature_names




def GetMultiSignals(self,cell_channel: None, nucl_channel:None, cyto_channel: None, features_channel: None,
                  class_: str):

    features={} # features dictionary
    intensities={} # signal dictionary
    
    if cell_channel != nucl_channel :
        
        if cell_channel != None :
            intensities[cell_channel] ={}
            intensities[cell_channel]["cell"]=[]
            cell_type=self.cells
            cell_content= self.channel_content[cell_channel]
            
        if nucl_channel != None :  
            print(nucl_channel)
            intensities[nucl_channel]= {}
            intensities[nucl_channel]["nucl"]=[]
            nucl_type=self.nuclei
            nucl_content= self.channel_content[nucl_channel]
        
        if cyto_channel != None : 
            
            if cyto_channel != cell_channel and cyto_channel == nucl_channel or cyto_channel== cell_channel and cyto_channel != nucl_channel:            
                intensities[cyto_channel]["cyto"]=[]
                cyto_type=self.cyto
                cyto_content= self.channel_content[cyto_channel]

            if cyto_channel != cell_channel and cyto_channel != nucl_channel :
                intensities[cyto_channel]= {}
                intensities[cyto_channel]["cyto"]=[]
                cyto_type=self.cyto
                cyto_content= self.channel_content[cyto_channel]

    else :
        
        if cell_channel != None :
            intensities[cell_channel] ={}
            intensities[cell_channel]["cell"]=[]
            cell_type=self.cells
            cell_content= self.channel_content[cell_channel]
            
        if nucl_channel != None : 
            intensities[nucl_channel]= {}
            intensities[nucl_channel]["nucl"]=[]
            nucl_type=self.nuclei
            nucl_content= self.channel_content[nucl_channel]
        
        if cyto_channel != None :
            intensities[cyto_channel]= {}
            intensities[cyto_channel]["cyto"]=[]
            cyto_type=self.cyto
            cyto_content= self.channel_content[cyto_channel]
      
        
        

    if features_channel != None:
        
        if cell_channel != None :
            
            features["cell"]={}
            features["cell"]["time_points"]=[]
            features["cell"]["pixels"]=[]
            features["cell"]["uid"]=[]
            features["cell"]["position"]=[]
            features["cell"]["area"]=[]
            features["cell"]["edge"]=[]
            features["cell"]["shift"]=[]
            features["cell"]["convex_area"]=[]
            features["cell"]["major_axis_length"]=[]
            features["cell"]["minor_axis_length"]=[]
            features["cell"]["perimeter"]=[]
            


        if nucl_channel != None :
            features["nucl"]={}
            features["nucl"]["time_points"]=[]
            features["nucl"]["pixels"]=[]
            features["nucl"]["uid"]=[]
            features["nucl"]["position"]=[]
            features["nucl"]["area"]=[]

        if cyto_channel != None :

            features["cyto"]={}
            features["cyto"]["time_points"]=[]
            features["cyto"]["pixels"]=[]
            features["cyto"]["uid"]=[]
            features["cyto"]["position"]=[]
            features["cyto"]["area"]=[]


    for c_idx, root_cell in enumerate(self.RootCells()):

        track = self.tracking.TrackContainingCell(root_cell)
        
        if track is not None:
            
            leaves=[val for val in track.nodes() if track.out_degree(val)==0 and track.in_degree(val)==1]
            
            for leaf in leaves : 
                # shortest path : list of nodes that constitue the shortest path between the root cell and the leaf
                sub_track=nx_.shortest_path(track, source=root_cell, target=leaf) # LISTE
            
                if cell_channel != None :
                    cell_signal=[]
                
                if nucl_channel != None :
                    nucl_signal=[]
                
                if cyto_channel != None :
                    cyto_signal=[]
        
                if features_channel != None:
                    
                    if cell_channel != None :
                        
                        cell_tp=[]
                        cell_pixels=[]
                        cell_uid=[]
                        cell_position=[]
                        cell_area=[]
                        cell_edge=[]
                        cell_shift=[0]
                        cell_convex_area=[]
                        cell_major_axis_length=[]
                        cell_minor_axis_length=[]
                        cell_perimeter=[]
                        
                    if nucl_channel != None :
                        nucl_tp=[]
                        nucl_pixels=[]
                        nucl_uid=[]
                        nucl_position=[]
                        nucl_area=[]
        
                    if cyto_channel != None :
            
                        cyto_tp=[]
                        cyto_pixels=[]
                        cyto_uid=[]
                        cyto_position=[]
                        cyto_area=[]

                  
                for nucl_ in sub_track:  
            
                    if track.out_degree[nucl_]< 3:
                        
                        tp=nucl_.time_point
                        #to_tp=to_nucl.time_point
                        
                        if cell_channel != None :
                            cell_frame=cell_content[tp]
                            #to_cell_frame=cell_content[to_tp]
                        
                        if nucl_channel != None :
                            nucl_frame=nucl_content[tp]
                            #to_nucl_frame=nucl_content[to_tp]
    
                        cmpt=0
                        
                        for idx, cell_ in enumerate(cell_type[tp]) :
                            
                            cell_pix=set(zip(*cell_.pixels))
                            nucl_pix=set(zip(*nucl_.pixels))
                            
                            
                            if len(nucl_pix.intersection(cell_pix))>0 and nucl_.features["area"]>50 and cell_.features["area"]>100:
                            
                                cmpt+=1
                                
                                if cmpt== 1 :
                                    
                                    if cell_channel != None :
                                        cell_signal.append(cell_frame[cell_.pixels])
                                        
                                    if nucl_channel != None :
                                        nucl_signal.append(nucl_frame[nucl_.pixels])
        
                                    if features_channel != None:
                                        if cell_channel!= None :
                                            cell_pixels.append(cell_.pixels)
                                            cell_tp.append(tp)
                                            cell_uid.append(cell_.uid)
                                            cell_position.append(cell_.position)
                                            cell_area.append(cell_.features["area"])
                                            cell_edge.append(cell_.features["edge"])
                                            cell_convex_area.append(cell_.features["convex_area"])
                                            cell_major_axis_length.append(cell_.features["major_axis_length"])
                                            cell_minor_axis_length.append(cell_.features["minor_axis_length"])
                                            cell_perimeter.append(cell_.features["perimeter"])
        
                                        if nucl_channel != None :
                                            nucl_pixels.append(nucl_.pixels)
                                            nucl_tp.append(tp)
                                            nucl_uid.append(nucl_.uid)
                                            nucl_position.append(nucl_.position)
                                            nucl_area.append(nucl_.features["area"])
    
    
                                    if cyto_seg == "yes" and  cyto_channel != None:
                                        
                                         
                                        
                                        cyto_frame=cyto_content[tp]
                                        
                                        
                                        for id_,cyto_ in enumerate(cyto_type[tp]):
                                            
                                        
                                            
                                            cyto_pix= set(zip(*cyto_.pixels))
        
                                            if len(cell_pix.intersection(cyto_pix))>0 and cyto_.features["area"]>50:
        
                                                cyto_signal.append(cyto_frame[cyto_.pixels])
        
                                                if features_channel != None:
                                                    cyto_pixels.append(cyto_.pixels)
                                                    cyto_tp.append(tp)
                                                    cyto_uid.append(cyto_.uid)
                                                    cyto_position.append(cyto_.position)
                                                    cyto_area.append(cyto_.features["area"])
                                                    
                                       
           
        if cell_channel != nucl_channel :
            
            if cell_channel != None :
                
                intensities[cell_channel]["cell"].append(cell_signal)
                
            if nucl_channel != None :
                
                intensities[nucl_channel]["nucl"].append(nucl_signal)

            if cyto_channel != None :#and cyto != cell and cyto != nucl:
                
                intensities[cyto_channel]["cyto"].append(cyto_signal)

        else :
            
            if cell_channel != None :
                intensities[cell_channel]["cell"].append(cell_signal)
                
            if nucl_channel != None :
                intensities[nucl_channel]["nucl"].append(nucl_signal)

            if cyto_channel != None :
                intensities[cyto_channel]["cyto"].append(cyto_signal)

        if features_channel != None:
            #cell
            if cell_channel != None :
                features["cell"]["pixels"].append(cell_pixels)
                features["cell"]["time_points"].append(cell_tp)
                features["cell"]["uid"].append(cell_uid)
                features["cell"]["position"].append(cell_position)
                features["cell"]["area"].append(cell_area)
                features["cell"]["edge"].append(cell_edge)
                features["cell"]["convex_area"].append(cell_convex_area)
                features["cell"]["major_axis_length"].append(cell_major_axis_length)
                features["cell"]["minor_axis_length"].append(cell_minor_axis_length)
                features["cell"]["perimeter"].append(cell_perimeter)
                
                
                for i in range (1,len(cell_position)):  # LE METTRE DANS UNE FONCTION A PART
                   
                    prev_cell=np_.empty((1, 2))
                    curr_cell=np_.empty((1, 2))
                    prev_cell[0,:]= cell_position[i-1]
                    curr_cell[0,:]= cell_position[i]
                    cell_shift.append(np_.linalg.norm(curr_cell-prev_cell))
                    
                features["cell"]["shift"].append(cell_shift)

            #nucl
            if nucl_channel != None :
                features["nucl"]["pixels"].append(nucl_pixels)
                features["nucl"]["time_points"].append(nucl_tp)
                features["nucl"]["uid"].append(nucl_uid)
                features["nucl"]["position"].append(nucl_position)
                features["nucl"]["area"].append(nucl_area)

            #cyto
            if cyto_channel != None :
                features["cyto"]["pixels"].append(cyto_pixels)
                features["cyto"]["time_points"].append(cyto_tp)
                features["cyto"]["uid"].append(cyto_uid)
                features["cyto"]["position"].append(cyto_position)
                features["cyto"]["area"].append(cyto_area)


    if features_channel != None:

        return intensities,features

    else :
        return intensities



def GetUniSignal(self,channel: None, features_channel: None, class_: str):

    features={} # features dictionary
    intensities={} # signal dictionary
    
    if class_ == "cell":

        #class_=class_.lower()
        type_=self.cells

        #intensities[class_]=[]

    elif class_ == "nucl":

        #class_=class_.lower()
        type_=self.nuclei

        #intensities[class_]=[]

    elif class_ == "cyto":

        #class_=class_.lower()
        type_=self.cyto
        
        
        
    class_=class_.lower()
    intensities[class_]=[]
      
    content= self.channel_content[channel] # frame content  
        

    if features_channel != None and channel== features_channel :
        
        features[class_]={}
        features[class_]["time_points"]=[]
        features[class_]["pixels"]=[]
        features[class_]["uid"]=[]
        features[class_]["position"]=[]
        features[class_]["area"]=[]
        
        if class_=="cell":
            features[class_]["edge"]=[]
            features[class_]["shift"]=[]
            features[class_]["convex_area"]=[]
            features[class_]["major_axis_length"]=[]
            features[class_]["minor_axis_length"]=[]
            features[class_]["perimeter"]=[]
            


    for c_idx, root_cell in enumerate(self.RootCells()):

        track = self.tracking.TrackContainingCell(root_cell)
        
        if track is not None:
            
            leaves=[val for val in track.nodes() if track.out_degree(val)==0 and track.in_degree(val)==1]
            
            for leaf in leaves : 
                # shortest path : list of nodes that constitue the shortest path between the root cell and the leaf
                sub_track=nx_.shortest_path(track, source=root_cell, target=leaf) # LISTE
            
               
                signal=[]
                time_points=[]
                pixels=[]
                uid=[]
                position=[]
                area=[]
                edge=[]
                shift=[0]
                convex_area=[]
                major_axis_length=[]
                minor_axis_length=[]
                perimeter=[]
                        
                                  
                for node in sub_track:  
            
                    if track.out_degree[node]< 3:
                        
                        tp=node.time_point
                        
                        frame=content[tp]
                            
                            
                        if node.features["area"]>100 or node.features["area"]>50:
                            
                            if max(node.pixels[0]) < frame.shape[0] and max(node.pixels[1]) < frame.shape[1]:
                                
                                signal.append(frame[node.pixels])
                                    
        
                                if features_channel != None and channel== features_channel :
                                    
                                    pixels.append(node.pixels)
                                    time_points.append(tp)
                                    uid.append(node.uid)
                                    position.append(node.position)
                                    area.append(node.features["area"])
                                    
                                    
                                    if class_== "cell" :
                                        edge.append(node.features["edge"])
                                        convex_area.append(node.features["convex_area"])
                                        major_axis_length.append(node.features["major_axis_length"])
                                        minor_axis_length.append(node.features["minor_axis_length"])
                                        perimeter.append(node.features["perimeter"])
                                
        intensities[class_].append(signal)
            

        if features_channel != None and channel== features_channel:
             
            features[class_]["pixels"].append(pixels)
            features[class_]["time_points"].append(time_points)
            features[class_]["uid"].append(uid)
            features[class_]["position"].append(position)
            features[class_]["area"].append(area)
            
            if class_=="cell":
                features[class_]["edge"].append(edge)
                features[class_]["convex_area"].append(convex_area)
                features[class_]["major_axis_length"].append(major_axis_length)
                features[class_]["minor_axis_length"].append(minor_axis_length)
                features[class_]["perimeter"].append(perimeter)
                
                
                for i in range (1,len(position)):  # LE METTRE DANS UNE FONCTION A PART
                   
                    prev=np_.empty((1, 2))
                    curr=np_.empty((1, 2))
                    prev[0,:]= position[i-1]
                    curr[0,:]= position[i]
                    shift.append(np_.linalg.norm(curr-prev))
                
                features[class_]["shift"].append(shift)

           
    if features_channel != None and channel== features_channel:

        return intensities,features

    else :
        return intensities










































