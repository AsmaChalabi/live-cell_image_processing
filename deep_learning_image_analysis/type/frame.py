#!/usr/bin/python3
#-*- coding: utf-8 -*-

from __future__ import annotations
from run_parameters import gray_rgb
from type.cell import cell_t
from type.nucleus import nucleus_t
from type.cytoplasm import cytoplasm_t
from task import segmentation as sg_, feature 
from run_parameters import segm_protocol
import numpy as np_
import skimage.feature as ft_
import skimage.measure as ms_
import skimage.morphology as mp_
import skimage.transform as tf_

import matplotlib.pyplot as pl_

class frame_t:
    
    def __init__(self) -> None:
        
        self.channel = ""  # Name of the channel of the frame
        self.time_point = -1  # 0 and up: time point of the frame
        self.size = None  # (height, width) of the frame
        self.contents = None  # numpy.ndarray of the frame contents
#        self.cells = None  # List of segmented cells
#        self.nuclei=None
#        self.cyto=None
        
    @classmethod
    def WithProperties(
        cls, channel: str, time_point: int, contents: np_.ndarray
    ) -> frame_t:
        
        """
        WithProperties method
        Initiates the frame properties
        """
        
        instance = cls()

        instance.channel = channel # channel name
        instance.time_point = time_point # the frame time point
        instance.size = contents.shape # frame size
        instance.contents = contents # frame contents
        
        
        return instance

    def ClearContents(self):
        
        """
        ClearContents method
        To free some memory up when all the cell properties are saved
        """
        
        self.contents = None  # To free some memory up 
        

    def RegisterOn(self, ref_frame: np_.ndarray) -> None: 
        
        """
        RegisterOn method (target frame, reference frame)
        - Calculates the between channel shift using the python "register_translation" method
        - Gets the transformation (translation) by the python "EuclideanTransform" method
        - Applies the transformation to the target frame using the python "warp" method
        """
        
        # calculate the shift
        shift = ft_.register_translation( 
            ref_frame,
            self.contents,
            upsample_factor=8,
            space="real",
            return_error=False,
            )
        
        # get the transformation: translation by the calculated shift
        translation = tf_.EuclideanTransform(translation=shift) 
        
        self.contents = tf_.warp(self.contents, translation) # warped image
        


    def GetProperties (self,frames,class_,cell_channel,segmentation_org: None):
        
        if class_ == "cell" :
            type_= cell_t
            self.cells = []
            
        elif class_ == "cyto" :
            type_= cytoplasm_t
            self.cyto = []
            
        elif class_ == "nucl" :
            type_= nucleus_t
            self.nuclei = []
            
        for time_point,mask in enumerate(segmentation_org) :
            
            # Segmentation labeling and cell instantiations
            labeled_sgm_org = mp_.label(mask, connectivity=1)  #Label connected regions:
                                                                   #Two pixels are connected when they are neighbors 
                                                                   #and have the same value              
            props_org = ms_.regionprops(labeled_sgm_org) # measure properties of labeled image regions 
            
            struct_list=[]
            for uid, prop in enumerate(props_org): 
               
                # cell properties
                coords_org = (prop.coords[:, 0], prop.coords[:, 1]) # get pixel coordinates
                
                structure = type_.WithProperties(uid, time_point, prop.centroid, coords_org)
                
                
                structure.features["area"] = prop.area # save the cell area in the features dictionary (see "cell" script)
                
                if class_== "cell":
                    structure.features["convex_area"]=prop.convex_area
                    structure.features["major_axis_length"]=prop.major_axis_length
                    structure.features["minor_axis_length"]=prop.minor_axis_length
                    structure.features["perimeter"]=prop.perimeter
                    structure.features["edge"]= np_.median(feature.Edginess(frames[cell_channel][time_point],
                                      coords_org,prop.centroid))
                    
                
                struct_list.append(structure)
                
            if class_ == "cell" :
               
                self.cells.append(struct_list) # save the cell properties 
            elif class_ == "cyto" :
                self.cyto.append(struct_list)
            elif class_ == "nucl" :   
                self.nuclei.append(struct_list)
        

    def SegmentObject(self,frames,cell_channel: None, nucl_channel: None, cyto_seg: bool, file_name:str) -> None:

        """
        SegmentCells method
        - Segments cells using the "CellSegmentation_Original" function
          from the "segmentation" script.
        - Gets the cell properties 
        """
      
        method = segm_protocol.lower()
        
        
        if method == "cell" and cell_channel != None:
            cell_channel=cell_channel.upper()
            segmentation = sg_.CellSegmentation(frames[cell_channel],gray_rgb,file_name) # call the "CellSegmentation" fct from 
                                                                                         # "segmentation" script
            frame_t.GetProperties(self,frames,"cell",cell_channel,segmentation)                                                                                     
            
        elif method == "nucl" and nucl_channel != None:
            nucl_channel=nucl_channel.upper()
            segmentation_org, segmentation_dil, segmentation_ero = sg_.NuclSegmentation(frames[nucl_channel],gray_rgb,file_name) # call the " NuclSegmentation" function from "segmentation script
            frame_t.GetProperties(self,frames,"nucl",cell_channel,segmentation_org) 
           
            
        elif method == "cell and nucl" and cell_channel != None and nucl_channel != None :
            cell_channel=cell_channel.upper()
            nucl_channel=nucl_channel.upper()
            
            # cell seg
            
            cell_segmentation = sg_.CellSegmentation(frames[cell_channel],gray_rgb,file_name) 
            frame_t.GetProperties(self,frames,"cell",cell_channel,cell_segmentation)  
            
            # nucl seg
            segmentation_org, segmentation_dil, segmentation_ero = sg_.NuclSegmentation(frames[nucl_channel],gray_rgb,file_name) # call the " NuclSegmentation" function from "segmentation script
            frame_t.GetProperties(self,frames,"nucl",cell_channel,segmentation_org) 
            
            if cyto_seg == True :
                #cyto seg
                cyto_segmentation = sg_.CytoSegmentation(cell_segmentation,segmentation_dil,file_name) 
                frame_t.GetProperties(self,frames,"cyto",cell_channel,cyto_segmentation) 
                  
        else:
            raise ValueError(f"{method}: Invalid segmentation method")



        

    def CellPositions(self, cells):
        
        """
        CellPositions method
        Puts the position of the cell centroid in the same vector 
        """
        
        if cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        positions = np_.empty((cells.__len__(), 2), dtype=np_.float64) # create an empty array

        for idx, cell in enumerate(cells):
            
                positions[idx, :] = cell.position # get the position of the cell centroid in the frame 
                
        

        return positions
    

    def CellSegmentation(self, binary: bool = False) -> np_.ndarray:
        
        """
        CellSegmentation function
        Gets the list of the segmented cell composing pixels
        """
        
        if self.cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        segmentation = np_.zeros(self.size, dtype=np_.uint16)

        if binary: # if binary segmentation
            for cell in self.cells: # for each cell in the segmented cells list
                segmentation[cell.pixels] = 1  #segmentation of pixels containing the current cell is = 1 
                                               #(all pixels containing cells in the segmentation array will = 1)
        else:
            for cell in self.cells:
                segmentation[cell.pixels] = cell.uid + 1 # 0 reserved for background
                                                         # for each pixel containing cell, 
                                                         # segmentation = uid of the cell

        return segmentation
