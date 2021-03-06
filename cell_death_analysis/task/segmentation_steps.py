
import matplotlib.image as img_



def PlotSegmentationSteps (idx,smoothed_frm, watershed_frm, threshold_frm):
    
    #save the smoothed frame image
    name_smooth ="outputs/segmentation_steps/Smoothed_frame_"+str(idx)+".png"
    img_.imsave(name_smooth,smoothed_frm)
    # save the watershed segmentation image
    name_ws ="outputs/segmentation_steps/Watershed_segmentation_"+str(idx)+".png"
    img_.imsave(name_ws,watershed_frm)
    # save the threshold-based binary segmentation
    name_thr ="outputs/segmentation_steps/threshold_based_segmentation_"+str(idx)+".png"
    img_.imsave(name_thr,threshold_frm)
    
    
   
    