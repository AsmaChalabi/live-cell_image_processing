#!/usr/bin/python3
#-*- coding: utf-8 -*-


import numpy as np_
import pandas as pd_
import scipy.stats as stats



def ComputeHistStats(traj):
        
    traj=traj[np_.logical_not(np_.isnan(traj))]
 #       title = f"frame {frame_id}"
    
    hist,bins= np_.histogram(traj,bins=20)
    #bins= np_.delete(bins, -1)
    bins = bins[0:-1]

    #bin_center = 0.5 * (bins[1:] + bins[:-1])
    #mean= np_.sum(hist * bins) / np_.sum(hist)
    
    var= (np_.sum(hist * (bins - np_.mean(bins))**2) / np_.sum(hist) ) / 1e10
 
    hist_norm = hist / np_.sum(hist)
    
    entropy = - np_.sum(hist_norm[hist_norm>0.0] * np_.log(hist_norm[hist_norm>0.0]))
    
    
    
    return var, entropy


 
def Fit_Traj (traj, pheno): 
    
    traj = traj[np_.logical_not(np_.isnan(traj))]
    
    dist_names = ['beta',
               'expon',
               'gamma',
               'lognorm',
               'norm',
               'pearson3',
               'triang',
               'uniform',
               'weibull_min', 
               'weibull_max',
               'gausshyper']
    
    errors = []
    
    
    max_intensity = 65535
    bin_width = 512 # Usually srqt(max_inyensity)
    n_bins = int(round(max_intensity / bin_width))
    
    hist, bin_edges = np_.histogram(traj, range=(0, max_intensity),bins=n_bins)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1] - bin_edges[0]
    
    hist_norm = (max_intensity / bin_width) * hist / np_.sum(hist)
    #print("Histogram", np.min(hist_norm), np.max(hist_norm))
    #print("Integral =", (1.0 / n_bins) * np.sum(hist_norm))
    for distribution in dist_names:
       # Set up distribution and get fitted distribution parameters
       dist = getattr(stats, distribution)
       dist_parameters = dist.fit(hist_norm)
       dist_sampling = dist.pdf(bin_centers / bin_edges[-1], *dist_parameters)
       fit_error = np_.sum(np_.fabs(dist_sampling - hist_norm)) / n_bins
       errors.append(fit_error)
       
           
    results = pd_.DataFrame()
    results['Distribution'] = dist_names
    results["Fit error"] = errors
    results.sort_values(['Fit error'], inplace=True)

    return results


def Create_data_frame () : 
    
    import pandas as pd
    
    title = ['No.', 'Hue', 'Saturation', 'Value',
             'Lightness', 'AComponent', 'BComponent',
             'Blue Channel', 'Green Channel', 'Red Channel']
    
    df = pd.DataFrame(columns = title) #this create your empty data frame
    
    
    from pandas import ExcelWriter

    writer = ExcelWriter('YourCSV.xlsx')
    df.to_excel(writer, 'Sheet1',index=0)
    writer.save() #this create Excel file with your desired titles
    
    
    
    
    
    
    
    
    