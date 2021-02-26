import numpy as np_
import scipy.interpolate as in_
import skimage as im_




def ContrastNormalized(image, block_shape: tuple) -> np_.ndarray:
    """
     ContrastNormalized function
     Call all the contrast normalization functions
     Return the contrast normalized image
    """
    assert (block_shape[0] % 2 == 1) and (block_shape[1] % 2 == 1) # the remainder after devision by 2 must be =1

    # call the contrast normalization functions 
    res_img = ImageCroppedToEntireBlocks(image, block_shape)
    lmp_img = LocalMostPresent(res_img, block_shape)
    rescaled = RescaledImage(lmp_img, block_shape, res_img.shape)


    return res_img - rescaled  #background subtructed image


def ImageCroppedToEntireBlocks(img: np_.ndarray, block_shape) -> np_.ndarray:
    """
     ImageCroppedToEntireBlocks (image, block_,shape)
     Rescale the input image if the remainder after cropping the image
     by the block shape is not zero, to get entire blocks.
     if not, the original image is returned.
    """
    row_margin = img.shape[0] % block_shape[0] # remainder after devision of rows by the block row size
    col_margin = img.shape[1] % block_shape[1] # remainder after devision of columns by the block column size
 
    row_half_margin = row_margin // 2
    col_half_margin = col_margin // 2

    if (row_margin > 0) and (col_margin > 0): # if the remainder after division > 0
        # return a cropped image (row,col)
        return np_.array(
            img[
                row_half_margin : (row_half_margin - row_margin),
                col_half_margin : (col_half_margin - col_margin),
            ]
        )
    elif row_margin > 0:
        return np_.array(img[row_half_margin : (row_half_margin - row_margin), :]) # rescale the image rows
    elif col_margin > 0:
        return np_.array(img[:, col_half_margin : (col_half_margin - col_margin)]) # rescale the image columns

    return np_.array(img) # return the cropped image


def LocalMostPresent(img: np_.ndarray, block_shape) -> np_.ndarray:
    """
     LocalMostPresent(img_, block_shape_) 
     Calculate the local background ( the background of each block )
     using a gaussian filter
    """
    
    view = im_.util.view_as_blocks(img, block_shape) # block view of the input image
    local_most = np_.empty(view.shape[:2], dtype=np_.float64) # create an empty array 

    for row in range(view.shape[0]): # for each row
        for col in range(view.shape[1]): # for each column
            block = view[row, col, :, :] # create the block
            # calculate the histogram
            hist, bin_edges = np_.histogram(
                np_.log(block + 1.0), bins=max(block.size // 100, 10), density=False
            )
            # apply a gaussian filter to the histogram
            hist = im_.filters.gaussian(hist.astype(np_.float64), sigma=3)#sig=3
            # search for the index of the max hist value
            most_present = np_.argmax(hist)
            # save the most present block
            local_most[row, col] = (
                np_.mean(np_.exp(bin_edges[most_present : (most_present + 2)])) - 1.0 # take the exp to cancel the log, 
                                                                                      # -1 to remove the +1 added to the hist computation
            )

    return local_most


def RescaledImage(img: np_.ndarray, block_shape, full_size) -> np_.ndarray:
    """
    RescaledImage(lmp_img, block_shape_, img_.shape)
     Reconstruction of the full background resolution using a pchip interpolation
     "pchip interpolation" is used to avoid negative interpolation values. 
    """

    block_half_shape = (block_shape[0] // 2, block_shape[1] // 2)
    new_size = (full_size[0] - block_shape[0] + 1, full_size[1] - block_shape[1] + 1)

    rescaled = np_.zeros((full_size[0], img.shape[1]), dtype=np_.float64)# empty vector (containing 0), x= full size image rows,
                                                                         # y= cropped image column
#===== rows
    old_rows = range(img.shape[0]) # cropped image rows
    flt_rows = np_.linspace(0, old_rows[-1], new_size[0])# array ( start=0, stop=cropped image rows-1, 
                                                         #         number of samples to generate: new size rows)
                                                         
    new_rows = slice(block_half_shape[0], rescaled.shape[0] - block_half_shape[0]) # object slice  (start:block half shape,
                                                                                   #                stop: rescaled rows-block half shape )
                                                                                   # rescale rows by block half shape rows
    
    for col in range(img.shape[1]):# for each column of the cropped image
        # full rows reconsruction with pchip interpolation
        rescaled[new_rows, col] = in_.pchip_interpolate(old_rows, img[:, col], flt_rows)

# ===== columns
    img = rescaled
    rescaled = np_.zeros(full_size, dtype=np_.float64) # same full size image shape 

    old_cols = range(img.shape[1]) # old column number
    flt_cols = np_.linspace(0, old_cols[-1], new_size[1]) # array ( start=0, stop=cropped image columns-1, 
                                                         #         number of samples to generate: new size columns)
    new_cols = slice(block_half_shape[1], rescaled.shape[1] - block_half_shape[1]) # object slice  (start:block half shape,
                                                                                   #  stop: rescaled columns-block half shape )
        
                                                                                   # rescale rows by block half shape columns
    for row in range(img.shape[0]): # for each row of the cropped image
        # full columns reconsruction with pchip interpolation
        rescaled[row, new_cols] = in_.pchip_interpolate(old_cols, img[row, :], flt_cols)
        
    return im_.filters.gaussian(rescaled, sigma=9) # return a full background recontructed image, 
                                                    # filtred with a gaussian filter to remove noise


def BlockBasedCroppedImage(img: np_.ndarray, block_shape) -> np_.ndarray:
    """
    BlockBasedCroppedImage(image,block_shape)
    Rescale the input image by a half block shape
    """
    
    block_half_shape = (block_shape[0] // 2, block_shape[1] // 2)
    # rescale the image by a half block shape 
    return np_.array(
        img[
            block_half_shape[0] : (img.shape[0] - block_half_shape[0]),
            block_half_shape[1] : (img.shape[1] - block_half_shape[1]),
        ]
    )
