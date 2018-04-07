#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 22:06:39 2018

@author: mary
"""
import numpy as np
#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

#%% EX3.1 
def loadImage(iPath):
    img = Image.open(iPath)
    a = np.array(img)
    plt.imshow(a)
    plt.show()
    return a
img = loadImage('/home/mary/Documents/coding/materials/3/slika.jpg')

def convertToGrey(img):
    iImageType = img.dtype
    greyImg = np.round(0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
    #Verify that the pixel values were correctly rounded and typecasted before saving the image
    greyImg = np.array(greyImg, dtype = iImageType)
    return greyImg
    
def saveGrey(img, iPath, iFormat):
    img.tofile(iPath, iFormat)
    
greyImg = convertToGrey(img)
saveGrey(greyImg, '/home/mary/Documents/coding/materials/3/slikaGray.jpg', 'uint8')

#%% EX3.2
def scaleImage(iImage, iSlopeA, iIntersectionB):
    iImageType = iImage.dtype
    oImage = np.round(iSlopeA * iImage + iIntersectionB)
    oImage = np.array(oImage, dtype = iImageType)
    plt.imshow(oImage)
    return oImage

a = np.ones([2,2], dtype='uint8')
scaleImage(greyImg, 0.1, 10)
scaleImage(greyImg, 254, 0) #Homework

#%%   vmesno
    
print(np.iinfo('uint8').min)
print(np.iinfo('uint8').max)
print(np.iinfo('int8').min)
print(np.iinfo('int8').max)
print(np.finfo('float').min)
print(np.finfo('float').max) 
#%% EX3.3
def windowImage(iImage, iCenter, iWidth):
    iImageType = iImage.dtype
    ls_1 = np.iinfo(iImageType).max
    f = ( ls_1 / iWidth )* (iImage - (iCenter - iWidth/2))
    n = f < np.iinfo(iImageType).min
    f[n] = np.iinfo(iImageType).min
    m = f > np.iinfo(iImageType).max
    f[m] = np.iinfo(iImageType).max
    plt.imshow(f)
    return f.astype(iImageType)

windowImage(greyImg, 40, 4)
windowImage(greyImg, 128, 204)  #Homework
#%% EX3.4
def thresholdImage(b,l):
    iImageType = b.dtype
    b = greyImg.copy()
    m = b > l
    b[m] = np.iinfo(iImageType).max
    b[np.logical_not(m)] = np.iinfo(iImageType).min
    plt.imshow(b)
    return b

thresholdImage(greyImg, 128)

#%% EX3.5
def gammaImage(iImage, iGamma):
    iImageType = iImage.dtype
    iImage=np.array(iImage,dtype='float')
    #lr_1 == ls_1 in our example as we have output the same typecast as input
    ls_1 = np.iinfo(iImageType).max
    oImage = ls_1 * ((iImage/ls_1)**iGamma)
    #plt.imshow(oImage,cmap='gray')
    return oImage.astype(iImageType)

gammaImage(greyImg, 0.5)

#%% HSV

def convertImageColorSpace(iImage, iConversionType):
    #iImage = np.array(iImage, dtype = 'float')
    if iConversionType == 'RGBtoHSV':
        iImage = iImage/255.0
        r,g,b = iImage[:,:,0], iImage[:,:,1], iImage[:,:,2]
        h = np.zeros_like(r)
        s = np.zeros_like(g)
        v = np.zeros_like(b) 
        
        Cmax = np.maximum(r, np.maximum(g,b))
        Cmin = np.minimum(r, np.minimum(g,b))
        
        delta = Cmax - Cmin + 1e-7
        h[Cmax == r] = 60.0 * ((g[Cmax == r] - b[Cmax == r]) / delta[Cmax == r] % 6.0)
        h[Cmax == g] = 60.0 * ((b[Cmax == g] - r[Cmax == g]) / delta[Cmax == g] + 2.0)
        h[Cmax == b] = 60.0 * ((r[Cmax == b] - g[Cmax == b]) / delta[Cmax == b] + 4.0)
        
        s[delta != 0.0] = delta[delta != 0.0] / (Cmax[delta != 0.0] + 1e-7)
        
        v = Cmax
        
        oImage = np.zeros_like(iImage)
        oImage[:,:,0] = h
        oImage[:,:,1] = s
        oImage[:,:,2] = v
        
        
        
    elif iConversionType == 'HSVtoRGB':
        
        h = iImage[:,:,0]; s = iImage[:,:,1]; v = iImage[:,:,2];    

        C = v * s
        X = C * (1.0 - np.abs( ( (h/60.0) % 2.0 ) - 1 ) )
        m = v - C
        
        r = np.zeros_like( h )
        g = np.zeros_like( h )
        b = np.zeros_like( h )        
        
        r[ (h>=0.0) * (h<60.0) ] = C[ (h>=0.0) * (h<60.0) ]
        g[ (h>=0.0) * (h<60.0) ] = X[ (h>=0.0) * (h<60.0) ]
        
        r[ (h>=60.0) * (h<120.0) ] = X[ (h>=60.0) * (h<120.0) ]
        g[ (h>=60.0) * (h<120.0) ] = C[ (h>=60.0) * (h<120.0) ]
        
        g[ (h>=120.0) * (h<180.0) ] = C[ (h>=120.0) * (h<180.0) ]
        b[ (h>=120.0) * (h<180.0) ] = X[ (h>=120.0) * (h<180.0) ]
        
        g[ (h>=180.0) * (h<240.0) ] = X[ (h>=180.0) * (h<240.0) ]
        b[ (h>=180.0) * (h<240.0) ] = C[ (h>=180.0) * (h<240.0) ]
        
        r[ (h>=240.0) * (h<300.0) ] = X[ (h>=240.0) * (h<300.0) ]
        b[ (h>=240.0) * (h<300.0) ] = C[ (h>=240.0) * (h<300.0) ]
        
        r[ (h>=300.0) * (h<360.0) ] = C[ (h>=300.0) * (h<360.0) ]
        b[ (h>=300.0) * (h<360.0) ] = X[ (h>=300.0) * (h<360.0) ]        
            
        r = r + m
        g = g + m
        b = b + m
        
        # ustvari izhodno sliko        
        oImage = np.zeros_like( iImage )
        oImage[:,:,0] = r; oImage[:,:,1] = g; oImage[:,:,2] = b;
        
        # zaokrozevanje vrednosti
        oImage = 255.0 * oImage
        oImage[oImage>255.0] = 255.0
        oImage[oImage<0.0] = 0.0
        
    oImage = np.array( oImage, dtype='uint8' )
        
    return oImage
        
#%%
def showImage(iImage, iTitle=''):
    plt.gca()
    plt.gcf()
    plt.imshow(iImage, cmap = cm.Greys_r)
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axes().set_aspect('equal', 'datalim') #raydalja na x osi je enaka razdalji na y osi, ne bo popačena
    plt.show() 
    
oImageHSV = convertImageColorSpace(img, 'RGBtoHSV')   
oImageRGB = convertImageColorSpace(oImageHSV, 'HSVtoRGB')      

#%%       
plt.imshow(img)
plt.imshow(oImageRGB)  

#============================================================================
showImage(oImageHSV[:,:,0] * 255 / 360.0, 'H')   
showImage(oImageHSV[:,:,1] * 255 / 255.0, 'S') 
showImage(oImageHSV[:,:,2] * 255 / 255.0, 'V') 
# H 0...360           S 0....1     V 0....1

showImage(oImageRGB[:,:,0], 'R')  
showImage(oImageRGB[:,:,1], 'G')
showImage(oImageRGB[:,:,2], 'B')    

#%%Homework
def rgbToHsvGammaToVHsvToRgb(iImage, iGamma):
    
    iImage = np.array(iImage, dtype = 'float')
    # 'RGBtoHSV':
    iImage = iImage/255.0
    r,g,b = iImage[:,:,0], iImage[:,:,1], iImage[:,:,2]
    h = np.zeros_like(r)
    s = np.zeros_like(g)
    v = np.zeros_like(b) 
    
    Cmax = np.maximum(r, np.maximum(g,b))
    Cmin = np.minimum(r, np.minimum(g,b))
    
    delta = Cmax - Cmin + 1e-7
    h[Cmax == r] = 60.0 * ((g[Cmax == r] - b[Cmax == r]) / delta[Cmax == r] % 6.0)
    h[Cmax == g] = 60.0 * ((b[Cmax == g] - r[Cmax == g]) / delta[Cmax == g] + 2.0)
    h[Cmax == b] = 60.0 * ((r[Cmax == b] - g[Cmax == b]) / delta[Cmax == b] + 4.0)
    
    s[delta != 0.0] = delta[delta != 0.0] / (Cmax[delta != 0.0] + 1e-7)
    
    v = Cmax
    v = gammaImage(v, iGamma)
    
    oImage = np.zeros_like(iImage)
    oImage[:,:,0] = h
    oImage[:,:,1] = s
    oImage[:,:,2] = v
        
    oImage = convertImageColorSpace(oImage, 'HSVtoRGB')
    return oImage

rgbToHsvGammaToVHsvToRgb(img, 0.5)