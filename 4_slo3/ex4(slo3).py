import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # kjižnica barvnih lestvic
import PIL.Image as im

#%% funkcije


def showImage(iImage,iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap = cm.Greys_r) # brez cmap vrne v osnovi sivinsko sliko s RGB barvami
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axes().set_aspect('equal','datalim') # x in y sta enakovredni, x in y imata isto merilo
    plt.show()


def loadImage( iPath ): 
    iImage = im.open(iPath)
    oImage=np.array(iImage,dtype='uint8') 
    return oImage


    # shranjujemo različne formate
def saveImage( iPath, iImage, iFormat ):
    xImage = im.fromarray(iImage)
    xImage.save(iPath+'.'+iFormat)
    
def colorToGrey(iImage):
    iImageType = iImage.dtype
    rgb = np.array(iImage,dtype= 'float')
    return (rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114).astype(iImageType)

#%%Naliga1
def discreteConvolution2D( iImage, iKernel ): # za sivinsko sliko
    iImage = np.asarray(iImage) # ce je vhodni parameter np.array ne naredi nič, drugače pa naredi np.array
    iKernel = np.asarray(iKernel)
    dy,dx = iImage.shape   # y vrstice, x stolpci.
    dv,du = iKernel.shape
    oImage = np.zeros((dy-dv+1, dx-du+1))
    
    for y in range(dy-dv+1):
        for x in range(dx-du+1):
            oImage[y,x] = np.sum(iKernel * iImage[y:y+dv, x:x+du])
    return np.array(oImage)
    
Image = loadImage('/home/mary/Documents/Luka/RV_3_OsnovnaObdelavaSlik/slika.JPG')
ImageG = colorToGrey(Image)
showImage(ImageG,'sivinska')
Kernel = np.ones([3,3])*0.1111111111111111
#Kernel[1,1] = 1

oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'konvolucija')

#saveImage('kuk1', oImageG, 'png')
#%%
Kernel = np.zeros([15,15])
Kernel[-1,-1] = 1

oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'konvolucija')
#%%Naloga2   box filter

Kernel = np.ones([7,7])
Kernel = Kernel/np.sum(Kernel)
oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'box')

#%% Naloga1.2  za robove  sobelov filter 
Kernel = np.array([[-1,-2,-1],[0, 0, 0], [1,2,1]])
Kernel = Kernel/np.sum(np.abs(Kernel))
oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'sobel')
#%% Naloga1.3   za robove
Kernel = np.full([3,3],-1)
Kernel[1,1] = 8
Kernel = Kernel/np.sum(Kernel)
oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'konvolucija3')

#%% Naloga1.4     Tudi za robove, ostrenje
Kernel = np.full([3,3],-1)
Kernel[1,1] = 17
Kernel = Kernel/np.sum(Kernel)
oImageG = discreteConvolution2D(ImageG, Kernel)
showImage(oImageG,'konvolucija4')
showImage(ImageG,'Original')

#%% Naloga2

def interpolate0Image2D( iImage, iCoorX, iCoorY ):
    iImage = np.asarray(iImage) # ce je vhodni parameter np.array ne naredi nič, drugače pa naredi np.array
    iCoorX = np.asarray(iCoorX, dtype = 'float')
    iCoorY = np.asarray(iCoorY, dtype = 'float')
    dy,dx = iImage.shape # najprej vrstice in potem sstolpci
    oShape = iCoorX.shape
    iCoorX = np.floor(iCoorX.flatten()).astype('int')  # zaokrozimo, interpolacija 0. reda
    iCoorY = np.floor(iCoorY.flatten()).astype('int')
    oImage = np.zeros(iCoorX.shape,dtype = iImage.dtype)
    for idx in range(oImage.size):
        tx = iCoorX[idx]
        ty = iCoorY[idx]
        if ty >= 0 and tx >= 0 and tx < dx and ty < dy:
            oImage[idx]= iImage[ty,tx]
    
    return np.reshape(oImage,oShape)

 
ImageGs = ImageG[210:280,260:360]
dy,dx = ImageGs.shape
CoorX, CoorY = np.meshgrid(np.arange(0,dx,1/3.0),np.arange(0,dy,1/3.0))
oImageGs = interpolate0Image2D(ImageGs,CoorX, CoorY)
showImage(ImageGs,'original povecan')
showImage(oImageGs, 'interpoliran original povecan')
print(oImageG.shape)  
#%% Naloga3
def interpolate1Image2D( iImage, iCoorX, iCoorY ):
    
    iImage = np.asarray(iImage) # ce je vhodni parameter np.array ne naredi nič, drugače pa naredi np.array
    iCoorX = np.asarray(iCoorX, dtype = 'float')
    iCoorY = np.asarray(iCoorY, dtype = 'float')
    dy,dx = iImage.shape # najprej vrstice in potem sstolpci
    oShape = iCoorX.shape
    iCoorX = iCoorX.flatten()
    iCoorY = iCoorY.flatten()
    
    #iCoorX = np.floor(iCoorX.flatten()).astype('int')
    #iCoorY = np.floor(iCoorY.flatten()).astype('int')
    oImage = np.zeros(iCoorX.shape,dtype = 'float')
    for idx in range(oImage.size):
        ix = int(np.floor(iCoorX[idx]))        
        iy = int(np.floor(iCoorY[idx]))
        sx = iCoorX[idx] - ix
        sy = iCoorY[idx] - iy
      
        if iy >= 0 and ix >= 0 and ix < (dx-1) and iy < (dy-1):
            a = (1-sx) * (1-sy)
            b = sx * (1-sy)
            c = sy * (1-sx)
            d = sx * sy
            
            oImage[idx]= a * iImage[iy,ix] + b * iImage[iy,ix +1] + c * iImage[iy+1,ix] + d * iImage[iy+1,ix+1]
           
    if iImage.dtype.kind in ('u','i'):
        oImage = np.clip(oImage, np.iinfo(iImage.dtype).min, np.iinfo(iImage.dtype).max)
    
    return np.array(np.reshape(oImage, oShape),dtype= iImage.dtype)

ImageGs = ImageG[210:280,260:360]
showImage(ImageGs,'original povecan')
dy,dx = ImageGs.shape
CoorX, CoorY = np.meshgrid(np.arange(0,dx,1/3.0),np.arange(0,dy,1/3.0))
oImageGs = interpolate1Image2D(ImageGs,CoorX, CoorY)

showImage(oImageGs, 'interpoliran original povecan')

oImageGr2 = ImageG[::2,::2]
oImageGr8 = ImageG[::8,::8]
showImage(ImageG,'original')
showImage(oImageGr2,'decimirana 2')
showImage(oImageGr8,'decimirana 8')

#%% Naloga 4

def decimateImage2D(iImage,ilevel): #2ˇiLevel
    iImage = np.asarray(iImage)
    iImageType = iImage.dtype
    iKernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]]) #zs vsak nivo sliko gladimo in zdecimiramo za 2.
    iImage = discreteConvolution2D(iImage,iKernel)
    iImage = iImage[::2,::2]
    
    
    if ilevel <= 1:
        return np.array(iImage,dtype = iImageType)
    else:
        return decimateImage2D(iImage, ilevel - 1)
        
oImageGrL8 = decimateImage2D(ImageG,1)
showImage(oImageGrL8,'decimirana')
#%%
np.fft.fft2(Image).shape
