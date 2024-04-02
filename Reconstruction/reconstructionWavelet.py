 # -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 08:56:29 2021

@author: lzmfor
"""
import cv2
import numpy as np
import time
import math

from math import sin
from math import sqrt
from math import log
from math import pi
from math import atan
from math import exp

'''
# Function Reconstruction | 重建
# Abbreviations
- Image             --> img
- Reconstruction    --> Recon
- Wavelet           --> Wt
'''

if 1 == 1:      # 仅仅是为了维持缩进
    
    timeStartReconTotal = time.time()
# 基本参数
    # 波长
    # 像素尺寸
    # 重建起始，重建末尾，重建间隔
    itemWaveLength = 532*1e-9       
    itemPixelSize = 1.72*1e-6             # 1.72*1e-6
    itemBeginZ, itemEndZ, itemIntervalZ= 0.0065, 0.0075, 0.0001
    print( 'Total',int((itemEndZ-itemBeginZ)/itemIntervalZ+1) )
# 读入图片        
    # imgInput = np.ones( (itemPixelNumX,itemPixelNumY) , dtype = np.float )
    imgInput = cv2.imread('Test_holo_300x300.jpg',0)
    itemPixelNumX = np.shape(imgInput)[0]
    itemPixelNumY = np.shape(imgInput)[1] 
    
# 原始图像频谱
    imgInputSpectrum = np.fft.fft2(imgInput)
    # imgInputSpectrum = np.fft.fftshift(imgInputSpectrum)

# x2+y2
    imgSquareOfxy = np.ones( (itemPixelNumX ,itemPixelNumY) , dtype = np.float )
    itemCenterX = ( itemPixelNumX + 1 ) / 2 - 1
    itemCenterY = ( itemPixelNumY + 1 ) / 2 - 1
        
    for i in range(0,itemPixelNumX):          
        for j in range(0,itemPixelNumY):                    
            imgSquareOfxy[i,j] = itemPixelSize**2 * ( (i - itemCenterX)**2 + (j - itemCenterY)**2 )
     
    timeStartReconWt = time.time()     
# 小波函数
    # 小波函数的合集 | 频谱的合集
    # 单个截面
    imgReconWtSet = []
    imgReconWtSetSpectrum = []
    # imgReconWtZ = np.ones( (itemPixelNumX ,itemPixelNumY) , dtype = np.float ) 放进循环
    
    # 参数 Epsilon
    itemReconWtEpsilon = 0.01   #log(100)
    itemReconWtEpsilonT = 1 / itemReconWtEpsilon
    for z in range( int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):
        # 重建位置
        itemReconZ = itemBeginZ + z*itemIntervalZ
        ''' chengxiaofeng: Holo
#       alpha = lamda * z / M_PI;//其实是alpha的平方，不用开根
#		sigma1 = imgScale * imgScale * pixelSize * pixelSize / (4 * alpha * log(100));
#		sigma2 = M_PI * M_PI * alpha / (4 * pixelSize * pixelSize * log(100));
#		sigma = sigma1 < sigma2 ? sigma1 : sigma2;
#		M = sin(atan(sigma)) / sqrt(1 + sigma * sigma);//和论文中的公式不一样
        '''
        # 小波 Alpha 的平方 (小波尺寸参数)  # itemReconWtA = sqrt( itemWaveLength*itemReconZ / pi )
        # 小波 Sigma 的平方 (带宽因子 )
        # 小波 K              
        itemReconWtSquareOfAlpha = itemWaveLength * itemReconZ / pi
        itemReconWtSquareOfSigma1 = itemPixelNumX**2 * itemPixelSize**2 / ( 4 * itemReconWtSquareOfAlpha * log(1/itemReconWtEpsilon) )
        itemReconWtSquareOfSigma2 = pi**2 * itemReconWtSquareOfAlpha / (4 * itemPixelSize**2 * log(1/itemReconWtEpsilon) )
        itemReconWtSquareOfSigma = min(itemReconWtSquareOfSigma1,itemReconWtSquareOfSigma2)    
        itemReconWtK = sin(atan(itemReconWtSquareOfSigma)) / sqrt( 1 + itemReconWtSquareOfSigma**2 )                 
        # 小波函数
        #此处每次循环都创建 imgReconWtZ，不然imgReconWtSet中，均为最后一个进入的imgReconWtZ
        imgReconWtZ = np.ones( (itemPixelNumX ,itemPixelNumY) , dtype = np.float )
        for i in range(0,itemPixelNumX):           
            for j in range(0,itemPixelNumY):
                imgReconWtZ[i,j] = 1 / itemReconWtSquareOfAlpha * \
                                ( sin( imgSquareOfxy[i,j] / itemReconWtSquareOfAlpha ) - itemReconWtK ) * \
                                 exp( -imgSquareOfxy[i,j] / itemReconWtSquareOfAlpha / itemReconWtSquareOfSigma**2  )
        # 小波函数频谱集          
        imgReconWtZSpectrum = np.fft.fft2(imgReconWtZ)
        # imgReconWtZSpectrum = np.fft.fftshift(imgReconWtZSpectrum)
        imgReconWtSet.append(imgReconWtZ)
        imgReconWtSetSpectrum.append(imgReconWtZSpectrum) 
        
        #此处删除imgReconWtZ，在新的循环中创建 imgReconWtZ，不然imgReconWtSet中，均为最后一个进入的imgReconWtZ
        del imgReconWtZ    
        print('No.',z+1,'Done')  
    timeEndReconWt = time.time()    
    print('Reconstruction Wavelet time cost',timeEndReconWt - timeStartReconWt)  
                                 
    timeStartReconIFFT = time.time()
# 重建
    # 重建截面合集
    # 单个截面
    imgReconSet = []
    imgReconZ = np.ones( (itemPixelNumX ,itemPixelNumY) , dtype = np.float )
   
    for z in range( int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):    
       imgReconZ = imgInputSpectrum * imgReconWtSetSpectrum[z]        
       imgReconZ = np.fft.ifft2(imgReconZ)
       imgReconZ = 1 - imgReconZ
       
       # Shift 对角交换，左上区块-右下区块 | 右上区块-左下区块
       # 这里拼接后，取中间矩阵，完成对角交换
       imgReconZAdjust = np.c_[imgReconZ,imgReconZ]
       imgReconZAdjust = np.r_[imgReconZAdjust,imgReconZAdjust]     
       imgReconZ = imgReconZAdjust[ int(itemPixelNumX/2) : 2*itemPixelNumX-int(itemPixelNumX/2) , \
                                        int(itemPixelNumY/2) : 2*itemPixelNumY-int(itemPixelNumY/2) ]
       
       # 其他 交换方式
       # imgReconQuarterRegion = np.ones( ( int(itemPixelNumX/2) ,int(itemPixelNumY/2) ) , dtype = np.float )
       # 左上区块
       # imgReconQuarterRegion = imgReconZ[0:int(itemPixelNumX/2),0:int(itemPixelNumY/2)]
   
       imgReconSet.append( imgReconZ )
       
    # 归一化
    # 这里归一化，考虑所有重建图的灰度值，取出其中的 max 和 min
    imgReconNorSet = []
    
    # Option 1 利用绝对值进行归一化 (则忽略了正负)
#    itemMaxInRecon ,itemMinInRecon = np.amax( np.abs( imgReconSet[0] ) ), np.amin( np.abs( imgReconSet[0] ) )
#    for z in range( int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):       
#        imgReconNorSet.append( np.abs( imgReconSet[z] ) )
#        if np.amax( imgReconNorSet[z] ) > itemMaxInRecon:
#            itemMaxInRecon = np.amax( imgReconNorSet[z] )
#        if np.amin( imgReconNorSet[z] ) < itemMinInRecon:
#            itemMinInRecon = np.amin( imgReconNorSet[z] )
#    imgReconNorSet = 255 * ( imgReconNorSet - itemMinInRecon ) / ( itemMaxInRecon - itemMinInRecon )    
       
    # Option 2 利用实数部分进行归一化 (考虑正负值)
    itemMaxInRecon ,itemMinInRecon = np.amax( np.float64( imgReconSet[0] ) ), np.amin( np.float64( imgReconSet[0] ) )
    for z in range( int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):       
        imgReconNorSet.append( np.float64( imgReconSet[z] ) )
        if np.amax( imgReconNorSet[z] ) > itemMaxInRecon:
            itemMaxInRecon = np.amax( imgReconNorSet[z] )
        if np.amin( imgReconNorSet[z] ) < itemMinInRecon:
            itemMinInRecon = np.amin( imgReconNorSet[z] )
    imgReconNorSet = 255 * ( imgReconNorSet - itemMinInRecon ) / ( itemMaxInRecon - itemMinInRecon )     
    
    del imgReconSet
    imgReconSet = imgReconNorSet  
    del imgReconNorSet 
    
    timeEndReconIFFT = time.time()
    print('Reconstruction iFFT time cost',timeEndReconIFFT - timeStartReconIFFT) 
    
    timeEndReconTotal = time.time()
    print('Reconstruction total time cost',timeEndReconTotal - timeStartReconTotal)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    