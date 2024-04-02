# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 13:48:33 2021

@author: lzmfor
"""
import cv2
import numpy as np
import time
import pywt # Wavelet 小波

'''
# Function Depth-of-field extension | 景深拓展
# Abbreviations
- Image     --> img
- Extend    --> Exd
- Wavelet   --> Wt
- Gradient  --> Grad
- Variance  --> Vari
- Average   --> Avg
'''

# 基于小波分解 ##################################################################
#       1. 先求每个截面的局部梯度方差                                             
#       2. 再构建融合的小波高低频系数  

if 1 == 1:      # 仅仅是为了维持缩进   
    timeStartExdWt = time.time()
    
################################# 预定义变量 ####################################
    
    # 局部区块大小 ( 是否需要做一个划分判据 ex. max( 16, itemPixelNumX/? ))
    # 局部区块 X 向个数 | 局部区块 Y 向个数
    itemExdDviSize = 15
    itemExdDviNumX , itemExdDviNumY = int( itemPixelNumX / 2 / itemExdDviSize ), int( itemPixelNumY / 2 / itemExdDviSize ) 
  
    # 保存 每个截面的小波系数
    imgExdWtSet = []
    
    # 保存 每个截面的 高频 | 低频 局部梯度方差
    itemExdGradLocalVariHighSet, itemExdGradLocalVariLowSet = [], []
    
    # 保存 Z 位置
    itemInfoZ = np.ones( (itemExdDviNumX ,itemExdDviNumY) , dtype = np.uint )

######################### 计算各个截面的局部梯度方差  ##############################
    
    for z in range( 0,int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):  
# 单层小波分解
        # (cA, (cH, cV, cD)) : tuple
        # 低频分量 LL，水平高频 HL、垂直高频 LH、对角线高频 HH
        # Approximation, horizontal detail, vertical detail, diagonal detail coefficients
        
        imgExdWtZ = pywt.dwt2( imgReconSet[z], 'haar')
        imgExdWtSet.append( imgExdWtZ )
        
        # Option 1
#        itemExdWtcA = imgExdWtZ[0] 
#        itemExdWtcH = imgExdWtZ[1][0] 
#        itemExdWtcV = imgExdWtZ[1][1] 
#        itemExdWtcD = imgExdWtZ[1][2] 
        
        # Option 2 转换类型 (复数部分会被去掉)
        itemExdWtcA = np.float64( imgExdWtZ[0] )
        itemExdWtcH = np.float64( imgExdWtZ[1][0] )
        itemExdWtcV = np.float64( imgExdWtZ[1][1] )
        itemExdWtcD = np.float64( imgExdWtZ[1][2] )

        # Option 3 取绝对值
#        itemExdWtcA = np.abs( imgExdWtZ[0] )
#        itemExdWtcH = np.abs( imgExdWtZ[1][0] )
#        itemExdWtcV = np.abs( imgExdWtZ[1][1] )
#        itemExdWtcD = np.abs( imgExdWtZ[1][2] )

#        itemExdWtcA = cv2.GaussianBlur( itemExdWtcA, (5,5), 1, 0)
#        itemExdWtcH = cv2.GaussianBlur( itemExdWtcH, (5,5), 0, 0)
#        itemExdWtcV = cv2.GaussianBlur( itemExdWtcV, (5,5), 0, 0)
#        itemExdWtcD = cv2.GaussianBlur( itemExdWtcD, (5,5), 1, 0)
    
# 方差计算
        ## 梯度分量 
        # 梯度 1 高频 x | 梯度 2 高频 y | 梯度 3 低频 x | 梯度 4 低频 y            
        
        # Option 1 cv2.Sobel 
#        imgExdGradHighX = cv2.Sobel( itemExdWtcH, cv2.CV_64F, 1, 0, ksize=3 ) +\
#                                    cv2.Sobel( itemExdWtcD, cv2.CV_64F, 1, 0, ksize=3 )        
#        imgExdGradHighY = cv2.Sobel( itemExdWtcV, cv2.CV_64F, 0, 1, ksize=3 ) +\
#                                    cv2.Sobel( itemExdWtcD, cv2.CV_64F, 0, 1, ksize=3 )        
#        imgExdGradLowX = cv2.Sobel( itemExdWtcA, cv2.CV_64F, 1, 0, ksize=3 )       
#        imgExdGradLowY = cv2.Sobel( itemExdWtcA, cv2.CV_64F, 0, 1, ksize=3 )

        # Option 2 cv2.filter2D (Sobel 算子)
        # borderType=cv2.BORDER_REFLECT | cv2.BORDER_REFLECT_101 | cv2.BORDER_REPLICATE 
        # (cba|abcdefgh|hgf) (dcb|abcdefgh|gfe) (aaa|abcdefgh|hhh)
        itemWtSx, itemWtSy = np.array( [[-1,0,1],[-2,0,2],[-1,0,1]] ), np.array( [[-1,-2,-1],[0,0,0],[1,2,1]] ) 
        # itemWtSx, itemWtSy = np.array( [[-3,0,3],[-10,0,10],[-3,0,3]] ), np.array( [[-3,-10,-3],[0,0,0],[3,10,3]] ) 
              
        imgExdGradHighX = cv2.filter2D( itemExdWtcH, -1, itemWtSx, borderType=cv2.BORDER_REPLICATE ) +\
                            cv2.filter2D( itemExdWtcD, -1, itemWtSx, borderType=cv2.BORDER_REPLICATE )
        imgExdGradHighY = cv2.filter2D( itemExdWtcV, -1, itemWtSy, borderType=cv2.BORDER_REPLICATE ) +\
                            cv2.filter2D( itemExdWtcD, -1, itemWtSy, borderType=cv2.BORDER_REPLICATE )
        imgExdGradLowX = cv2.filter2D( itemExdWtcA, -1, itemWtSx, borderType=cv2.BORDER_REPLICATE )
        imgExdGradLowY = cv2.filter2D( itemExdWtcA, -1, itemWtSy, borderType=cv2.BORDER_REPLICATE )      
        
        ## 系数图梯度
        # 高频系数图梯度 High Coefficient | 低频系数图梯度 Low Coefficient
        # 归一化 (optional)        
        imgExdGradHigh = np.sqrt( imgExdGradHighX**2 + imgExdGradHighY**2 )
        imgExdGradLow = np.sqrt( imgExdGradLowX**2 + imgExdGradLowY**2 )        
        # imgOutput1 = np.zeros( imgExdGradHighX.shape )      
        # imgExdGradHigh = cv2.normalize( imgExdGradHigh, imgOutput1, 0, 255, cv2.NORM_MINMAX )                   
        # imgOutput2 = np.zeros( imgExdGradLowX.shape )
        # imgExdGradLow = cv2.normalize( imgExdGradLow , imgOutput2, 0, 255, cv2.NORM_MINMAX ) 
        
        ## 局部平均梯度
        # 高频系数 局部区域平均梯度 High LocalAvg | 低频系数 局部区域平均梯度 Low LocalAvg 
        imgExdGradLocalAvgHighZ = np.ones( (itemExdDviNumX ,itemExdDviNumY) , dtype = np.float )
        imgExdGradLocalAvgLowZ = np.ones( (itemExdDviNumX ,itemExdDviNumY) , dtype = np.float )
        for i in range( 0, itemExdDviNumX ):
            for j in range( 0, itemExdDviNumY ):
                imgExdGradLocalAvgHighZ[i,j] = np.mean( imgExdGradHigh[ i*itemExdDviSize:(i+1)*itemExdDviSize ,\
                                                                         j*itemExdDviSize:(j+1)*itemExdDviSize ] )                
                imgExdGradLocalAvgLowZ[i,j] = np.mean( imgExdGradLow[ i*itemExdDviSize:(i+1)*itemExdDviSize ,\
                                                                         j*itemExdDviSize:(j+1)*itemExdDviSize ] )                                    
        ## 局部梯度方差
        # 高频系数图 局部梯度方差 矩阵 High Variance | 低频系数图 局部梯度方差 矩阵 Low Variance
        # 得在循环内定义，不然xxxxSet中，均为最后一个进入的xxxxZ                  
        itemExdGradLocalVariHighZ = np.ones( (itemExdDviNumX ,itemExdDviNumY) , dtype = np.float )
        itemExdGradLocalVariLowZ = np.ones( (itemExdDviNumX ,itemExdDviNumY) , dtype = np.float )
        for i in range( 0, itemExdDviNumX ):
            for j in range( 0, itemExdDviNumY ):
                imgExdGradLocal = imgExdGradHigh[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize]
                imgExdGradLocal = np.square( imgExdGradLocal - imgExdGradLocalAvgHighZ[i,j] )
                itemExdGradLocalVariHighZ[i,j] = sum( sum( imgExdGradLocal ))
                del imgExdGradLocal
                imgExdGradLocal = imgExdGradLow[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize]
                imgExdGradLocal = np.square( imgExdGradLocal - imgExdGradLocalAvgLowZ[i,j] )
                itemExdGradLocalVariLowZ[i,j] = sum( sum( imgExdGradLocal ))
                del imgExdGradLocal       
        itemExdGradLocalVariHighSet.append( itemExdGradLocalVariHighZ )
        itemExdGradLocalVariLowSet.append( itemExdGradLocalVariLowZ )
    
        del imgExdGradLocalAvgHighZ
        del imgExdGradLocalAvgLowZ        
        del itemExdGradLocalVariHighZ
        del itemExdGradLocalVariLowZ 
        del itemExdWtcA
        del itemExdWtcH
        del itemExdWtcV
        del itemExdWtcD        
        del imgExdGradHighX
        del imgExdGradHighY
        del imgExdGradLowX
        del imgExdGradLowY
        del imgExdGradHigh
        del imgExdGradLow
                
#################################### 深度索引 ###################################           
    # itemHighMax = np.zeros( (1,int((itemEndZ-itemBeginZ)/itemIntervalZ+1)) )
    # itemLowMax = np.zeros( (1,int((itemEndZ-itemBeginZ)/itemIntervalZ+1)) )
    
    # 初始化 合成高低频系数 
    imgExdWtSyntheticcA = imgExdWtSet[0][0]
    imgExdWtSyntheticcH = imgExdWtSet[0][1][0]
    imgExdWtSyntheticcV = imgExdWtSet[0][1][1]
    imgExdWtSyntheticcD = imgExdWtSet[0][1][2]  
    
    for i in range(0,itemExdDviNumX):           
        for j in range(0,itemExdDviNumY):
            itemSquareVariMax = np.zeros( (1,int((itemEndZ-itemBeginZ)/itemIntervalZ+1)) )
            itemHighMax = np.zeros( (1,int((itemEndZ-itemBeginZ)/itemIntervalZ+1)) )
            itemLowMax = np.zeros( (1,int((itemEndZ-itemBeginZ)/itemIntervalZ+1)) )            
            for z in range( 0,int((itemEndZ-itemBeginZ)/itemIntervalZ+1) ):
                # Option 1 取用高频最大、低频中最大的 Z                           
                # itemHighMax[0,z] = itemExdGradLocalVariHighSet[z][i][j]
                # itemLowMax[0,z] = itemExdGradLocalVariLowSet[z][i][j]
                # itemInfoZ[i][j] = int( max( itemHighMax.argmax() , itemLowMax.argmax() ) ) 
  
                # Option 2 取用高频、低频平方和 最大的 Z                           
#                 itemSquareVariMax[0,z] = itemExdGradLocalVariHighSet[z][i][j]**2 + itemExdGradLocalVariLowSet[z][i][j]**2
#                 itemInfoZ[i][j] = int( itemSquareVariMax.argmax() )
                
                # Option 3 取用高频、低频同时 最大的 Z
                itemHighMax[0,z] = itemExdGradLocalVariHighSet[z][i][j]
                itemLowMax[0,z] = itemExdGradLocalVariLowSet[z][i][j]                
                if itemHighMax.argmax() == itemLowMax.argmax():                    
                    itemInfoZ[i][j] = int( itemHighMax.argmax() )
                else:
                    itemInfoZ[i][j] = 0
                                    
            # cA 
            imgExdWtSyntheticcA[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ] = \
                            imgExdWtSet[ itemInfoZ[i][j] ][0][ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ]                
            # cH
            imgExdWtSyntheticcH[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ] = \
                            imgExdWtSet[ itemInfoZ[i][j] ][1][0][ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ]
            # cV
            imgExdWtSyntheticcV[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ] = \
                            imgExdWtSet[ itemInfoZ[i][j] ][1][1][ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ] 
            # cD    
            imgExdWtSyntheticcD[ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ] = \
                            imgExdWtSet[ itemInfoZ[i][j] ][1][2][ i*itemExdDviSize:(i+1)*itemExdDviSize, j*itemExdDviSize:(j+1)*itemExdDviSize ]     
            del itemSquareVariMax
            del itemHighMax
            del itemLowMax

############################ 小波反变换-->景深拓展图 ##############################
    imgExdWtSynthetic = ( imgExdWtSyntheticcA, ( imgExdWtSyntheticcH, imgExdWtSyntheticcV, imgExdWtSyntheticcD ) )
    imgExd = pywt.idwt2( imgExdWtSynthetic , 'haar' )
    
    timeEndExdWt = time.time()
    print( 'Extend total time cost', timeEndExdWt - timeStartExdWt )

# 区块大小的确定
#timeStartExtend = time.time()
#
#aTestNumA = 380
#aTestNumB = 400
#aTestCount = 1
#
#while aTestCount <= min( aTestNumA , aTestNumB ) :
#    if aTestNumA % aTestCount == 0 and aTestNumB % aTestCount ==0 :
#        print(aTestCount)    
#    aTestCount = aTestCount + 1
#   
#
#timeEndExtend = time.time()
#print('totally cost',timeEndExtend-timeStartExtend)



















































