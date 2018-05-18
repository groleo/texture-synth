#!/usr/bin/python

# https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf

import cv2
import sys
import numpy as np
from random import randint
#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def OverlapErrorVertical( outPx, inpPx ):
    iLeft,jLeft = outPx
    iRight,jRight = inpPx
    OverlapErr = 0
    diff = np.zeros((3))
    for i in range( g_PatchSize ):
        for j in range( g_OverlapWidth ):
            diff[0] =  int(g_img_out[i + iLeft, j+ jLeft][0]) - int(g_img_inp[i + iRight, j + jRight][0])
            diff[1] =  int(g_img_out[i + iLeft, j+ jLeft][1]) - int(g_img_inp[i + iRight, j + jRight][1])
            diff[2] =  int(g_img_out[i + iLeft, j+ jLeft][2]) - int(g_img_inp[i + iRight, j + jRight][2])
            OverlapErr += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
    return OverlapErr

def OverlapErrorHorizntl( leftPx, rightPx ):
    iLeft,jLeft = leftPx
    iRight,jRight = rightPx
    OverlapErr = 0
    diff = np.zeros((3))
    for i in range( g_OverlapWidth ):
        for j in range( g_PatchSize ):
            diff[0] =  int(g_img_out[i + iLeft, j+ jLeft][0]) - int(g_img_inp[i + iRight, j + jRight][0])
            diff[1] =  int(g_img_out[i + iLeft, j+ jLeft][1]) - int(g_img_inp[i + iRight, j + jRight][1])
            diff[2] =  int(g_img_out[i + iLeft, j+ jLeft][2]) - int(g_img_inp[i + iRight, j + jRight][2])
            OverlapErr += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
    return OverlapErr

def GetBestPatches(px):
    PixelList = []
    #check for top layer
    if px[0] == 0:
        for i in range(g_sample_height - g_PatchSize):
            for j in range(g_OverlapWidth, g_sample_width - g_PatchSize ):
                error = OverlapErrorVertical( (px[0], px[1] - g_OverlapWidth), (i, j - g_OverlapWidth)  )
                if error  < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error < g_ErrThreshold/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px[1] == 0:
        for i in range(g_OverlapWidth, g_sample_height - g_PatchSize ):
            for j in range(g_sample_width - g_PatchSize):
                error = OverlapErrorHorizntl( (px[0] - g_OverlapWidth, px[1]), (i - g_OverlapWidth, j)  )
                if error  < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error < g_ErrThreshold/2:
                    return [(i,j)]
    #for pixel placed inside
    else:
        for i in range(g_OverlapWidth, g_sample_height - g_PatchSize):
            for j in range(g_OverlapWidth, g_sample_width - g_PatchSize):
                error_Vertical   = OverlapErrorVertical( (px[0], px[1] - g_OverlapWidth), (i,j - g_OverlapWidth)  )
                error_Horizntl   = OverlapErrorHorizntl( (px[0] - g_OverlapWidth, px[1]), (i - g_OverlapWidth,j) )
                if error_Vertical  < g_ErrThreshold and error_Horizntl < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error_Vertical < g_ErrThreshold/2 and error_Horizntl < g_ErrThreshold/2:
                    return [(i,j)]
    return PixelList

#-----------------------------------------------------------------------------------------------#
#|                              Quilting and related Functions                                 |#
#-----------------------------------------------------------------------------------------------#
def SumOfSquaredDifferences_mono( offset_x, offset_y, outPx, inpPx ):
    block_0 = g_img_out[outPx[0] + offset_x, outPx[1] + offset_y]
    block_1 = g_img_inp[inpPx[0] + offset_x, inpPx[1] + offset_y]
    err_r = int(block_0[0]) - int(block_1[0])
    err_g = int(block_0[1]) - int(block_1[1])
    err_b = int(block_0[2]) - int(block_1[2])
    err_mono = (0.2125 * err_r**2) + (0.7154 * err_g**2) + (0.0721 * err_b**2);
    return err_mono

def SumOfSquaredDifferences_rgbmean( offset_x, offset_y, outPx, inpPx ):
    block_0 = g_img_out[outPx[0] + offset_x, outPx[1] + offset_y]
    block_1 = g_img_inp[inpPx[0] + offset_x, inpPx[1] + offset_y]
    err_r = int(block_0[0]) - int(block_1[0])
    err_g = int(block_0[1]) - int(block_1[1])
    err_b = int(block_0[2]) - int(block_1[2])
    return (err_r**2 + err_g**2 + err_b**2)/3.0


def SSD(offset_x, offset_y, outPx, inpPx):
    return SumOfSquaredDifferences_mono(offset_x, offset_y, outPx, inpPx)

# 2.1 Minimum Error Boundary Cut
def MinimumErrorVerticalCut(outPx, inpPx):
    # E(row,col) = e(row,col) + min[ E(row-1,col-1), E(row-1,col), E(row-1,col+1)]
    Cost = np.zeros((g_PatchSize, g_OverlapWidth))
    for col in range(g_OverlapWidth):
        for row in range(g_PatchSize):
            if row == g_PatchSize - 1:
                Cost[row,col] = SSD(row, col - g_OverlapWidth, outPx, inpPx)
            elif col == 0 :
                Cost[row,col] = SSD(row, col - g_OverlapWidth, outPx, inpPx) + \
                            min(
                                SSD(row +1, col - g_OverlapWidth, outPx, inpPx)
                               ,SSD(row +1, col - g_OverlapWidth +1, outPx, inpPx)
                               )
            elif col == g_OverlapWidth - 1:
                Cost[row,col] = SSD(row, col - g_OverlapWidth, outPx, inpPx) + \
                            min(
                                SSD(row +1, col - g_OverlapWidth, outPx, inpPx)
                               ,SSD(row +1, col - g_OverlapWidth -1 , outPx, inpPx)
                               )
            else:
                Cost[row,col] = SSD(row, col - g_OverlapWidth, outPx, inpPx) + \
                            min(
                                SSD(row +1, col - g_OverlapWidth -1, outPx, inpPx)
                               ,SSD(row +1, col - g_OverlapWidth, outPx, inpPx)
                               ,SSD(row +1, col - g_OverlapWidth +1, outPx, inpPx)
                               )
    return Cost

def MinimumErrorHorizntlCut(outPx, inpPx):
    Cost = np.zeros((g_OverlapWidth, g_PatchSize))
    for row in range( g_OverlapWidth ):
        for col in range( g_PatchSize ):
            if col == g_PatchSize - 1:
                Cost[row,col] = SSD(row - g_OverlapWidth, col, outPx, inpPx)
            elif row == 0:
                Cost[row,col] = SSD(row - g_OverlapWidth, col, outPx, inpPx) + \
                            min(
                                SSD(row - g_OverlapWidth, col + 1, outPx, inpPx)
                               ,SSD(row + 1 - g_OverlapWidth, col + 1, outPx, inpPx)
                               )
            elif row == g_OverlapWidth - 1:
                Cost[row,col] = SSD(row - g_OverlapWidth, col, outPx, inpPx) + \
                            min(
                                SSD(row - g_OverlapWidth, col + 1, outPx, inpPx)
                               ,SSD(row - 1 - g_OverlapWidth, col + 1, outPx, inpPx)
                               )
            else:
                Cost[row,col] = SSD(row - g_OverlapWidth, col, outPx, inpPx) + \
                            min(
                                SSD(row - g_OverlapWidth, col + 1, outPx, inpPx)
                               ,SSD(row - g_OverlapWidth + 1, col + 1, outPx, inpPx)
                               ,SSD(row - g_OverlapWidth - 1, col + 1, outPx, inpPx)
                               )
    return Cost

#---------------------------------------------------------------#
#|                  Finding Minimum Cost Path                  |#
#---------------------------------------------------------------#

def FindMinCostPathVertical(Cost):
    Boundary = np.zeros((g_PatchSize),np.int)
    ParentMatrix = np.zeros((g_PatchSize, g_OverlapWidth),np.int)
    for i in range(1, g_PatchSize):
        for j in range(g_OverlapWidth):
            if j == 0:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j+1] else j+1
            elif j == g_OverlapWidth - 1:
                ParentMatrix[i,j] = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
            else:
                curr_min = j if Cost[i-1,j] < Cost[i-1,j-1] else j-1
                ParentMatrix[i,j] = curr_min if Cost[i-1,curr_min] < Cost[i-1,j+1] else j+1
            Cost[i,j] += Cost[i-1, ParentMatrix[i,j]]
    minIndex = 0
    for j in range(1,g_OverlapWidth):
        minIndex = minIndex if Cost[g_PatchSize - 1, minIndex] < Cost[g_PatchSize - 1, j] else j
    Boundary[g_PatchSize-1] = minIndex

    for i in range(g_PatchSize - 1,0,-1):
        Boundary[i - 1] = ParentMatrix[i,Boundary[i]]

    return Boundary

def FindMinCostPathHorizntl(Cost):
    Boundary = np.zeros(( g_PatchSize),np.int)
    ParentMatrix = np.zeros((g_OverlapWidth, g_PatchSize),np.int)
    for j in range(1, g_PatchSize):
        for i in range(g_OverlapWidth):
            if i == 0:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i+1,j-1] else i + 1
            elif i == g_OverlapWidth - 1:
                ParentMatrix[i,j] = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
            else:
                curr_min = i if Cost[i,j-1] < Cost[i-1,j-1] else i - 1
                ParentMatrix[i,j] = curr_min if Cost[curr_min,j-1] < Cost[i-1,j-1] else i + 1
            Cost[i,j] += Cost[ParentMatrix[i,j], j-1]
    minIndex = 0
    for i in range(1,g_OverlapWidth):
        minIndex = minIndex if Cost[minIndex, g_PatchSize - 1] < Cost[i, g_PatchSize - 1] else i
    Boundary[g_PatchSize-1] = minIndex
    for j in range(g_PatchSize - 1,0,-1):
        Boundary[j - 1] = ParentMatrix[Boundary[j],j]
    return Boundary

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

def QuiltVertical(Boundary, outPx, inpPx):
    for i in range(g_PatchSize):
        for j in range(Boundary[i], 0, -1):
            g_img_out[outPx[0] + i, outPx[1] - j] = g_img_inp[ inpPx[0] + i, inpPx[1] - j ]
def QuiltHorizntl(Boundary, outPx, inpPx):
    for j in range(g_PatchSize):
        for i in range(Boundary[j], 0, -1):
            g_img_out[outPx[0] - i, outPx[1] + j] = g_img_inp[inpPx[0] - i, inpPx[1] + j]

def QuiltPatches( outPx, inpPx ):
    #check for top layer
    if outPx[0] == 0:
        Cost = MinimumErrorVerticalCut(outPx, inpPx)
        # Getting boundary to stitch
        Boundary = FindMinCostPathVertical(Cost)
        #Quilting Patches
        QuiltVertical(Boundary, outPx, inpPx)
    #check for leftmost layer
    elif outPx[1] == 0:
        Cost = MinimumErrorHorizntlCut(outPx, inpPx)
        #Boundary to stitch
        Boundary = FindMinCostPathHorizntl(Cost)
        #Quilting Patches
        QuiltHorizntl(Boundary, outPx, inpPx)
    #for pixel placed inside
    else:
        CostVertical = MinimumErrorVerticalCut(outPx, inpPx)
        CostHorizntl = MinimumErrorHorizntlCut(outPx, inpPx)
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        QuiltVertical(BoundaryVertical, outPx, inpPx)
        QuiltHorizntl(BoundaryHorizntl, outPx, inpPx)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

def FillImage( outPx, inpPx ):
    for i in range(g_PatchSize):
        for j in range(g_PatchSize):
            g_img_out[ outPx[0] + i, outPx[1] + j ] = g_img_inp[ inpPx[0] + i, inpPx[1] + j ]




def PickInitialPatch():
    randomPatch_x = randint(0, g_sample_height - g_PatchSize)
    randomPatch_y = randint(0, g_sample_width - g_PatchSize)
    for x in range(g_PatchSize):
        for y in range(g_PatchSize):
            g_img_out[x, y] = g_img_inp[randomPatch_x + x, randomPatch_y + y]
##########################################################################

print "file patch-size overlap-width threshold"
InputName = str(sys.argv[1])
g_PatchSize = int(sys.argv[2])
g_OverlapWidth = int(sys.argv[3])
g_InitialThreshold = float(sys.argv[4])

g_img_inp = cv2.imread(InputName)
g_sample_width = g_img_inp.shape[1]
g_sample_height = g_img_inp.shape[0]
print "input dim: %sx%s"%(g_sample_width,g_sample_height)

img_width  = g_sample_width*2
img_height = g_sample_height*2
print "output dim: %sx%s"%(img_width,img_height)

g_img_out = np.zeros((img_height,img_width,3), np.uint8)

PickInitialPatch()

#initializating next
g_GrowPatchLocation = (0,g_PatchSize)

patchesCompleted = 1

TotalPatches = (img_height*img_width)/(g_PatchSize**2)
while g_GrowPatchLocation[0] + g_PatchSize <= img_height:
    ThresholdConstant = g_InitialThreshold
    #set progress to zer0
    progress = 0
    while progress == 0:
        g_ErrThreshold = ThresholdConstant * g_PatchSize * g_OverlapWidth
        #Get Best matches for current pixel
        List = GetBestPatches(g_GrowPatchLocation)
        if len(List) > 0:
            progress = 1
            #Make A random selection from best fit pxls
            sampleMatch = List[ randint(0, len(List) - 1) ]
            FillImage( g_GrowPatchLocation, sampleMatch )
            #Quilt this with in curr location
            QuiltPatches( g_GrowPatchLocation, sampleMatch )
            #upadate cur pixel location
            g_GrowPatchLocation = (g_GrowPatchLocation[0], g_GrowPatchLocation[1] + g_PatchSize)
            if g_GrowPatchLocation[1] + g_PatchSize > img_width:
                g_GrowPatchLocation = (g_GrowPatchLocation[0] + g_PatchSize, 0)
        #if not progressed, increse threshold
        else:
            ThresholdConstant *= 1.1
    sys.stdout.write('\r')
    sys.stdout.write("PatchesCompleted:%d/%d | ThresholdConstant:%f" % ( patchesCompleted,TotalPatches, ThresholdConstant))
    sys.stdout.flush()
    patchesCompleted += 1

cv2.imwrite('out.png',g_img_out)

cv2.imshow('Sample Texture',g_img_inp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Generated Image',g_img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
