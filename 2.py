#!/usr/bin/python

# https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf

import cv2
import sys
import numpy as np
from random import randint
#---------------------------------------------------------------------------------------#
#|                      Best Fit Patch and related functions                           |#
#---------------------------------------------------------------------------------------#
def OverlapErrorVertical( iLeft, jLeft, inpLoc ):
    iRight,jRight = inpLoc
    OverlapErr = 0
    diff = np.zeros((3))
    for i in range( g_PatchSize ):
        for j in range( g_OverlapWidth ):
            diff[0] =  int(g_img_out[i + iLeft, j+ jLeft][0]) - int(g_img_inp[i + iRight, j + jRight][0])
            diff[1] =  int(g_img_out[i + iLeft, j+ jLeft][1]) - int(g_img_inp[i + iRight, j + jRight][1])
            diff[2] =  int(g_img_out[i + iLeft, j+ jLeft][2]) - int(g_img_inp[i + iRight, j + jRight][2])
            OverlapErr += (diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5
    return OverlapErr

def OverlapErrorHorizntl( iLeft, jLeft, rightPx ):
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

def GetBestPatches(px_row,px_col):
    PixelList = []
    #check for top layer
    if px_row == 0:
        for i in range(g_sample_height - g_PatchSize):
            for j in range(g_OverlapWidth, g_sample_width - g_PatchSize ):
                error = OverlapErrorVertical(
                          px_row
                        , px_col - g_OverlapWidth
                        , (i, j - g_OverlapWidth)
                        )
                if error  < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error < g_ErrThreshold/2:
                    return [(i,j)]
    #check for leftmost layer
    elif px_col == 0:
        for i in range(g_OverlapWidth, g_sample_height - g_PatchSize ):
            for j in range(g_sample_width - g_PatchSize):
                error = OverlapErrorHorizntl(
                          px_row - g_OverlapWidth
                        , px_col
                        , (i - g_OverlapWidth, j)
                        )
                if error  < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error < g_ErrThreshold/2:
                    return [(i,j)]
    #for pixel placed inside
    else:
        for i in range(g_OverlapWidth, g_sample_height - g_PatchSize):
            for j in range(g_OverlapWidth, g_sample_width - g_PatchSize):
                error_Vertical   = OverlapErrorVertical(
                                          px_row
                                        , px_col - g_OverlapWidth
                                        , (i,j - g_OverlapWidth)
                                        )
                error_Horizntl   = OverlapErrorHorizntl(
                                          px_row - g_OverlapWidth
                                        , px_col
                                        , (i - g_OverlapWidth,j)
                                        )
                if error_Vertical  < g_ErrThreshold and error_Horizntl < g_ErrThreshold:
                    PixelList.append((i,j))
                elif error_Vertical < g_ErrThreshold/2 and error_Horizntl < g_ErrThreshold/2:
                    return [(i,j)]
    return PixelList

#-----------------------------------------------------------------------------------------------#
#|                              Quilting and related Functions                                 |#
#-----------------------------------------------------------------------------------------------#
def SumOfSquaredDifferences_mono( offset_row, offset_col, outLoc_row, outLoc_col, inpLoc ):
    block_0 = g_img_out[outLoc_row + offset_row, outLoc_col + offset_col]
    block_1 = g_img_inp[inpLoc[0] + offset_row, inpLoc[1] + offset_col]
    err_r = int(block_0[0]) - int(block_1[0])
    err_g = int(block_0[1]) - int(block_1[1])
    err_b = int(block_0[2]) - int(block_1[2])
    err_mono = (0.2125 * err_r**2) + (0.7154 * err_g**2) + (0.0721 * err_b**2);
    return err_mono

def SumOfSquaredDifferences_rgbmean( offset_row, offset_col, outLoc_row,outLoc_col, inpLoc ):
    block_0 = g_img_out[outLoc_row + offset_row, outLoc_col + offset_col]
    block_1 = g_img_inp[inpLoc[0] + offset_row, inpLoc[1] + offset_col]
    err_r = block_0[0] - block_1[0]
    err_g = block_0[1] - block_1[1]
    err_b = block_0[2] - block_1[2]
    print err_r
    return (err_r**2 + err_g**2 + err_b**2)/3.0


def SSD(offset_row, offset_col, outLoc_row,outLoc_col, inpLoc):
    return SumOfSquaredDifferences_rgbmean(offset_row, offset_col, outLoc_row, outLoc_col, inpLoc)

# 2.1 Minimum Error Boundary Cut
def MinimalCumulativeVerticalCut(outLoc_row,outLoc_col, inpLoc):
    Cost = np.zeros((g_PatchSize, g_OverlapWidth))
    for col in range(g_OverlapWidth-1):
        for row in range(g_PatchSize):

            Ev = SSD(row, col, outLoc_row,outLoc_col, inpLoc)

            if row == g_PatchSize-1:
                Cost[row,col] = Ev
            elif col == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,   outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row+1, col+1, outLoc_row,outLoc_col, inpLoc)
                               )
            elif col == g_OverlapWidth - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,     outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row+1, col -1 , outLoc_row,outLoc_col, inpLoc)
                               )
            else:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col -1, outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row+1, col,    outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row+1, col +1, outLoc_row,outLoc_col, inpLoc)
                               )
    return Cost

def MinimalCumulativeHorizntlCut(outLoc_row,outLoc_col, inpLoc):
    # Eh(row,col) = min( Eh(row-1,col+1), Eh(row,col+1), Eh(row+1,col+1))
    Cost = np.zeros((g_OverlapWidth, g_PatchSize))
    for row in range( g_OverlapWidth ):
        for col in range(g_PatchSize-1):

            Eh = SSD(row, col, outLoc_row,outLoc_col, inpLoc)

            if col == g_PatchSize-1:
                Cost[row,col] = Eh
            if row == 0:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row +1, col+1, outLoc_row,outLoc_col, inpLoc)
                               )
            elif row == g_OverlapWidth - 1:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row -1, col+1, outLoc_row,outLoc_col, inpLoc)
                               )
            else:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row +1, col+1, outLoc_row,outLoc_col, inpLoc)
                               ,SSD(row -1, col+1, outLoc_row,outLoc_col, inpLoc)
                               )
    return Cost

# Finding Minimum Cost Trace
def FindMinCostPathVertical(Cost):
    return np.argmin(Cost,axis=1)

def FindMinCostPathHorizntl(Cost):
    return np.argmin(Cost,axis=0)

#---------------------------------------------------------------#
#|                      Quilting                               |#
#---------------------------------------------------------------#

def QuiltVertical(Boundary, outLoc_row, outLoc_col, inpLoc):
    for row in range(g_PatchSize):
        for col in range(Boundary[row], g_OverlapWidth):
            g_img_out[outLoc_row + row, outLoc_col + col] = g_img_inp[inpLoc[0] + row, inpLoc[1] + col ]

def QuiltHorizntl(Boundary, outLoc_row, outLoc_col, inpLoc):
    for col in range(g_PatchSize):
        for row in range(Boundary[col], g_OverlapWidth):
            g_img_out[outLoc_row + row, outLoc_col + col] = g_img_inp[inpLoc[0] + row, inpLoc[1] + col]

def QuiltPatches(outLoc_row, outLoc_col, inpLoc ):
    if outLoc_row == 0:
        Cost = MinimalCumulativeVerticalCut(outLoc_row, outLoc_col-g_OverlapWidth, inpLoc)
        Boundary = FindMinCostPathVertical(Cost)
        QuiltVertical(Boundary, outLoc_row,outLoc_col-g_OverlapWidth, inpLoc)
    elif outLoc_col == 0:
        Cost = MinimalCumulativeHorizntlCut(outLoc_row-g_OverlapWidth,outLoc_col, inpLoc)
        Boundary = FindMinCostPathHorizntl(Cost)
        QuiltHorizntl(Boundary, outLoc_row-g_OverlapWidth,outLoc_col, inpLoc)
    else:
        CostVertical = MinimalCumulativeVerticalCut(outLoc_row-g_OverlapWidth, outLoc_col-g_OverlapWidth, inpLoc)
        CostHorizntl = MinimalCumulativeHorizntlCut(outLoc_row-g_OverlapWidth, outLoc_col-g_OverlapWidth, inpLoc)
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        QuiltVertical(BoundaryVertical, outLoc_row-g_OverlapWidth, outLoc_col-g_OverlapWidth, inpLoc)
        QuiltHorizntl(BoundaryHorizntl, outLoc_row-g_OverlapWidth, outLoc_col-g_OverlapWidth, inpLoc)

#--------------------------------------------------------------------------------------------------------#
#                                   Growing Image Patch-by-patch                                        |#
#--------------------------------------------------------------------------------------------------------#

def FillImage( outLoc, inpLoc ):
    for i in range(g_PatchSize):
        for j in range(g_PatchSize):
            g_img_out[ outLoc[0] + i, outLoc[1] + j ] = g_img_inp[ inpLoc[0] + i, inpLoc[1] + j ]




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

g_GrowPatchLocation_row = 0
g_GrowPatchLocation_col = g_PatchSize
PickInitialPatch()

patchesCompleted = 1

TotalPatches = (img_height*img_width)/(g_PatchSize**2)
while g_GrowPatchLocation_row + g_PatchSize <= img_height:
    ThresholdConstant = g_InitialThreshold
    #set progress to zer0
    progress = 0
    while progress == 0:
        g_ErrThreshold = ThresholdConstant * g_PatchSize * g_OverlapWidth
        #Get Best matches for current pixel
        List = GetBestPatches(g_GrowPatchLocation_row, g_GrowPatchLocation_col)
        if len(List) > 0:
            progress = 1
            #Make A random selection from best fit pxls
            sampleMatch = List[ randint(0, len(List) - 1) ]
            #FillImage( g_GrowPatchLocation, sampleMatch )
            #Quilt this with in curr location
            QuiltPatches(g_GrowPatchLocation_row, g_GrowPatchLocation_col, sampleMatch )
            #upadate cur pixel location
            g_GrowPatchLocation_col += g_PatchSize
            if g_GrowPatchLocation_col + g_PatchSize > img_width:
                g_GrowPatchLocation_row += g_PatchSize
                g_GrowPatchLocation_col = 0
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
