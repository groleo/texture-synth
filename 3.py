#!/usr/bin/python

# https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf

import time
import random
import cv2
import sys
import numpy as np
from random import randint
from scipy.spatial import distance

def PickInitialBlock(off_row, off_col, texture, block_size):
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]

    randomPatch_x = randint(0, texture_row - block_size)
    randomPatch_y = randint(0, texture_col - block_size)
    o_synth = np.zeros((block_size,block_size,3), np.uint8)
    for x in range(block_size):
        for y in range(block_size):
            o_synth[x, y] = texture[randomPatch_x + x, randomPatch_y + y]
    return o_synth

def GetBestBlocks(off_row, off_col, texture, synth, block_size, overlap_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    block_cur = synth[off_row:off_row+block_size, off_col:off_col+block_size]
    err_v    = np.full((block_size,overlap_size), err_threshold)
    err_h    = np.full((overlap_size,block_size), err_threshold)
    #check for top layer
    if off_row == 0:
        for row in range(0,texture_row-block_size):
            for col in range(0, texture_col-block_size):
                block_nxt = texture[row:row+block_size, col:col+block_size]
                error  = VerticalOverlapError(block_cur, block_nxt, overlap_size)
                if np.less(error,err_v).all():
                    PixelList.append((row,col))
                elif error < err_threshold/2:
                    return [(row,col)]
    #check for leftmost layer
    elif off_col == 0:
        for row in range(0, texture_row-block_size):
            for col in range(0,texture_col-block_size):
                block_nxt = texture[row:row+block_size, col:col+block_size]

                error = HorizntlOverlapError(block_cur, block_nxt, overlap_size)

                if np.less(error,err_h).all():
                    PixelList.append((row,col))
                elif error < err_threshold/2:
                    return [(row,col)]
    #for pixel placed inside
    else:
        for row in range(0, texture_row-block_size):
            for col in range(0, texture_col-block_size):
                block_nxt = texture[row:row+block_size, col:col+block_size]

                error_Vertical = VerticalOverlapError(block_cur, block_nxt, overlap_size)
                error_Horizntl = HorizntlOverlapError(block_cur, block_nxt, overlap_size)

                if (np.less(error_Vertical,err_v).all()
                and np.less(error_Horizntl,err_h).all()
                   ):
                    PixelList.append((row,col))
                elif (np.less(error_Vertical,err_v).all()
                and   np.less(error_Horizntl,err_h).all()
                   ):
                    return [(row,col)]
    return PixelList

def diff(block_cur, block_nxt, overlap_size):
    ovrlp_0 = block_cur[-overlap_size:]
    ovrlp_1 = block_nxt[:overlap_size]
    err =  ovrlp_0 - ovrlp_1

    return err

def VerticalOverlapError(block_cur, block_nxt, overlap_size):

    diff_r =  diff(block_cur[:, :, 0].T, block_nxt[:, :, 0].T, overlap_size).T
    diff_g =  diff(block_cur[:, :, 1].T, block_nxt[:, :, 1].T, overlap_size).T
    diff_b =  diff(block_cur[:, :, 2].T, block_nxt[:, :, 2].T, overlap_size).T

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.uint8)
    return dist

def HorizntlOverlapError(block_cur, block_nxt, overlap_size):
    diff_r =  diff(block_cur[:, :, 0], block_nxt[:, :, 0], overlap_size)
    diff_g =  diff(block_cur[:, :, 1], block_nxt[:, :, 1], overlap_size)
    diff_b =  diff(block_cur[:, :, 2], block_nxt[:, :, 2], overlap_size)

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.uint8)
    return dist


def PickBlock(off_row, off_col, texture, block_size):
    if off_row==0 and off_col==0:
        return PickInitialBlock(off_row, off_col, texture, block_size)

    texture_row = texture.shape[0]
    texture_col = texture.shape[1]

    randomPatch_x = randint(0, texture_row - block_size)
    randomPatch_y = randint(0, texture_col - block_size)
    o_synth = np.zeros((block_size,block_size,3), np.uint8)
    for x in range(block_size):
        for y in range(block_size):
            o_synth[x, y] = texture[randomPatch_x + x, randomPatch_y + y]

    return o_synth

def GetBlock(off_row, off_col, texture, block_size):
    out = np.zeros((block_size,block_size,3), np.uint8)
    out = texture[off_row:off_row+block_size,off_col:off_col+block_size]
    return out

def CopyBlock(off_row, off_col, block, out):
    block_row = block.shape[0]
    block_col = block.shape[1]

    out[off_row:off_row+block_row,off_col:off_col+block_col] = block

    cv2.imshow('Synthethised',out)
    cv2.waitKey(300)
    cv2.destroyAllWindows()
##########################################################################
def SumOfSquaredDifferences_mono( offset_row, offset_col, off_row, off_col, inpLoc ):
    block_0 = g_img_out[off_row + offset_row, off_col + offset_col]
    block_1 = g_img_inp[inpLoc[0] + offset_row, inpLoc[1] + offset_col]
    err_r = int(block_0[0]) - int(block_1[0])
    err_g = int(block_0[1]) - int(block_1[1])
    err_b = int(block_0[2]) - int(block_1[2])
    err_mono = (0.2125 * err_r**2) + (0.7154 * err_g**2) + (0.0721 * err_b**2);
    return err_mono

def SumOfSquaredDifferences_rgbmean( offset_row, offset_col, off_row,off_col, block_1 ):
def SumOfSquaredDifferences_rgbmean( block_cur, block_next, overlap_size ):

    err_r = block_cur[0] - block_1[0]
    err_g = block_cur[1] - block_1[1]
    err_b = block_0[2] - block_1[2]
    print err_r
    return (err_r**2 + err_g**2 + err_b**2)/3.0


def SSD(offset_row, offset_col, off_row,off_col, inpLoc):
    return SumOfSquaredDifferences_rgbmean(offset_row
                                          ,offset_col
                                          ,off_row
                                          ,off_col
                                          ,inpLoc
                                          )

# 2.1 Minimum Error Boundary Cut
def MinimumErrorVerticalCut(off_row, off_col, block, overlap_size):
    block_row = block.shape[0]
    Cost = np.zeros((block_row, overlap_size))
    for col in range(0,overlap_size-1):
        for row in range(0,block_row):

            Ev = SSD(row, col, off_row, off_col, block)

            if row == block_size-1:
                Cost[row,col] = Ev
            elif col == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,   off_row,off_col, inpLoc)
                               ,SSD(row+1, col+1, off_row,off_col, inpLoc)
                               )
            elif col == overlap_size - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,     off_row,off_col, inpLoc)
                               ,SSD(row+1, col -1 , off_row,off_col, inpLoc)
                               )
            else:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col -1, off_row,off_col, inpLoc)
                               ,SSD(row+1, col,    off_row,off_col, inpLoc)
                               ,SSD(row+1, col +1, off_row,off_col, inpLoc)
                               )
    return Cost

def MinimumErrorHorizntlCut(off_row,off_col, inpLoc):
    # Eh(row,col) = min( Eh(row-1,col+1), Eh(row,col+1), Eh(row+1,col+1))
    Cost = np.zeros((overlap_size, block_size))
    for row in range( overlap_size ):
        for col in range(block_size-1):

            Eh = SSD(row, col, off_row,off_col, inpLoc)

            if col == block_size-1:
                Cost[row,col] = Eh
            if row == 0:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, off_row,off_col, inpLoc)
                               ,SSD(row +1, col+1, off_row,off_col, inpLoc)
                               )
            elif row == overlap_size - 1:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, off_row,off_col, inpLoc)
                               ,SSD(row -1, col+1, off_row,off_col, inpLoc)
                               )
            else:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, off_row,off_col, inpLoc)
                               ,SSD(row +1, col+1, off_row,off_col, inpLoc)
                               ,SSD(row -1, col+1, off_row,off_col, inpLoc)
                               )
    return Cost

# Finding Minimum Cost Trace
def FindMinCostPathVertical(Cost):
    return np.argmin(Cost,axis=1)

def FindMinCostPathHorizntl(Cost):
    return np.argmin(Cost,axis=0)
############################################################################
def QuiltVertical(Boundary, off_row, off_col, inpLoc):
    for row in range(block_size):
        for col in range(Boundary[row], overlap_size):
            g_img_out[off_row + row, off_col + col] = g_img_inp[inpLoc[0] + row, inpLoc[1] + col ]

def QuiltHorizntl(Boundary, off_row, off_col, inpLoc):
    for col in range(block_size):
        for row in range(Boundary[col], overlap_size):
            g_img_out[off_row + row, off_col + col] = g_img_inp[inpLoc[0] + row, inpLoc[1] + col]

def QuiltPatches(off_row, off_col, best_block,overlap_size,block_size):
    if off_row == 0:
        Cost = MinimumErrorVerticalCut(off_row
                                      ,off_col
                                      ,best_block
                                      ,overlap_size
                                      )
        sys.exit(0)
        Boundary = FindMinCostPathVertical(Cost)
        QuiltVertical(Boundary, off_row,off_col-overlap_size, inpLoc)
    elif off_col == 0:
        Cost = MinimumErrorHorizntlCut(off_row-overlap_size,off_col, inpLoc)
        Boundary = FindMinCostPathHorizntl(Cost)
        QuiltHorizntl(Boundary, off_row-overlap_size,off_col, inpLoc)
    else:
        CostVertical = MinimumErrorVerticalCut(off_row-overlap_size, off_col-overlap_size, inpLoc)
        CostHorizntl = MinimumErrorHorizntlCut(off_row-overlap_size, off_col-overlap_size, inpLoc)
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        QuiltVertical(BoundaryVertical, off_row-overlap_size, off_col-overlap_size, inpLoc)
        QuiltHorizntl(BoundaryHorizntl, off_row-overlap_size, off_col-overlap_size, inpLoc)

def main():
    input_fname  = str(sys.argv[1])
    block_size   = int(sys.argv[2])
    overlap_size = int(sys.argv[3])
    err_threshold = float(sys.argv[4])

    i_texture = cv2.imread(input_fname)
    i_texture_row = i_texture.shape[0]
    i_texture_col = i_texture.shape[1]
    required_blocks = (i_texture_row*i_texture_col)/(block_size**2)

    o_synth_col = i_texture_col*2
    o_synth_row = i_texture_row*2
    o_synth = np.zeros((o_synth_row,o_synth_col,3), np.uint8)
    print "input dim : %sx%s"%(i_texture_col,i_texture_row)
    print "output dim: %sx%s (%s blocks)"%(o_synth_col,o_synth_row, required_blocks)
    block = PickBlock(0, 0, i_texture, block_size)
    CopyBlock(0, 0, block, o_synth)

    for o_cur_row in range(0, o_synth_row, block_size):
        for o_cur_col in range(0, o_synth_col, block_size):

            best_blocks = GetBestBlocks(o_cur_row
                         ,o_cur_col
                         ,i_texture
                         ,o_synth
                         ,block_size
                         ,overlap_size
                         ,err_threshold
                         )
            if len(best_blocks) > 0:
                best_location = random.choice(best_blocks)
                best_block = GetBlock(best_location[0]
                                     ,best_location[1]
                                     ,i_texture
                                     ,block_size
                                     )
                QuiltPatches(o_cur_row
                            ,o_cur_col
                            ,best_block
                            ,overlap_size
                            ,block_size
                            )



main()
