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

def BlockSelect(block_left, block_above, texture, synth, block_size, overlap_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    err_v    = np.full((block_size,overlap_size), err_threshold)
    err_h    = np.full((overlap_size,block_size), err_threshold)

    if block_above is None and block_left is None:
        random_block_row = randint(0, texture_row - block_size)
        random_block_col = randint(0, texture_col - block_size)
        return [(random_block_row,random_block_col)]

    elif block_above is None:
        for row in range(0,texture_row-block_size):
            for col in range(0, texture_col-block_size):
                block_next = GetBlock(row, col, texture, block_size)

                error_Vertical = VerticalOverlapError(block_left, block_next, overlap_size)

                if np.less(error_Vertical,err_v/2).all():
                    return [(row,col)]
                if np.less(error_Vertical,err_v).all():
                    PixelList.append((row,col))
    elif block_left is None:
        for row in range(0, texture_row-block_size):
            for col in range(0,texture_col-block_size):
                block_next = GetBlock(row, col, texture, block_size)

                error_Horizntl = HorizntlOverlapError(block_above, block_next, overlap_size)

                if np.less(error_Horizntl,err_h/2).all():
                    return [(row,col)]
                if np.less(error_Horizntl,err_h).all():
                    PixelList.append((row,col))
    else:
        for row in range(0, texture_row-block_size):
            for col in range(0, texture_col-block_size):
                block_next = GetBlock(row, col, texture, block_size)

                error_Vertical = VerticalOverlapError(block_left, block_next, overlap_size)
                error_Horizntl = HorizntlOverlapError(block_above, block_next, overlap_size)

                if (np.less(error_Vertical,err_v/2).all()
                and np.less(error_Horizntl,err_h/2).all()
                   ):
                    return [(row,col)]

                if (np.less(error_Vertical,err_v).all()
                and np.less(error_Horizntl,err_h).all()
                   ):
                    PixelList.append((row,col))

    return PixelList

def diff(block_left, block_next, overlap_size):
    ovrlp_0 = block_left[-overlap_size:]
    ovrlp_1 = block_next[:overlap_size]
    err =  ovrlp_0 - ovrlp_1

    return err

def VerticalOverlapError(block_left, block_next, overlap_size):

    diff_r =  diff(block_left[:, :, 0].T, block_next[:, :, 0].T, overlap_size).T
    diff_g =  diff(block_left[:, :, 1].T, block_next[:, :, 1].T, overlap_size).T
    diff_b =  diff(block_left[:, :, 2].T, block_next[:, :, 2].T, overlap_size).T

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.uint8)
    return dist

def HorizntlOverlapError(block_above, block_next, overlap_size):
    diff_r =  diff(block_above[:, :, 0], block_next[:, :, 0], overlap_size)
    diff_g =  diff(block_above[:, :, 1], block_next[:, :, 1], overlap_size)
    diff_b =  diff(block_above[:, :, 2], block_next[:, :, 2], overlap_size)

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

def GetBlockAbove(off_row, off_col, texture, block_size):
    off_row -= block_size
    if off_row <0:
        return None
    return GetBlock(off_row, off_col, texture, block_size)

def GetBlockLeft(off_row, off_col, texture, block_size):
    off_col -= block_size
    if off_col <0:
        return None
    return GetBlock(off_row, off_col, texture, block_size)

def PutBlock(off_row, off_col, block, out):
    if block is None:
        print "empty block"
        raise

    block_row = block.shape[0]
    block_col = block.shape[1]

    out[off_row:off_row+block_row,off_col:off_col+block_col] = block

    cv2.imshow('Synthethised',out)
    cv2.waitKey(300)
    cv2.destroyAllWindows()
##########################################################################
def SSD(row, col, block_left, block_next):
    block_size = block_left.shape[0]
    diff_r =  block_left[row, col, 0].astype(np.int) - block_next[row, col, 0].astype(np.int)
    diff_g =  block_left[row, col, 1].astype(np.int) - block_next[row, col, 1].astype(np.int)
    diff_b =  block_left[row, col, 2].astype(np.int) - block_next[row, col, 2].astype(np.int)

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)/3.0).astype(np.int)
    return dist


# 2.1 Minimum Error Boundary Cut
def MinVerticalCumulativeError(block_left, block_next, overlap_size):
    block_row = block_next.shape[0]
    Cost = np.zeros((block_row, overlap_size))

    for col in range(0,overlap_size-1):
        for row in range(0,block_row):
            Ev = SSD(row, col, block_left, block_next)

            if row == block_row-1:
                Cost[row,col] = Ev
            elif col == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,    block_left, block_next)
                               ,SSD(row+1, col +1, block_left, block_next)
                               )
            elif col == overlap_size - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,    block_left, block_next)
                               ,SSD(row+1, col -1, block_left, block_next)
                               )
            else:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col -1, block_left, block_next)
                               ,SSD(row+1, col,    block_left, block_next)
                               ,SSD(row+1, col +1, block_left, block_next)
                               )
    return Cost

def MinHorizntlCumulativeError(block_above, block_next, overlap_size):
    # Eh(row,col) = min( Eh(row-1,col+1), Eh(row,col+1), Eh(row+1,col+1))
    block_col = block_next.shape[0]
    Cost = np.zeros((overlap_size, block_col))
    for row in range( overlap_size ):
        for col in range(block_col-1):

            Eh = SSD(row, col, block_above, block_next)

            if col == block_col-1:
                Cost[row,col] = Eh
            if row == 0:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, block_above, block_next)
                               ,SSD(row +1, col+1, block_above, block_next)
                               )
            elif row == overlap_size - 1:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, block_above, block_next)
                               ,SSD(row -1, col+1, block_above, block_next)
                               )
            else:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,    col+1, block_above, block_next)
                               ,SSD(row +1, col+1, block_above, block_next)
                               ,SSD(row -1, col+1, block_above, block_next)
                               )
    return Cost

def FindMinCostPathVertical(Cost):
    return np.argmin(Cost,axis=1)

def FindMinCostPathHorizntl(Cost):
    return np.argmin(Cost,axis=0)
############################################################################
def MinimumErrorCut(block_left, block_above, block_next, overlap_size):
    if block_above is None and block_left is None:
        return (None,None)
    elif block_above is None:
        Cost = MinVerticalCumulativeError(
                                          block_left
                                         ,block_next
                                         ,overlap_size
                                         )
        Boundary = FindMinCostPathVertical(Cost)
        return (Boundary,None)
    elif block_left is None:
        Cost = MinHorizntlCumulativeError(
                                          block_above
                                         ,block_next
                                         ,overlap_size
                                         )
        Boundary = FindMinCostPathHorizntl(Cost)
        return (None,Boundary)
    else:
        CostVertical = MinVerticalCumulativeError(
                                          block_left
                                         ,block_next
                                         ,overlap_size
                                         )
        CostHorizntl = MinHorizntlCumulativeError(
                                          block_above
                                         ,block_next
                                         ,overlap_size
                                         )
        BoundaryVertical = FindMinCostPathVertical(CostVertical)
        BoundaryHorizntl = FindMinCostPathHorizntl(CostHorizntl)
        return (BoundaryVertical,BoundaryHorizntl)
###################################################
def QuiltPatches(block_left, block_next, boundaries):
    if block_left is None:
        print "caca"
        return block_next
    else:
        return block_next
###################################################
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
    #block = PickBlock(0, 0, i_texture, block_size)
    #PutBlock(0, 0, block, o_synth)

    for o_cur_row in range(0, o_synth_row, block_size):
        for o_cur_col in range(0, o_synth_col, block_size):

            block_left  = GetBlockLeft(o_cur_row, o_cur_col, o_synth, block_size)
            block_above = GetBlockAbove(o_cur_row, o_cur_col, o_synth, block_size)
            best_blocks = BlockSelect(
                             block_left
                            ,block_above
                            ,i_texture
                            ,o_synth
                            ,block_size
                            ,overlap_size
                            ,err_threshold
                            )
            if len(best_blocks) > 0:
                best_location = random.choice(best_blocks)
                block_next = GetBlock(
                                      best_location[0]
                                     ,best_location[1]
                                     ,i_texture
                                     ,block_size
                                     )
                boundaries = MinimumErrorCut(
                                           block_left
                                          ,block_above
                                          ,block_next
                                          ,overlap_size
                                          )
                block_qlt = QuiltBlockAbove(
                                          block_above
                                         ,block_next
                                         ,boundaries
                                         )
                PutBlock(
                          o_cur_row
                         ,o_cur_col
                         ,block_qlt
                         ,o_synth
                        )



main()
