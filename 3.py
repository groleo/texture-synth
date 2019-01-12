#!/usr/bin/python

# https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf

import time
import math
import random
import cv2
import sys
import numpy as np
from random import randint
from scipy.spatial import distance

def PickInitialBlock(off_row, off_col, texture, block_side_length):
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]

    randomPatch_x = randint(0, texture_row - block_side_length)
    randomPatch_y = randint(0, texture_col - block_side_length)
    o_synth_array = np.zeros((block_side_length,block_side_length,3), np.uint8)
    for x in range(block_side_length):
        for y in range(block_side_length):
            o_synth_array[x, y] = texture[randomPatch_x + x, randomPatch_y + y]
    return o_synth_array

def BlockSelect(block_left, block_above, texture, synth, block_side_length, overlap_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    err_v    = np.full((block_side_length,overlap_size), err_threshold)
    err_h    = np.full((overlap_size,block_side_length), err_threshold)

    # Initial block
    if block_above is None and block_left is None:
        random_block_row = randint(0, texture_row - block_side_length)
        random_block_col = randint(0, texture_col - block_side_length)
        rv = GetBlock(texture
                     ,random_block_row
                     ,random_block_col
                     ,block_side_length
                     ,block_side_length
                     )
        print rv.shape[1]
        return [rv]

    elif block_above is None:
        for row in range(0,texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                ovrlp_left = GetVertOverlapPrev(block_left,overlap_size)
                ovrlp_next = GetVertOverlapNext(block_next,overlap_size)

                error_Vertical = VerticalOverlapError(ovrlp_left, ovrlp_next)

                if np.less(error_Vertical,err_v).all():
                    PixelList.append(block_next)
                elif np.less(error_Vertical,err_v/2).all():
                    return [block_next]

    elif block_left is None:
        for row in range(0, texture_row-block_side_length):
            for col in range(0,texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                error_Horizntl = HorizntlOverlapError(block_above, block_next, overlap_size)

                if np.less(error_Horizntl,err_h/2).all():
                    return [(row,col)]
                if np.less(error_Horizntl,err_h).all():
                    PixelList.append((row,col))
    else:
        for row in range(0, texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

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

def VerticalOverlapError(block_left, block_next):

    diff_r =  block_left[:, :, 0] - block_next[:, :, 0]
    diff_g =  block_left[:, :, 1] - block_next[:, :, 1]
    diff_b =  block_left[:, :, 2] - block_next[:, :, 2]

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.int)
    #dist = ((0.2125 * diff_r**2) + (0.7154 * diff_g**2) + (0.0721 * diff_b**2))
    #dist = ((diff_r**2 + diff_g**2 + diff_b**2)/3.0).astype(np.int)
    return dist

def HorizntlOverlapError(block_above, block_next, overlap_size):
    diff_r =  diff(block_above[:, :, 0], block_next[:, :, 0], overlap_size)
    diff_g =  diff(block_above[:, :, 1], block_next[:, :, 1], overlap_size)
    diff_b =  diff(block_above[:, :, 2], block_next[:, :, 2], overlap_size)

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.uint8)
    return dist


def PickBlock(off_row, off_col, texture, block_side_length):
    if off_row==0 and off_col==0:
        return PickInitialBlock(off_row, off_col, texture, block_side_length)

    texture_row = texture.shape[0]
    texture_col = texture.shape[1]

    randomPatch_x = randint(0, texture_row - block_side_length)
    randomPatch_y = randint(0, texture_col - block_side_length)
    o_synth_array = np.zeros((block_side_length,block_side_length,3), np.uint8)
    for x in range(block_side_length):
        for y in range(block_side_length):
            o_synth_array[x, y] = texture[randomPatch_x + x, randomPatch_y + y]

    return o_synth_array

def GetBlock(texture, off_row, off_col, block_row_height, block_col_width):
    out = np.zeros((block_row_height,block_col_width,3), np.uint8)
    out = texture[off_row:off_row+block_row_height,
                  off_col:off_col+block_col_width
                 ]
    return out

def GetAboveOverlap(texture, off_row, off_col, block_side_length, overlap_size):
    off_row -= overlap_size
    if off_row <0:
        return None
    return GetBlock(texture, off_row, off_col, overlap_size, block_side_length)

def GetLeftOverlap(texture, off_row, off_col, block_side_length, overlap_size):
    off_col -= overlap_size
    print "col:%s" % off_col
    if off_col < 0:
        print "no blocks found at the Left"
        return None
    return GetBlock(texture, off_row, off_col, block_side_length, overlap_size)

def PutBlock(out, off_row, off_col, block):
    if block is None:
        print "empty block"
        raise

    block_row = block.shape[0]
    block_col = block.shape[1]

    out[off_row:off_row+block_row,
        off_col:off_col+block_col
       ] = block


##########################################################################
# Sum of Squared Differences
def SSD(row, col, block_left, block_next):
    diff_r =  block_left[row, col, 0].astype(np.int) - block_next[row, col, 0].astype(np.int)
    diff_g =  block_left[row, col, 1].astype(np.int) - block_next[row, col, 1].astype(np.int)
    diff_b =  block_left[row, col, 2].astype(np.int) - block_next[row, col, 2].astype(np.int)

    #dist = ((diff_r**2 + diff_g**2 + diff_b**2)/3.0).astype(np.uint)
    #dist = ((0.2125 * diff_r**2) + (0.7154 * diff_g**2) + (0.0721 * diff_b**2))
    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.int)
    return dist

def GetVertOverlapPrev(block,overlap_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, overlap_size))
    ovrlp = block[:,-overlap_size:]
    return ovrlp

def GetVertOverlapNext(block,overlap_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, overlap_size))
    ovrlp = block[:,:overlap_size,:]
    return ovrlp

def display(name, block):
    cv2.imshow(name,block)
def displayw(name, block):
    cv2.imshow(name,block)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 2.1 Minimum Error Boundary Cut

#######################
# E
# . . . . @ . . .
# . . . @ . . . .
# . . . . @ . . .
# . . . @ . . . .
# . . . @ . . . .
# . . @ . . . . .
# . . . @ . . . .
# . . . . @ . . .
# . . . . @ . . .
# . . . . . @ . .
# . . . . . @ . .
# . . . . @ . . .
# . . . . @ . . .
#######################

# E[2,2] = e[2,2] + Min {E[1,1], E[1,2],E[1,3]}

# e[i,j] = (Block1[i,j] - Block2[i,j])**2
#
# E[i,j] = e[i,j] when i==1
# E[i,j] = e[i,j] + Min {E[i-1,j-1], E[i-1,j],E[i-1,j+1]} when i>1
def MinVerticalCumulativeError(ovrlp_left, ovrlp_next, overlap_size):
    ovrlp_row = ovrlp_next.shape[0]
    Cost = np.zeros((ovrlp_row, overlap_size))

    for row in range(ovrlp_row):
        for col in range(overlap_size):
            Ev = SSD(row, col, ovrlp_left, ovrlp_next)

            if row == 0:
                Cost[row,col] = Ev
                continue
            if col == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row-1, col,    ovrlp_left, ovrlp_next)
                               ,SSD(row-1, col +1, ovrlp_left, ovrlp_next)
                               )
            elif col == overlap_size - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row-1, col -1, ovrlp_left, ovrlp_next)
                               ,SSD(row-1, col,    ovrlp_left, ovrlp_next)
                               )
            else:
                Cost[row,col] = Ev + min(
                                SSD(row-1, col -1, ovrlp_left, ovrlp_next)
                               ,SSD(row-1, col,    ovrlp_left, ovrlp_next)
                               ,SSD(row-1, col +1, ovrlp_left, ovrlp_next)
                               )
    cost_min = Cost.min()
    cost_max = Cost.max()
    return (Cost - cost_min)/(cost_max - cost_min)

def MinVerticalCumulativeError_v1(ovrlp_left, ovrlp_next, overlap_size):
    ovrlp_row = ovrlp_next.shape[0]
    Cost = np.zeros((ovrlp_row, overlap_size))

    for col in range(overlap_size):
        for row in range(ovrlp_row):
            Ev = SSD(row, col, ovrlp_left, ovrlp_next)

            if row == ovrlp_row-1:
                Cost[row,col] = Ev
            elif col == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col,    ovrlp_left, ovrlp_next)
                               ,SSD(row+1, col +1, ovrlp_left, ovrlp_next)
                               )
            elif col == overlap_size - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col -1, ovrlp_left, ovrlp_next)
                               ,SSD(row+1, col,    ovrlp_left, ovrlp_next)
                               )
            else:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row+1, col -1, ovrlp_left, ovrlp_next)
                               ,SSD(row+1, col,    ovrlp_left, ovrlp_next)
                               ,SSD(row+1, col +1, ovrlp_left, ovrlp_next)
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
    cost_min = Cost.min()
    cost_max = Cost.max()
    return (Cost - cost_min)/(cost_max - cost_min)

def FindMinCostPathVertical(Cost):
    rows = Cost.shape[0]
    min_error_cut = np.zeros(rows,np.uint8)
    argmin_col = np.argmin(Cost[-1:],axis=1)[0]
    min_error_cut[rows-1] = argmin_col

    print Cost
    cv2.imshow("cost", Cost)
    for row in range(rows-1,0,-1):
        col = min_error_cut[row] -1
        nb_elem = 3
        if col < 0:
            col = 0
            nb_elem = 2

        look_into = Cost[row-1, col:col+nb_elem]
        argmin_col = np.argmin(look_into)
        min_error_cut[row-1] = col+argmin_col
        print "%s --> %s --> %s" %(col, look_into, col+argmin_col)
    return min_error_cut



def FindMinCostPathHorizntl(Cost):
    return np.argmin(Cost,axis=0)
############################################################################
def MinimumErrorCut(block_left, block_above, block_next, overlap_size):
    if block_above is None and block_left is None:
        return (None,None)

    elif block_above is None:
        print "case 3"
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
#def QuiltVertical(Boundary, outLoc_row, outLoc_col, inpLoc):
def QuiltBlockLeft(overlap_left, block_next, boundaries, o_cur_row, o_cur_col, o_synth_array, overlap_size):
    if overlap_left is None:
        print "QuiltBlockLeft:leftmost block"
        PutBlock(o_synth_array, o_cur_row, o_cur_col, block_next)
        return block_next.shape

    block_row = block_next.shape[0]
    block_col = block_next.shape[1]
    boundary  = boundaries[0]
    o_cur_col -= overlap_size

    quilted_block = np.copy(block_next)
    for row in range(block_row):
        print "left until:%s" % boundary[row]
        for col in range(block_col):
            if col < boundary[row]:
                quilted_block[row,col] = overlap_left[row,col]

    cv2.imshow("quilted_block", quilted_block)
    PutBlock(o_synth_array, o_cur_row, o_cur_col, quilted_block)
    return (block_next.shape[0],block_next.shape[1]-overlap_size)

def align_up(in_val, in_base):
    return int(in_base*math.ceil((1.0*in_val)/in_base))

###################################################
def main():
    input_fname       = str(sys.argv[1])
    block_side_length = int(sys.argv[2])
    overlap_size      = int(sys.argv[3])
    err_threshold     = float(sys.argv[4])
    print "overlap_size:%s" % overlap_size

    i_texture = cv2.imread(input_fname)

    i_texture_height_rows = i_texture.shape[0]
    i_texture_width_cols = i_texture.shape[1]

    o_synth_row = align_up(i_texture_height_rows,block_side_length)
    o_synth_col = align_up(i_texture_width_cols,block_side_length)

    required_blocks =  (o_synth_row*o_synth_col) / (block_side_length**2)

    o_synth_array = np.zeros((o_synth_row,o_synth_col,3), np.uint8)

    print "input dim : %sx%s"%(i_texture_width_cols,i_texture_height_rows)
    print "output dim: %sx%s (%s blocks)"%(o_synth_col,o_synth_row, required_blocks)

    o_cur_row = 0
    o_cur_col = 0
    while o_cur_row < o_synth_row:
        while o_cur_col < o_synth_col:
            left_overlap = GetLeftOverlap(
                                       o_synth_array
                                      ,o_cur_row
                                      ,o_cur_col
                                      ,block_side_length
                                      ,overlap_size
                                      )
            if left_overlap is not None:
                cv2.imshow('left_overlap',left_overlap)

            above_overlap = GetAboveOverlap(
                                        o_synth_array
                                       ,o_cur_row
                                       ,o_cur_col
                                       ,block_side_length
                                       ,overlap_size
                                       )
            best_blocks = BlockSelect(
                             left_overlap
                            ,above_overlap
                            ,i_texture
                            ,o_synth_array
                            ,block_side_length
                            ,overlap_size
                            ,err_threshold
                            )
            if not best_blocks:
                break
            block_next = random.choice(best_blocks)
            cv2.imshow('block_next',block_next)

            ovrlp_next = GetVertOverlapNext(block_next,overlap_size)
            cv2.imshow('overlap_next',ovrlp_next)
            boundaries = MinimumErrorCut(
                                       left_overlap
                                      ,above_overlap
                                      ,ovrlp_next
                                      ,overlap_size
                                      )
            ovrlp_qlt = QuiltBlockLeft(
                                      left_overlap
                                     ,block_next
                                     ,boundaries
                                     ,o_cur_row
                                     ,o_cur_col
                                     ,o_synth_array
                                     ,overlap_size
                                     )
            #o_cur_row += ovrlp_qlt[0]
            o_cur_col += ovrlp_qlt[1]
            img_zoomed=cv2.resize(o_synth_array, (256,256), interpolation=cv2.INTER_NEAREST )
            cv2.imshow('Generated Image',img_zoomed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        break
    cv2.imwrite('out.png',o_synth_array)

main()
