#!/usr/bin/python

# Copyright 2020 Adrian-Marius Negreanu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# https://people.eecs.berkeley.edu/~efros/research/quilting/quilting.pdf

import math
import cv2
import sys
import numpy as np
import random
from scipy.spatial import distance

def GetLeftStrip(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    #ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:,:ovrlp_size,:]
    return ovrlp


def GetRightStrip(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    #ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:,-ovrlp_size:]
    return ovrlp

def GetBottomStrip(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    #ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[-ovrlp_size:]
    return ovrlp

def GetTopStrip(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    #ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:ovrlp_size]
    return ovrlp


def GetBlock(texture, off_row, off_col, block_row_height, block_col_width):
    if ( off_row < 0
      or off_col < 0
       ):
        return None

    #out = np.zeros((block_row_height,block_col_width,3), np.uint8)
    out = texture[off_row:off_row+block_row_height,
                  off_col:off_col+block_col_width
                 ]
    return out


def PutBlock(out, off_row, off_col, block):
    if block is None:
        print "PutBlock: empty block"
        raise

    if ( off_row < 0
      or off_col < 0
       ):
        print "PutBlock: invalid coords"
        raise

    block_row = block.shape[0]
    block_col = block.shape[1]

    out[off_row:off_row+block_row,
        off_col:off_col+block_col
       ] = block


def BlockSelect_1(block_left, block_above, texture, synth, block_side_length, ovrlp_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    err_v    = np.full((block_side_length,ovrlp_size), err_threshold*3)
    err_h    = np.full((ovrlp_size,block_side_length), err_threshold*3)

    # Initial block
    if block_above is None and block_left is None:
        random_block_row = random.randint(0, texture_row - block_side_length)
        random_block_col = random.randint(0, texture_col - block_side_length)
        rv = GetBlock(texture
                     ,random_block_row
                     ,random_block_col
                     ,block_side_length
                     ,block_side_length
                     )
        return [rv]

    elif block_above is None:
        for row in range(0,texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)
                VertCost = MinVertCumulativeError(
                                                block_left
                                                ,block_next
                                                ,ovrlp_size
                                                )

                if np.less(VertCost,err_v/2).all():
                    return [block_next]
                if np.less(VertCost,err_v).all():
                    PixelList.append(block_next)

    elif block_left is None:
        for row in range(0, texture_row-block_side_length):
            for col in range(0,texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)
                HorzCost = MinHorzCumulativeError(
                                                block_above
                                                ,block_next
                                                ,ovrlp_size
                                                )

                if np.less(HorzCost,err_h/2).all():
                    return [block_next]
                if np.less(HorzCost,err_h).all():
                    PixelList.append(block_next)
    else:
        for row in range(0, texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)
                VertCost = MinVertCumulativeError(
                                                block_left
                                                ,block_next
                                                ,ovrlp_size
                                                )
                HorzCost = MinHorzCumulativeError(
                                                block_above
                                                ,block_next
                                                ,ovrlp_size
                                                )

                if (np.less(VertCost,err_v/2).all()
                and np.less(HorzCost,err_h/2).all()
                   ):
                    return [block_next]

                if (np.less(VertCost,err_v).all()
                and np.less(HorzCost,err_h).all()
                   ):
                    PixelList.append(block_next)
    return PixelList

def Normalize(Cost):
    cost_min = Cost.min()
    cost_max = Cost.max()
    if cost_min == 0 and cost_max == 0:
        return Cost
    return (Cost - cost_min)/(cost_max - cost_min)

def SqrBlockDistance(block_A, block_B):
    diff_r =  block_A[:, :, 0].astype(np.int) - block_B[:, :, 0]
    diff_g =  block_A[:, :, 1].astype(np.int) - block_B[:, :, 1]
    diff_b =  block_A[:, :, 2].astype(np.int) - block_B[:, :, 2]

    #dist = (diff_r**2 + diff_g**2 + diff_b**2)#**0.5
    #dist = (diff_r**2 + diff_g**2 + diff_b**2)/3.0
    dist = (0.2125*(diff_r**2) + 0.7154*(diff_g**2) + 0.0721*(diff_b**2))#**0.5
    return dist

# Sum of Squared Differences
def SSD(row, col, block_A, block_B):
    diff_r =  block_A[row, col, 0].astype(np.int) - block_B[row, col, 0]
    diff_g =  block_A[row, col, 1].astype(np.int) - block_B[row, col, 1]
    diff_b =  block_A[row, col, 2].astype(np.int) - block_B[row, col, 2]

    #dist = (diff_r**2 + diff_g**2 + diff_b**2)#**0.5
    #dist = (diff_r**2 + diff_g**2 + diff_b**2)/3.0
    dist = (0.2125*(diff_r**2) + 0.7154*(diff_g**2) + 0.0721*(diff_b**2))#**0.5
    return dist

def SSD_2(row, col, block_A, block_B):
    s = np.sum((block_A[:,:,0:3] - block_B[:,:,0:3])**2)
    return s

def BlockSelect(block_left, block_above, texture, block_side_length, ovrlp_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    err_v    = np.full((block_side_length,ovrlp_size), err_threshold**2)
    err_h    = np.full((ovrlp_size,block_side_length), err_threshold**2)

    # Initial block
    if block_above is None and block_left is None:
        random_block_row = random.randint(0, texture_row - block_side_length)
        random_block_col = random.randint(0, texture_col - block_side_length)
        rv = GetBlock(texture
                     ,random_block_row
                     ,random_block_col
                     ,block_side_length
                     ,block_side_length
                     )
        return [rv]

    elif block_above is None:
        ovrlp_left = GetRightStrip(block_left,ovrlp_size)
        for row in range(0,texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)
                ovrlp_next = GetLeftStrip(block_next,ovrlp_size)
                error_Vertical = SqrBlockDistance(ovrlp_left, ovrlp_next)

                if np.less(error_Vertical,err_v).all():
                    PixelList.append(block_next)

    elif block_left is None:
        ovrlp_above = GetBottomStrip(block_above,ovrlp_size)
        for row in range(0, texture_row-block_side_length):
            for col in range(0,texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)
                ovrlp_next = GetTopStrip(block_next,ovrlp_size)
                error_Horizntl = SqrBlockDistance(ovrlp_above, ovrlp_next)

                if np.less(error_Horizntl,err_h).all():
                    PixelList.append(block_next)
    else:
        ovrlp_above = GetBottomStrip(block_above,ovrlp_size)
        ovrlp_left = GetRightStrip(block_left,ovrlp_size)
        for row in range(0, texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                ovrlp_vnext = GetLeftStrip(block_next,ovrlp_size)
                error_Vertical = SqrBlockDistance(ovrlp_left, ovrlp_vnext)
                if not np.less(error_Vertical,err_v).all():
                 #   col += ovrlp_size/2
                    continue

                ovrlp_hnext = GetTopStrip(block_next,ovrlp_size)
                error_Horizntl = SqrBlockDistance(ovrlp_above, ovrlp_hnext)
                if not np.less(error_Horizntl,err_h).all():
                #    col += ovrlp_size/2
                    continue

                PixelList.append(block_next)

    return PixelList





##########################################################################

# 2.1 Minimum Error Vertical Boundary Cut
###########################
# . . . . E . . . . . . . .
# . . . V . . . . . . . . .
# . . . . V . . . . . . . .
# . . . V . . . . . . . . .
# . . . V . . . . . . . . .
# . . V . . . . . . . . . .
# . . . V . . . . . . . . .
# . . . . V . . . . . . . .
# . . . . V . . . . . . . .
# . . . . . V . . . . . . .
# . . . . . V . . . . . . .
# . . . . V . . . . . . . .
# . . . . V . . . . . . . .
###########################

# E[2,2] = e[2,2] + Min {E[1,1], E[1,2],E[1,3]}

# e[i,j] = (Block1[i,j] - Block2[i,j])**2
#
# E[i,j] = e[i,j] when i==1
# E[i,j] = e[i,j] + Min {E[i-1,j-1], E[i-1,j],E[i-1,j+1]} when i>1
def MinVertCumulativeError(ovrlp_left, ovrlp_next, ovrlp_size):
    ovrlp_row = ovrlp_next.shape[0]
    Cost = np.zeros((ovrlp_row, ovrlp_size))

    for row in range(ovrlp_row):
        for col in range(ovrlp_size):
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
            elif col == ovrlp_size - 1:
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
    return Cost


# 2.1 Minimum Error Horizontal Boundary Cut
###########################
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . H . . . .
# . . . . . . H H . H H . .
# . H H . . H . . . . . H H
# E . . H H . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
###########################

# E[2,2] = e[2,2] + Min {E[1,1], E[1,2],E[1,3]}

# e[i,j] = (Block1[i,j] - Block2[i,j])**2
#
# E[i,j] = e[i,j] when i==1
# E[i,j] = e[i,j] + Min {E[i-1,j-1], E[i-1,j],E[i-1,j+1]} when i>1
# row,col
def MinHorzCumulativeError(ovrlp_above, ovrlp_next, ovrlp_size):
    ovrlp_col = ovrlp_next.shape[1]
    Cost = np.zeros((ovrlp_size, ovrlp_col))

    for col in range(ovrlp_col):
        for row in range(ovrlp_size):
            Eh = SSD(row, col, ovrlp_above, ovrlp_next)

            if col == 0:
                Cost[row,col] = Eh
                continue

            if row == 0 :
                Cost[row,col] = Eh + \
                            min(
                                SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row+1, col -1, ovrlp_above, ovrlp_next)
                               )
            elif row == ovrlp_size - 1:
                Cost[row,col] = Eh + \
                            min(
                                SSD(row-1, col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               )
            else:
                Cost[row,col] = Eh + min(
                                SSD(row-1, col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row+1, col -1, ovrlp_above, ovrlp_next)
                               )
    return Cost

# 2.1 Minimum Error Diagonal Boundary Cut
###########################
# E . . . . . . . . . . . .
# . o . . . . . . . . . . .
# . . o . . . . . . . . . .
# . . . o . . . . . . . . .
# . . . . o . . . . . . . .
# . . . . . o . . . . . . .
# . . . . . . o . . . . . .
# . . . . . . . o . . . . .
# . . . . . . . . o . . . .
# . . . . . . . . . o . . .
# . . . . . . . . . . o . .
# . . . . . . . . . . . o .
# . . . . . . . . . . . . o
###########################

# E[2,2] = e[2,2] + Min {E[1,1], E[1,2],E[1,3]}

# e[i,j] = (Block1[i,j] - Block2[i,j])**2
#
# E[i,j] = e[i,j] when i==1
# E[i,j] = e[i,j] + Min {E[i-1,j-1], E[i-1,j],E[i-1,j+1]} when i>1
# row,col
def MinDiagCumulativeError(ovrlp_above, ovrlp_next, ovrlp_size):
    ovrlp_col = ovrlp_next.shape[1]
    Cost = np.zeros((ovrlp_size, ovrlp_col))

    for col in range(ovrlp_size):
        for row in range(ovrlp_size):
            Ev = SSD(row, col, ovrlp_above, ovrlp_next)

            if col == 0:
                Cost[row,col] = Ev
                continue

            if row == 0 :
                Cost[row,col] = Ev + \
                            min(
                                SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row+1, col -1, ovrlp_above, ovrlp_next)
                               )
            elif row == ovrlp_size - 1:
                Cost[row,col] = Ev + \
                            min(
                                SSD(row-1, col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               )
            else:
                Cost[row,col] = Ev + min(
                                SSD(row-1, col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row,   col -1, ovrlp_above, ovrlp_next)
                               ,SSD(row+1, col -1, ovrlp_above, ovrlp_next)
                               )
    return Cost

def FindMinCostVertPath(Cost):
    rows = Cost.shape[0]
    min_error_cut = np.zeros(rows,np.int)
    argmin_col = np.argmin(Cost[-1:],axis=1)[0]
    min_error_cut[rows-1] = argmin_col

    #cv2.imshow("vert_cost", Cost)
    for row in range(rows-1,0,-1):
        col = min_error_cut[row] -1
        nb_elem = 3
        if col < 0:
            col = 0
            nb_elem = 2

        look_into = Cost[row-1, col:col+nb_elem]
        argmin_col = np.argmin(look_into)
        min_error_cut[row-1] = col + argmin_col
        #print "VERT: %s --> %s --> %s" %(col, look_into, col+argmin_col)
    return min_error_cut



def FindMinCostHorzPath(Cost):
    cols = Cost.shape[1]
    min_error_cut = np.zeros(cols,np.int)

    argmin_row = np.argmin(Cost[:,cols-1])
    min_error_cut[cols-1] = argmin_row

    #cv2.imshow("horz_cost", Cost)
    for col in range(cols-1,0,-1):
        row = min_error_cut[col] -1
        nb_elem = 3
        if row < 0:
            row = 0
            nb_elem = 2

        look_into = Cost[row:row+nb_elem, col-1]
        argmin_row = np.argmin(look_into)
        min_error_cut[col-1] = row + argmin_row
        #print "HORZ: %s --> %s --> %s" %(row, look_into, row+argmin_row)
    return min_error_cut

############################################################################
def MinimumErrorCuts(block_left, block_above, block_vert_next, block_horz_next, ovrlp_size):
    if block_above is None and block_left is None:
        return (None,None)

    elif block_above is None:
        VertCost = Normalize(MinVertCumulativeError(
                                          block_left
                                         ,block_vert_next
                                         ,ovrlp_size
                                         ))
        VertBoundary = FindMinCostVertPath(VertCost)
        return (VertBoundary,None)

    elif block_left is None:
        HorzCost = Normalize(MinHorzCumulativeError(
                                          block_above
                                         ,block_horz_next
                                         ,ovrlp_size
                                         ))
        HorzBoundary = FindMinCostHorzPath(HorzCost)
        return (None,HorzBoundary)

    else:
        VertCost = Normalize(MinVertCumulativeError(
                                          block_left
                                         ,block_vert_next
                                         ,ovrlp_size
                                         ))
        HorzCost = Normalize(MinHorzCumulativeError(
                                          block_above
                                         ,block_horz_next
                                         ,ovrlp_size
                                         ))
        VertBoundary = FindMinCostVertPath(VertCost)
        HorzBoundary = FindMinCostHorzPath(HorzCost)
        return (VertBoundary,HorzBoundary)

###################################################
###########################
# C C C C H H H H H H H H H H H H
# C C C C H H H h H H H H H H H H
# C C C C H H H H H H H H H H H H
# C C C C H H H H H H H H H H H H
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
# V V V V . . . . . . . . . . . .
###########################
def QuiltBlocks(ovrlp_left, ovrlp_above, block_next, boundaries, cur_synth_row, cur_synth_col, o_synth_array, ovrlp_size):
    if ovrlp_left is None and ovrlp_above is None:
        PutBlock(o_synth_array, cur_synth_row, cur_synth_col, block_next)
        return block_next.shape

    block_row = block_next.shape[0]
    block_col = block_next.shape[1]

    boundary_vert  = boundaries[0]
    boundary_horz  = boundaries[1]

    quilted_block = np.copy(block_next)

    # quilt a vertical strip
    if boundary_vert is not None:
        for row in range(block_row):
            for col in range(block_col):
                if col < boundary_vert[row]:
                    quilted_block[row,col] = ovrlp_left[row,col]#*[1.5,1,1]
        cur_synth_col -= ovrlp_size
        block_col -= ovrlp_size

    # quilt a horizontal strip
    if boundary_horz is not None:
        for col in range(block_col):
            #print "horiz until:%s" % boundary_horz[col]
            for row in range(block_row):
                if row < boundary_horz[col]:
                    quilted_block[row,col] = ovrlp_above[row,col]#*[1.5,1,1]
        cur_synth_row -= ovrlp_size
        block_row -= ovrlp_size


#    cv2.imshow("quilted_block", quilted_block)
    PutBlock(o_synth_array, cur_synth_row, cur_synth_col, quilted_block)
    return (block_row,block_col)


def align_up(in_val, in_base):
    return int(in_base*math.ceil((1.0*in_val)/in_base))


###################################################
def main():
    block_side_length = int(sys.argv[1])
    ovrlp_size        = int(sys.argv[2])
    err_threshold     = float(sys.argv[3])
    input_fname       = str(sys.argv[4])
    output_fname      = str(sys.argv[5])

    i_texture = cv2.imread(input_fname)

    i_texture_height_rows = i_texture.shape[0]
    i_texture_width_cols = i_texture.shape[1]

    if (block_side_length > i_texture_height_rows
     or block_side_length > i_texture_width_cols
       ):
        print "block size(%s) is bigger than input texture(%sx%s)" % (block_side_length, i_texture_width_cols, i_texture_height_rows)
        sys.exit(1)

    o_synth_row = align_up(i_texture_height_rows*1.5,block_side_length-ovrlp_size) + block_side_length
    o_synth_col = align_up(i_texture_width_cols*1.5,block_side_length-ovrlp_size) + block_side_length

    o_synth_array = np.zeros((o_synth_row,o_synth_col,3), np.uint8)

    print "ovrlp_size:%s" % ovrlp_size
    print "input dim : %sx%s"%(i_texture_width_cols,i_texture_height_rows)
    print "output dim: %sx%s"%(o_synth_col,o_synth_row)
    #sys.exit(0)
    cur_synth_row = 0
    while cur_synth_row < o_synth_row:
        cur_synth_col = 0
        while cur_synth_col < o_synth_col:
#            img_zoomed=cv2.resize(o_synth_array, (256,256), interpolation=cv2.INTER_NEAREST )
#            cv2.imshow('Synth Array Before',img_zoomed)

            block_left = GetBlock(
                                    o_synth_array
                                    ,cur_synth_row - ovrlp_size if cur_synth_row else cur_synth_row
                                    ,cur_synth_col - block_side_length
                                    ,block_side_length
                                    ,block_side_length
                                    )
            block_above = GetBlock(
                                    o_synth_array
                                    ,cur_synth_row - block_side_length
                                    ,cur_synth_col - ovrlp_size if cur_synth_col else cur_synth_col
                                    ,block_side_length
                                    ,block_side_length
                                    )

            block_prev_vert_ovrlp = GetRightStrip(
                                    block_left
                                    ,ovrlp_size
                                    )
            block_prev_horz_ovrlp = GetBottomStrip(
                                    block_above
                                    ,ovrlp_size
                                    )
            best_blocks = None
            cur_err_threshold = err_threshold
            while True:
                best_blocks = BlockSelect(
                                 block_prev_vert_ovrlp
                                ,block_prev_horz_ovrlp
                                ,i_texture
                                ,block_side_length
                                ,ovrlp_size
                                ,cur_err_threshold
                                )
                if not best_blocks:
                    print "no best blocks w/ err-threshold:%s" % cur_err_threshold
                    cur_err_threshold += 5
                    continue
                break

            block_next = random.choice(best_blocks)

            block_next_vert_ovrlp = GetLeftStrip(block_next ,ovrlp_size)
            block_next_horz_ovrlp = GetTopStrip(block_next ,ovrlp_size)

#            if block_prev_vert_ovrlp is not None:
#                cv2.imshow('block_prev_vert_ovrlp',block_prev_vert_ovrlp)
#            if block_prev_horz_ovrlp is not None:
#                cv2.imshow('block_prev_horz_ovrlp',block_prev_horz_ovrlp)
#            if block_above is not None:
#                cv2.imshow('block_above',block_above)
#            if block_left is not None:
#                cv2.imshow('block_left',block_left)
#            cv2.imshow('block_next',block_next)
#            cv2.imshow('block_next_vert_ovrlp',block_next_vert_ovrlp)
#            cv2.imshow('block_next_horz_ovrlp',block_next_horz_ovrlp)
            boundaries = MinimumErrorCuts(
                                    block_prev_vert_ovrlp
                                    ,block_prev_horz_ovrlp
                                    ,block_next_vert_ovrlp
                                    ,block_next_horz_ovrlp
                                    ,ovrlp_size
                                    )
            ovrlp_qlt = QuiltBlocks(
                                block_prev_vert_ovrlp
                                ,block_prev_horz_ovrlp
                                ,block_next
                                ,boundaries
                                ,cur_synth_row
                                ,cur_synth_col
                                ,o_synth_array
                                ,ovrlp_size
                                )
            cur_synth_col += ovrlp_qlt[1]
            img_zoomed=cv2.resize(o_synth_array, (256,256), interpolation=cv2.INTER_NEAREST )
            cv2.imshow('Synth Array After',img_zoomed)
            cv2.waitKey(1)
        cur_synth_row += ovrlp_qlt[0]
        print
    cv2.imwrite(output_fname,o_synth_array)
    print "wrote %s" % output_fname

    cv2.destroyAllWindows()
    cv2.imshow(input_fname,i_texture)
    cv2.imshow(output_fname,o_synth_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()
