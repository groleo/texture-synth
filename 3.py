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


def VerticalOvrlpError(block_left, block_next):

    diff_r =  block_left[:, :, 0] - block_next[:, :, 0]
    diff_g =  block_left[:, :, 1] - block_next[:, :, 1]
    diff_b =  block_left[:, :, 2] - block_next[:, :, 2]

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.int)
    #dist = ((0.2125 * diff_r**2) + (0.7154 * diff_g**2) + (0.0721 * diff_b**2))
    #dist = ((diff_r**2 + diff_g**2 + diff_b**2)/3.0).astype(np.int)
    return dist


def HorizntlOvrlpError(block_above, block_next):
    diff_r =  block_above[:, :, 0] - block_next[:, :, 0]
    diff_g =  block_above[:, :, 1] - block_next[:, :, 1]
    diff_b =  block_above[:, :, 2] - block_next[:, :, 2]

    dist = ((diff_r**2 + diff_g**2 + diff_b**2)**0.5).astype(np.int)
    return dist


def GetVertOvrlpPrev(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:,-ovrlp_size:]
    return ovrlp

def GetVertOvrlpNext(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:,:ovrlp_size,:]
    return ovrlp

def GetHorzOvrlpPrev(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[-ovrlp_size:]
    return ovrlp

def GetHorzOvrlpNext(block,ovrlp_size):
    if block is None:
        return None
    block_row = block.shape[0]
    ovrlp = np.zeros((block_row, ovrlp_size))
    ovrlp = block[:ovrlp_size]
    return ovrlp


def BlockSelect(block_left, block_above, texture, synth, block_side_length, ovrlp_size, err_threshold):
    PixelList = []
    texture_row = texture.shape[0]
    texture_col = texture.shape[1]
    err_v    = np.full((block_side_length,ovrlp_size), err_threshold)
    err_h    = np.full((ovrlp_size,block_side_length), err_threshold)

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
        #print rv.shape[1]
        return [rv]

    elif block_above is None:
        for row in range(0,texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                ovrlp_left = GetVertOvrlpPrev(block_left,ovrlp_size)
                ovrlp_next = GetVertOvrlpNext(block_next,ovrlp_size)

                error_Vertical = VerticalOvrlpError(ovrlp_left, ovrlp_next)

                if np.less(error_Vertical,err_v).all():
                    PixelList.append(block_next)
                elif np.less(error_Vertical,err_v/2).all():
                    return [block_next]

    elif block_left is None:
        for row in range(0, texture_row-block_side_length):
            for col in range(0,texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                ovrlp_above = GetHorzOvrlpPrev(block_above,ovrlp_size)
                ovrlp_next = GetHorzOvrlpNext(block_next,ovrlp_size)

                error_Horizntl = HorizntlOvrlpError(ovrlp_above, ovrlp_next)

                if np.less(error_Horizntl,err_h/2).all():
                    return [block_next]
                if np.less(error_Horizntl,err_h).all():
                    PixelList.append(block_next)
    else:
        for row in range(0, texture_row-block_side_length):
            for col in range(0, texture_col-block_side_length):
                block_next = GetBlock(texture, row, col, block_side_length, block_side_length)

                ovrlp_above = GetHorzOvrlpPrev(block_above,ovrlp_size)
                ovrlp_left = GetVertOvrlpPrev(block_left,ovrlp_size)

                ovrlp_vnext = GetVertOvrlpNext(block_next,ovrlp_size)
                ovrlp_hnext = GetHorzOvrlpNext(block_next,ovrlp_size)

                error_Vertical = VerticalOvrlpError(ovrlp_left, ovrlp_vnext)
                error_Horizntl = HorizntlOvrlpError(ovrlp_above, ovrlp_hnext)

                if (np.less(error_Vertical,err_v/2).all()
                and np.less(error_Horizntl,err_h/2).all()
                   ):
                    return [block_next]

                if (np.less(error_Vertical,err_v).all()
                and np.less(error_Horizntl,err_h).all()
                   ):
                    PixelList.append(block_next)

    return PixelList

#def diff(block_left, block_next, ovrlp_size):
#    ovrlp_0 = block_left[-ovrlp_size:]
#    ovrlp_1 = block_next[:ovrlp_size]
#    err =  ovrlp_0 - ovrlp_1
#
#    return err


#def PickBlock(off_row, off_col, texture, block_side_length):
#    if off_row==0 and off_col==0:
#        return PickInitialBlock(off_row, off_col, texture, block_side_length)
#
#    texture_row = texture.shape[0]
#    texture_col = texture.shape[1]
#
#    randomPatch_x = randint(0, texture_row - block_side_length)
#    randomPatch_y = randint(0, texture_col - block_side_length)
#    o_synth_array = np.zeros((block_side_length,block_side_length,3), np.uint8)
#    for x in range(block_side_length):
#        for y in range(block_side_length):
#            o_synth_array[x, y] = texture[randomPatch_x + x, randomPatch_y + y]
#
#    return o_synth_array

def GetBlock(texture, off_row, off_col, block_row_height, block_col_width):
    out = np.zeros((block_row_height,block_col_width,3), np.uint8)
    out = texture[off_row:off_row+block_row_height,
                  off_col:off_col+block_col_width
                 ]
    return out

def PutBlock(out, off_row, off_col, block):
    if block is None:
        print "empty block"
        raise

    block_row = block.shape[0]
    block_col = block.shape[1]

    out[off_row:off_row+block_row,
        off_col:off_col+block_col
       ] = block

def GetHorzOvrlp(texture, off_row, off_col, block_side_length, ovrlp_size):
    off_row -= ovrlp_size
    print "row:%03s" % off_row
    if off_row <0:
        return None
    return GetBlock(texture, off_row, off_col, ovrlp_size, block_side_length)


def GetVertOvrlp(texture, off_row, off_col, block_side_length, ovrlp_size):
    off_col -= ovrlp_size
    print "col:%03s" % off_col
    if off_col < 0:
        #print "no blocks found at the Left"
        return None
    return GetBlock(texture, off_row, off_col, block_side_length, ovrlp_size)


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

def display(name, block):
    cv2.imshow(name,block)
def displayw(name, block):
    cv2.imshow(name,block)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 2.1 Minimum Error Boundary Cut
###########################
# E
# . . . . @ . . . . . . . .
# . . . @ . . . . . . . . .
# . . . . @ . . . . . . . .
# . . . @ . . . . . . . . .
# . . . @ . . . . . . . . .
# . . @ . . . . . . . . . .
# . . . @ . . . . . . . . .
# . . . . @ . . . . . . . .
# . . . . @ . . . . . . . .
# . . . . . @ . . . . . . .
# . . . . . @ . . . . . . .
# . . . . @ . . . . . . . .
# . . . . @ . . . . . . . .
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
    cost_min = Cost.min()
    cost_max = Cost.max()
    return (Cost - cost_min)/(cost_max - cost_min)


# 2.1 Minimum Error Boundary Cut
###########################
# E
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . . . . . .
# . . . . . . . . @ . . . .
# . . . . . . @ @ . @ @ . .
# . @ @ . . @ . . . . . @ @
# @ . . @ @ . . . . . . . .
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
    cost_min = Cost.min()
    cost_max = Cost.max()
    return (Cost - cost_min)/(cost_max - cost_min)


def FindMinVertCostPath(Cost):
    rows = Cost.shape[0]
    min_error_cut = np.zeros(rows,np.uint8)
    argmin_col = np.argmin(Cost[-1:],axis=1)[0]
    min_error_cut[rows-1] = argmin_col

    #print Cost
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



def FindMinHorzCostPath(Cost):
    cols = Cost.shape[1]
    min_error_cut = np.zeros(cols,np.uint8)
#
    argmin_row = np.argmin(Cost[:,cols-1])
    min_error_cut[cols-1] = argmin_row

    #print Cost
    cv2.imshow("horz_cost", Cost)
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
def MinimumErrorCut(block_left, block_above, block_vert_next, block_horz_next, ovrlp_size):
    if block_above is None and block_left is None:
        return (None,None)

    elif block_above is None:
        VertCost = MinVertCumulativeError(
                                          block_left
                                         ,block_vert_next
                                         ,ovrlp_size
                                         )
        VertBoundary = FindMinVertCostPath(VertCost)
        return (VertBoundary,None)

    elif block_left is None:
        HorzCost = MinHorzCumulativeError(
                                          block_above
                                         ,block_horz_next
                                         ,ovrlp_size
                                         )
        HorzBoundary = FindMinHorzCostPath(HorzCost)
        return (None,HorzBoundary)

    else:
        VertCost = MinVertCumulativeError(
                                          block_left
                                         ,block_vert_next
                                         ,ovrlp_size
                                         )
        HorzCost = MinHorzCumulativeError(
                                          block_above
                                         ,block_horz_next
                                         ,ovrlp_size
                                         )
        VertBoundary = FindMinVertCostPath(VertCost)
        HorzBoundary = FindMinHorzCostPath(HorzCost)
        return (VertBoundary,HorzBoundary)

###################################################
###########################
# @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ @ @ @ @ @ @ @ @ @ @ @
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
# @ @ . . . . . . . . . . .
###########################

def QuiltBlock(overlap_left, overlap_above, block_next, boundaries, o_cur_row, o_cur_col, o_synth_array, ovrlp_size):
    if overlap_left is None and overlap_above is None:
        PutBlock(o_synth_array, o_cur_row, o_cur_col, block_next)
        return block_next.shape

    block_row = block_next.shape[0]
    block_col = block_next.shape[1]

    boundary_vert  = boundaries[0]
    boundary_horz  = boundaries[1]

    quilted_block = np.copy(block_next)

    if boundary_vert is not None:
        for row in range(block_row):
            for col in range(block_col):
                if col < boundary_vert[row]:
                    quilted_block[row,col] = overlap_left[row,col]
        o_cur_col -= ovrlp_size
        block_col -= ovrlp_size

    if boundary_horz is not None:
        for col in range(block_col):
            #print "horiz until:%s" % boundary_horz[col]
            for row in range(block_row):
                if row < boundary_horz[col]:
                    quilted_block[row,col] = overlap_above[row,col]
        o_cur_row -= ovrlp_size
        block_row -= ovrlp_size

    cv2.imshow("quilted_block", quilted_block)
    PutBlock(o_synth_array, o_cur_row, o_cur_col, quilted_block)
    return (block_row,block_col)


def align_up(in_val, in_base):
    return int(in_base*math.ceil((1.0*in_val)/in_base))


#def QuiltBlockLeft(overlap_left, block_next, boundaries, o_cur_row, o_cur_col, o_synth_array, ovrlp_size):
#    if overlap_left is None:
#        print "QuiltBlockLeft:leftmost block"
#        PutBlock(o_synth_array, o_cur_row, o_cur_col, block_next)
#        return block_next.shape
#
#    block_row = block_next.shape[0]
#    block_col = block_next.shape[1]
#    boundary  = boundaries[0]
#    o_cur_col -= ovrlp_size
#
#    quilted_block = np.copy(block_next)
#    for row in range(block_row):
#        print "left until:%s" % boundary[row]
#        for col in range(block_col):
#            if col < boundary[row]:
#                quilted_block[row,col] = overlap_left[row,col]
#
#    cv2.imshow("quilted_block", quilted_block)
#    PutBlock(o_synth_array, o_cur_row, o_cur_col, quilted_block)
#    return (block_next.shape[0],block_next.shape[1]-ovrlp_size)

###################################################
def main():
    input_fname       = str(sys.argv[1])
    block_side_length = int(sys.argv[2])
    ovrlp_size      = int(sys.argv[3])
    err_threshold     = float(sys.argv[4])
    print "ovrlp_size:%s" % ovrlp_size

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
    while o_cur_row < o_synth_row:
        o_cur_col = 0
        while o_cur_col < o_synth_col:
            if o_cur_row > 1 and o_cur_col >16:
                sys.exit(0)
            vert_ovrlp = GetVertOvrlp(
                                       o_synth_array
                                      ,o_cur_row
                                      ,o_cur_col
                                      ,block_side_length
                                      ,ovrlp_size
                                      )
            if vert_ovrlp is not None:
                cv2.imshow('vert_ovrlp',vert_ovrlp)

            horz_ovrlp = GetHorzOvrlp(
                                        o_synth_array
                                       ,o_cur_row
                                       ,o_cur_col
                                       ,block_side_length
                                       ,ovrlp_size
                                       )
            if horz_ovrlp is not None:
                cv2.imshow('horz_ovrlp', horz_ovrlp)
                cv2.waitKey(0)

            best_blocks = BlockSelect(
                             vert_ovrlp
                            ,horz_ovrlp
                            ,i_texture
                            ,o_synth_array
                            ,block_side_length
                            ,ovrlp_size
                            ,err_threshold
                            )
            if not best_blocks:
                print "ERROR:no best blocks"
                break

            block_next = random.choice(best_blocks)
            cv2.imshow('block_next',block_next)

            ovrlp_vert_next = GetVertOvrlpNext(
                                            block_next
                                           ,ovrlp_size
                                           )
            cv2.imshow('ovrlp_vert_next',ovrlp_vert_next)

            ovrlp_horz_next = GetHorzOvrlpNext(
                                            block_next
                                           ,ovrlp_size
                                           )
            cv2.imshow('ovrlp_horz_next',ovrlp_horz_next)
            boundaries = MinimumErrorCut(
                                       vert_ovrlp
                                      ,horz_ovrlp
                                      ,ovrlp_vert_next
                                      ,ovrlp_horz_next
                                      ,ovrlp_size
                                      )
            ovrlp_qlt = QuiltBlock(
                                   vert_ovrlp
                                  ,horz_ovrlp
                                  ,block_next
                                  ,boundaries
                                  ,o_cur_row
                                  ,o_cur_col
                                  ,o_synth_array
                                  ,ovrlp_size
                                  )
            o_cur_col += ovrlp_qlt[1]
            img_zoomed=cv2.resize(o_synth_array, (256,256), interpolation=cv2.INTER_NEAREST )
            cv2.imshow('Generated Image',img_zoomed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        o_cur_row += ovrlp_qlt[0]
        print
    cv2.imwrite('out.png',o_synth_array)

main()
