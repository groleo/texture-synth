#! /usr/bin/python

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from time import time

filename = "img2.gif"
output_file = "out2.png"
new_size_x = 56
new_size_y = 56
kernel_size = 21

# GrowImage and Leung 1999 implementation
"""
function GrowImage(SampleImage,Image,WindowSize)
  while Image not filled do
    progress = 0
    PixelList = GetUnfilledNeighbors(Image)
    foreach Pixel in PixelList do
      Template = GetNeighborhoodWindow(Pixel)
      BestMatches = FindMatches(Template, SampleImage)
      BestMatch = RandomPick(BestMatches)
      if (BestMatch.error < MaxErrThreshold) then
        Pixel.value = BestMatch.value
        progress = 1
      end
    end
    if progress == 0 
      then MaxErrThreshold = MaxErrThreshold * 1.1
  end
  return Image
end

Function GetUnfilledNeighbors() returns a list of all unfilled pixels that have filled pixels as their neighbors (the image is subtracted from its morphological dilation). The list is randomly permuted and then sorted by decreasing number of filled neighbor pixels. GetNeigborhoodWindow() returns a window of size WindowSize around a given pixel. RandomPick() picks an element randomly from the list. FindMatches() is as follows:

function FindMatches(Template,SampleImage)
  ValidMask = 1s where Template is filled, 0s otherwise
  GaussMask = Gaussian2D(WindowSize,Sigma)
  TotWeight = sum i,j GaussiMask(i,j)*ValidMask(i,j)
  for i,j do
    for ii,jj do
      dist = (Template(ii,jj)-SampleImage(i-ii,j-jj))^2
      SSD(i,j) = SSD(i,j) + dist*ValidMask(ii,jj)*GaussMask(ii,jj)
    end
    SSD(i,j) = SSD(i,j) / TotWeight
  end
  PixelList = all pixels (i,j) where SSD(i,j) <= min(SSD)*(1+ErrThreshold)
  return PixelList
end
"""

def FindMatches(x, y, img_data, new_img_data, mask, kernel_size):

    x0 = max(0, x - kernel_size)
    y0 = max(0, y - kernel_size)
    x1 = min(new_img_data.shape[0] - 1, x + kernel_size)
    y1 = min(new_img_data.shape[1] - 1, y + kernel_size)

    neigh_window = new_img_data[x0 : x1, y0 : y1]

    mask_window = mask[x0 : x1, y0 : y1]
    len_mask = float(len(mask_window==True))

    xs, ys = neigh_window.shape
    img_xsize, img_ysize = img_data.shape

    cx = int(np.floor(xs/2))
    cy = int(np.floor(ys/2))

    candidates = []
    dists = []

    for i in range(xs, img_xsize - xs):
        for j in range(ys, img_ysize - ys):
            if(randint(0,2) != 0): continue
            sub_window = img_data[i : i+xs, j : j+ys]

            # distance
            s = (sub_window - neigh_window)

            summ = s*s*mask_window

            d = np.sum(summ) / len_mask

            candidates.append(sub_window[cx, cy])
            dists.append(d)

    mask = dists - np.min(dists) < 0.2

    candidates = np.extract(mask, candidates)

    # pick random among candidates
    if len(candidates) < 1:
        return 0.0
    else:
        if len(candidates) != 1:
            r = randint(0, len(candidates) - 1)
        else:
            r = 0

    return candidates[r]



def GrowImage(img_data
        , new_size_x
        , new_size_y
        , kernel_size
        , t):

    patch_size_x, patch_size_y = img.size
    size_seed_x = size_seed_y = 3

    seed_x = randint(0, size_seed_x)
    seed_y = randint(0, size_seed_y)

    # take 3x3 start image (seed) in the original image
    seed_data = img_data[seed_x : seed_x + size_seed_x, seed_y : seed_y + size_seed_y]

    new_image_data = np.zeros((new_size_x, new_size_y))
    mask = np.ones((new_size_x, new_size_y)) == False

    mask[0:size_seed_x, 0:size_seed_y, 0:size_seed_z] = True

    new_image_data[0:size_seed_x, 0:size_seed_y] = seed_data


    it = 0
    for i in range(size_seed_x, new_size_x ):
        print "Process ", i, " / ", new_size_x, ". Time: ", time() - t, " seconds"

        last_y = size_seed_x + it
        # xxxxxxx
        for j in range(0, last_y + 1):

            v = FindMatches(i, j, img_data, new_image_data, mask, kernel_size)

            new_image_data[i, j] = v
            mask[i, j] = True


        # x
        # x
        # x
        for x in range(0, size_seed_y + it + 1):

            v = FindMatches(x, last_y, img_data, new_image_data, mask, kernel_size)

            new_image_data[x, last_y] = v
            mask[x, last_y] = True

        it += 1


        if(it % 10 == 0) :
            img_new = Image.fromarray(new_image_data)
            img_new.convert("RGB").save(output_file)


    return img_new

# main program

img = Image.open(filename)
img_data = img.convert("RGB")
img_data = np.array(img_data)

t = time()
img_new = GrowImage(img_data, new_size_x, new_size_y, kernel_size/2, t)
print "Total Time: ", time() - t, " seconds"

plt.imshow(img_new) #, cmap = "RGB")
plt.show()

img_new.convert("RGB").save(output_file)
