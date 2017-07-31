import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


AVG_MEAN = 0
AVG_MEDIAN = 1

x = []
y = []


def find_star(im, roi=None, grid_size=5):
    '''
    find a star in the image (im) within the region of interest (roi),
    using a matching grid size
    im: numpy array, image in grayscale
    roi: region of interest, a nested tuple with coord of top-left and bottom-right point,
        defaults to the whole image
    grid_size: an number indicating the searching grid size

    returns the center coord of the found star as a tuple
    '''
    assert (grid_size%2) == 1  # should be positive odd integer

    if roi == None:
        roi = [[0,0],[im.shape[1],im.shape[0]]]

    roi_width = roi[1][0] - roi[0][0]
    roi_height = roi[1][1] - roi[0][1]
    start_x = roi[0][0]
    start_y = roi[0][1]

    scores = []
    for j in range(roi_height - (grid_size-1)):
        for i in range(roi_width - (grid_size-1)):
            test_matrix = im[(j+start_y):(j+start_y+grid_size), (i+start_x):(i+start_x+grid_size)]
            scores.append(np.sum(test_matrix))

    max_index = scores.index(max(scores))
    max_x = start_x + (grid_size-1)//2 + (max_index % (roi_width - (grid_size-1)))
    max_y = start_y + (grid_size-1)//2 + (max_index // (roi_width - (grid_size-1)))
    #      base coord + offset to the inner + offset in the inner
    return (max_x, max_y)


def align_images(images, kps):
    """align images and crop the extra edges"""
    
    x = []
    y = []
    for i in kps:
        x.append(i[0])
        y.append(i[1])
    
    window_left_bound = min(x)
    window_right_bound = max(x)
    window_up_bound = min(y)
    window_down_bound = max(y)
    image_dimy, image_dimx = images[0].shape
    #print("x={}".format(x))  # TEST
    #print("y={}".format(y))  # TEST

    for i in range(len(images)):
        images[i] = images[i][
                    y[i] - window_up_bound : image_dimy - (window_down_bound - y[i]),
                    x[i] - window_left_bound : image_dimx - (window_right_bound - x[i])]

    return images


images=[]


def main(num_images):
    global x
    global y

    global images
    stddev = []

    t_start = time.time()

    x = [1311]
    y = [848]
    kps = []
    images = []

    for i in range(num_images):
        im = cv2.imread(str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE)  # read in grayscale, start with the second image
        images.append(im)
        kps.append(find_star(im, ((1260, 790), (1370, 910))))  # region of interest found by inspecting the first image manually

    images = align_images(images, kps)
    
    t_mid = time.time()
    print("Preprocessing took {:.3f} seconds.".format(t_mid-t_start))
    im = np.mean(images,0)
    cv2.imwrite('out-mean.jpg',im)
    t_end = time.time()
    print("Pixel-wise mean of {} images took {:.3f} seconds.".format(num_images, t_end-t_mid))

    #print("Average standard deviation between the pixel values: {}".format(np.mean(stddev)))


def auto_process(num_images, align_tolerance=60):  # tolerance in alignment, unit is number of pixels
    """automate the process"""
    images = []
    t_start = time.time()

# read images
    for i in range(num_images):
        images.append(cv2.imread(str(i+1)+'.jpg', cv2.IMREAD_GRAYSCALE))  # read in grayscale, start with the second image
# find the first star
    kps = [find_star(images[0])]
    roi = [[kps[0][0] - align_tolerance, kps[0][1] - align_tolerance],
           [kps[0][0] + align_tolerance, kps[0][1] + align_tolerance]]
# look for the same star in all other images
    for i in range(num_images-1):
        kps.append(find_star(images[i+1], roi))
    #print("len(kps)={}".format(len(kps)))  # TEST
# align
    images = align_images(images, kps)
    t_mid = time.time()
    print("Preprocessing took {:.3f} seconds.".format(t_mid-t_start))
# process
    imout = np.median(images, 0)
    print("Pixel-wise median over {} images took {:.3f} seconds.".format(num_images, time.time()-t_mid))
# output
    cv2.imwrite('out-median.jpg', imout)


if __name__ == '__main__':
    #main(15)
    auto_process(15)
