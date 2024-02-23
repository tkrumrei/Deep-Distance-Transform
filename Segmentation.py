'''
Schritte:
- Seeds finden (lokale Minima und lokale Maxima Hintergrund)
- Seed based watershed Algorithm aus Deep Distance Transform Paper
Pseudocode:
Data: D                             // distance transform estimate
Smin                                // list of filtered local minima
Smax                                // list of background seeds
Result: list of masks
foreach s1 in Smin do
    foreach s2 in Smin\{s1} do
        , mask1, mask2 = watershed(-abs(D), Smax, s1, s2)
    if mask1 in candidates then
        inc mask count(mask1)
    else
        candidates.append(mask1)
    // repeat for mask 2
    end
end
sortCandidatesByCount(masks)        // highest count first
    foreach mask in candidates do
        if (!any(mask \ candidates == 0) then
    list of masks.append(mask)
    end
end
'''
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2



def find_local_minima(dist_transform, size=5):
    dist_transform = -dist_transform

    coordinates = peak_local_max(dist_transform, min_distance=size, threshold_abs=0.0)

    # only for visualization
    
    # display results
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(dist_transform, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(dist_transform, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('get local minima')

    fig.tight_layout()

    plt.show()
    
    print("find_local_minima fertig")
    return coordinates

def find_local_maxima(dist_transform, size=20):

    coordinates = peak_local_max(dist_transform, min_distance=size, threshold_abs=0.0)
    filtered_coordinates = []

    # exclude coordinates that are too far away from the cells
    for cor in coordinates:
        top_left_y = cor[0] - size
        top_left_x = cor[1] - size
        bottom_right_y = cor[0] + size
        bottom_right_x = cor[1] + size

        area = dist_transform[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

        equal = np.all(area == area[size,size])

        if not equal:
            filtered_coordinates.append(cor)
    
    filtered_coordinates = np.array(filtered_coordinates)

    # only for visualization
    '''
    print("peak_local_max berechnet")

    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(dist_transform, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(dist_transform, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    ax[2].imshow(dist_transform, cmap=plt.cm.gray)
    ax[2].autoscale(False)
    ax[2].plot(filtered_coordinates[:, 1], filtered_coordinates[:, 0], 'r.')
    ax[2].axis('off')
    ax[2].set_title('Peak local max filtered')

    fig.tight_layout()

    plt.show()
    '''
    print("find_local_maxima fertig")
    return filtered_coordinates

def segmentation(dist_transform, smax, s1, s2):
    markers = np.zeros_like(dist_transform, dtype=int)
    markers[s1[0],s1[1]] = 1  # s1
    markers[s2[0],s2[1]] = 2  # s2
    for i, seed in enumerate(smax, start=3):
        markers[seed[0],seed[1]] = i

    # watershed algorithm
    labels = watershed(-np.abs(dist_transform), markers)

    mask1 = (labels == 1)
    mask2 = (labels == 2)

    return(mask1, mask2)

def mask_to_area(mask, cell_area):
    area = 0.0
    min_area = cell_area / 2.5
    max_area = cell_area * 2.5

    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        area = cv2.contourArea(contours[0])
    
    
    return min_area <= area <= max_area

def mask_generation_from_seeds(dist_transform, cell_area):
    smin = find_local_minima(dist_transform, size=5)
    print(len(smin))
    smax = find_local_maxima(dist_transform, size=20)
    print(len(smax))

    candidates = []
    mask_count = {}
    counter = 0

    # make candidate masks and add them to candidates
    for s1 in smin:
        for s2 in smin:
            if not(np.array_equal(s2,s1)):
                mask1, mask2 = segmentation(dist_transform, smax, s1, s2)
                
                # have mask1 and mask2 unusual cells widths
                mask1_valid = mask_to_area(mask1, cell_area)
                mask2_valid = mask_to_area(mask2, cell_area)

                if mask1_valid:
                    for i, candidate in enumerate(candidates):
                        if np.array_equal(candidate, mask1):
                            mask_count[i] += 1
                            break
                    else:
                        candidates.append(mask1)
                        mask_count[len(candidates) - 1] = 1

                if mask2_valid:
                    for i, candidate in enumerate(candidates):
                        if np.array_equal(candidate, mask2):
                            mask_count[i] += 1
                            break
                    else:
                        candidates.append(mask2)
                        mask_count[len(candidates) - 1] = 1
        print("watershed Segmentierung für s1 ist fertig")
        counter += 1
        if counter == 3:
            break


    # sortCandidatesByCount(masks) // highest count first
    sorted_mask_count = sorted(mask_count.items(), key=lambda item: item[1], reverse=True)
    sorted_candidates = []

    for i, count in sorted_mask_count:
        sorted_candidates.append(candidates[i])
    
    list_of_masks = []

    # candidates are only addes if they contain pixels not covered by any other accepted mask
    for candidate in sorted_candidates:
        is_unique = True 
        for accepted_mask in list_of_masks:
            if np.any(candidate & accepted_mask):
                is_unique = False
                break
        if is_unique:
            list_of_masks.append(candidate)
    
    print("Liste von Einzelmasken ist fertig")
    return list_of_masks

def make_mask(dist_transform, cell_area):
    list_of_masks = mask_generation_from_seeds(dist_transform, cell_area)
    mask = np.zeros_like(dist_transform, dtype=int)
    # make mask
    for i, candidate in enumerate(list_of_masks):
        mask[candidate] = i + 1

    print("Maske ist fertig")
    # only for visualization
    '''
    # display results
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(dist_transform, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(mask, cmap=plt.cm.gray)
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    fig.tight_layout()

    plt.show()
    '''
    return mask

'''
dist_transform = np.load("C:/Users/Tobias/Desktop/test2/test2/distance_transform/11657-28091999_01_x1=1223_y1=3855_x2=2247_y2=4879_dt.npy")
dist_transform2 = np.load("D:/Datasets/Testis_Model/Cellpose/Train/distance_transform/0108_dt.npy")
dist_transform3 = np.load("D:/Datasets/Testis_Model/Cellpose/Train/distance_transform/0340_dt.npy")
'''