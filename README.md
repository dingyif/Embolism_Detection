# Embolism Detection Method Summary
 
## Current version: 11 or 12
 

### Notation:
Images from one experiment = the images in one folder
Usually there are five to ten experiments for one species (depending on whether they have images for both stem and leaf).


## I. Differencing:
A.     Take the difference between consecutive gray-scale images
B.     Note: flip sign for leafs. because embolism seems to take places where the vein becomes "darker” for leafs 
(If we don’t flip the sign, there’s barely any embolism. ImageJ is doing this way as well. Chris doesn’t know why either.)


## II.  Thresholding
A.     clip pixel values smaller than 3 to 0 (3 is suggested by Chris)


## III.Binarizing
A.     convert all positive pixel values to 1, the resulting image only has values (1 or 0),
 where 0 means there’s no change at that pixel between two consecutive gray-scale images.

## IV.  Foreground Background Segmentation (stem only)
A.     Motivation: To reduce false positive, because there are clearly parts that are background (i.e. not stem) due to the way images are captured under current setting

B.     Challenge:
					1.[shift] Stem is almost always shifting due to gel movement. 
      It’s just that sometimes it’s a negligible small shift, while sometimes it’s a big shift that would cause people to think there are a lot of embolism events just by looking at the binarized difference image because the binarized difference image is very dark.
     2.[bark] Separate stem from bark (ex: In5_Stem, in3 stem)
					3.[shrink] Stem shrinks as time goes by because of dehydration.
     4.[browner] Stem changes color as time goes by because of dehydration. (ex: green to brown)
     5.[color] Stem in the images might not be “green with a red background”. There are cases where the stem is light yellow and the background is dark brown.
	
C.     Have tried:

     1.If a pixel doesn’t change gray-scaled value too often across time, it’s classified as foreground
								a.Steps: Apply low pass filter (Gaussian filter) on mean image (take the average of all images from one experiment). Then use a fixed threshold (for all species) for thresholding.
								b.Output: The same mask (mask would be a matrix that determines whether this pixel position is foreground or not) for all images in an experiment (i.e. a folder)
								c.Assumption:
												i.There would be many random noises in the background, while less in the foreground. (If pixel value > threshold in LPF mean image, should be classified as noise)
												ii.Foreground barely moves and foreground object shape is not changing over time.
								d.Problem: challenge 1 [shift] and 3 [shrink] violate assumption ii
								
					2.[version<10] If a pixel is green enough and blue enough, it’s classified as foreground
a.     Steps: Apply a low pass filter (Gaussian filter) on one image’s green layer and use a fixed threshold (for all species) for thresholding. Only keep the connected component with the largest area. Also do the same on one image’s blue layer. Then take the intersection between these two results.
b.     Output: One mask for every image in an experiment. Drop the last mask to match the size of binarized difference image stack.
c.      Assumption:
i.       stem is greener than background
ii.      both stem and bark might be very green, but stem is whiter than bark. In an attempt to address challenge 2 [bark]
iii.    color doesn’t change across time (so that a fixed threshold is reasonable)
d.     Problem:
i.       challenge 4 [browner] violates assumption iii, making this method unstable because the thresholds for green and blue layers are fixed (last images in Alclat3_stem, Alclat5_stemDoneBad)
ii.      Only use the connected component of the largest area (Alclat3_stem) might not be robust to variations of noises
3.     [version >= 10.1] Shift the user-given stem mask using correlation.
a.     Steps: Use one user-predetermined stem mask (input/stem.jpg) as the 1st mask in. Next, detect shifting using correlation, then shift the 1st mask accordingly.
b.     Output: One mask for every image in an experiment. Drop the last mask to match the size of binarized difference image stack. (is this reasonable?)
c.      Assumption:
i.       Given the 1st mask for every experiment
ii.      The shape of stem doesn’t change across time
d.     Problem:
i.       assumption ii is violated by challenge 1[shrink], causing there to be many false positive near edge boundaries.
ii.      Results might be sensitive to input/stem.jpg?
iii.    Error propagation: have to choose the minimum threshold that would be qualified as shifting (shift_px_min) carefully.
e.     Note: [version 10] doesn’t have assumption ii.
It takes the intersection of the above results with the results obtained using thresholding green layer (without largest area requirement)
i.       Problem: thresholding green layer produces unstable results as it’s a fixed threshold with the existence of challenge 4 [browner]
 
 
## V.  Poor Quality (stem only)
A.     Apply Hough transformations on each median filtered image to detect circles (which could be bubbles/embolism/shifting/plastic cover). If the maximum of the connected component area is greater than a fixed threshold, then the image is classified as poor quality.
B.     Output: a vector of image index that are considered as poor quality (poor_qual_set_cc)
C.     Motivation:
1.     To reduce false positive coming from poor quality images (too many bubbles/ too much shifting)
2.     Thresholding maximum of connected component area instead of thresholding total circle area is motivated by the figures below. Because sometimes the algorithm mistakes plastic cover/embolism as bubbles.
D.     Problem:
1.     Bubbles sometimes might not look like circles (Ex: half of a circle), so those won’t be able to detect correctly.
2.     Currently can’t separate bubbles, embolism, shifting, plastic cover correctly (TODO: shifting can probably be detected using correlation in previous step: Foreground Background Segmentation)


## VI.  Detect embolism (1st stage)
A.     Steps:
For each image (not in poor_qual_set_cc for stem):
1.     connect the embolism parts in the median-filtered image (more false positive)
a.     opening then closing (larger closing kernel than that in step2)
b.     keep connected components with area > a fixed threshold (area_th) (for stem: area_th = 1)
c.  	expand embolism candidate by closing and dilating
2.     shrink the embolism parts in the median-filtered image (more false negative)
a.     opening then closing
b.     keep connected components with area > a fixed threshold (area_th2)
3.     pixels from step2 correspond to which connected component from step1
a.     For each connected component in step 2:
i.       (leaf only) If (the area of a connected component in step1)/(the area of intersection of connected component in step2 and binarized difference image) < a fixed threshold (ratio_th), then keep “the intersection of connected component in step 1 and binarized difference image” as an embolism event
a.     Motivation: To avoid the case that a small noise in step 2 leads to a really big embolism part in step 1, which usually happens in leafs.
4.     reduce false positive
a.     (stem only) if the total predicted embolism area is too BIG (probably due to shifting) or too SMALL (negligible), predict no embolism
b.     Then discard connected components with low density and small area


## VII. Reduce False Positive (2nd stage)
A.     Stem:
1.     Don't count as embolism if it keeps appearing (probably is plastic cover or bubbles)
a.     Steps: Look at first 200 images, if the frequency of predicted embolism at the same pixel location is too high, treat it as no embolism. Then repeat the process for the next 200 images (image index: 201~400) (rolling window).
2.     Basic shape analysis:
a.     Separate into strong embolism candidate set and weak embolism candidate set by setting threshold on length and area of a connected component
i.       If the connected component is too short/ too small/ too big/ too wide/ wide but not long, treat it as not an embolism event. [version >= 11] The output image stack is saved as final_stack_strong_emb_cand
ii.      However, if the connected only too short or only too small, store the image index of it into the candidate set for weak embolism (weak_emb_cand_set)
b.     [version >= 9.9] Separate weak embolism from noise using density:
i.       For images in weak embolism candidate set, we still consider it as embolism if the density of the connected component is greater than a certain threshold
c.      [version 11] Output: weak embolism + strong embolism candidate (i.e. strong embolism + noise)
d.     [version 12] Separate strong embolism from noise using CNN (cat-dog structure)
i.       Output: weak embolism + (strong embolism + less noise)
ii.      Problem:
a.     Though false positive decrease, false negative increases. L
b.     Have tried 2 different dual-input (also feed in binarized difference image stack as input) CNN, but the accuracy doesn’t seem high enough (i.e. would still sacrifice false negative). Hence, haven’t combined them with weak embolism results
B.     Leaf:
1.     If any of the following is satisfied, discard the connected component.
a.     the connected component is too small/too big
b.     90% quantile of {intersection of the c.c. with uniform filtered image} is too small
2.     Motivation:
a.     Separating the embolism parts that are connected to the noises at the boundaries of images (Reduce false positive)
b.     Use 90% quantile because density of noise c.c. is pretty similar to that of the embolism
 

