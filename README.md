# Simple_Photo_Stacking
A simple python program that stacks multiple photos to reduce random noise.

Currently it looks for the brightest 5*5 area in the image as the reference feature to align the images, and crop the edges to align them.
Therefore the script can't do nonlinear transformations (yet). It also has no command line arguments support (yet).

Demo photo taken by Nexus 5:

Original(grayscale):
![original grayscale](/images/original-1.jpg?raw=true "Original grayscale")

Mean stacked:
![mean stacked](/images/out-mean.jpg?raw=true "Mean stacked")

Median stacked:
![median stacked](/images/out-median.jpg?raw=true "Median stacked")
