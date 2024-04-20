#Skimage library for Python

# Image= skimage.io.imread(‘image.jpg’) – Load an image as a numeric pixel array. 
# Image= matplotlib.image.imread(‘image.jpg’) – Alternative method for reading an image.
# Image_gray= skimage.color.rgb2gray(image) – Convert  a color image to grayscale.
# Skimage.transform.rescale(image_gray, 0.25) – Scale an image down to 25% of its original size, keeping the aspect ratio.
# Skimage.util.crop(image_gray, ( (0,0), (0,160)) ) – Crop image based on image’s top, bottom, left and right pixels
# Skimage.transform.resize(image_gray,  (320, 320)) – Reshape an image to fit into the provided dimension (expand or contract) 
# Skimage.transform.rotate(image_gray,  30) – Rotate the image counterclockwise by 30 degrees. Use negative values for clockwise.
# Tform= skimage.transform.SimilarityTransform(translation= (0,50)) – Create a transformation object (tform) that will translate (offset) the image 50 pixels from the bottom
# Skimage.transform.warp(image_gray, tform) – Perform transformation using the tform object
# Skimage.util.random_noise(image_gray) – Add random noise to the image
# Skimage.filters.gaussian(image_gray) – Add a blur effect to the image
