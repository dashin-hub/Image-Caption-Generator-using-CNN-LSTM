import numpy as np
from PIL import Image

# Create a simple test image (a colored rectangle)
img = np.zeros((224, 224, 3), dtype=np.uint8)
img[50:150, 50:150] = [255, 0, 0]  # Red rectangle
img[100:200, 100:200] = [0, 255, 0]  # Green rectangle

# Save the image
Image.fromarray(img).save('static/image/test.jpg')