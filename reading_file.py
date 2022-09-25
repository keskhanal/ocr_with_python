from PIL import Image

im_file = "data/test.jpg"

im = Image.open(im_file)
im.save("temp/test.jpg")