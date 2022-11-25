import os
from PIL import Image
from pdf2image import convert_from_path

def readFile(filePath):
    #read image file
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    if filePath.lower().endswith(('.png', '.jpg', '.jpeg')):
        files.append(os.path.join("/", filePath))
    elif filePath.endswith('pdf'):
        images = convert_from_path(filePath)
        images.save("images")