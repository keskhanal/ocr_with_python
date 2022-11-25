import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='Input')
parser.add_argument('--gpu_list', type=str, default='0')
parser.add_argument('--output_dir', type=str, default='Output')
FLAGS = parser.parse_args()

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.input_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    print(files)
                    break
    print('Find {} images'.format(len(files)))
    return files

get_images()