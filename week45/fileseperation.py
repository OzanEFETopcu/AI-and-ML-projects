from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed, random

# Name and create the subfolders
dataset_home = 'images/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
    # Create directories with appropriate names (the name of the directory is the label of the images in it) -> the name of the file has no longer any impact when the files have been divided into the directories
    labeldirs = ['dogs/', 'cats/','horses/']
    for labeldir in labeldirs:
        newdir = dataset_home + subdir + labeldir
        makedirs(name=newdir, exist_ok=True)

# Create also other needed directories for this tutorial
makedirs(name='models/', exist_ok=True)
makedirs(name='plots/', exist_ok=True)

# Intialise the random number generator with a seed
seed(1)

# Define how many of the pictures are test images
test_share = 0.2  # 20 %

# Copy the original images based on the beginning part of the file name (cat or dog)
# Decide the copying to train and test directories according to the random number
src_directory = 'original_data/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < test_share:
        dst_dir = 'test/'
    if file.startswith('cat'):
        destination = dataset_home + dst_dir + 'cats/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('dog'):
        destination = dataset_home + dst_dir + 'dogs/' + file
        copyfile(src, destination)
        print("Created: " + destination)
    elif file.startswith('horse'):
        destination = dataset_home + dst_dir + 'horses/' + file
        copyfile(src, destination)
        print("Created: " + destination)
        
