#!/usr/bin/env python

import os
import cv2
import argparse

# Main function to process all images
def resize_images(images_dir, output_dir, resize_factor):
    # Process each image
    image_names = os.listdir(images_dir)
    l = len(image_names)
    for i, image_name in enumerate(image_names):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

            # Save the image
            output_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_path, image)        
        printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Draw bounding boxes on images using txt files.')
    parser.add_argument('--input_folder', required=True, help='Directory with the input images')
    parser.add_argument('--output_folder', required=True, help='Directory where the processed images will be saved')
    parser.add_argument('--resize_factor', type=float, required=True, help='Resize factor for the frames')
    return parser.parse_args()

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

# Main
if __name__ == '__main__':
    args = parse_arguments()

    # If input and output directories are the same, exit
    if args.input_folder == args.output_folder:
        print('Input and output directories cannot be the same.')
        exit()

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Process all images
    resize_images(args.input_folder, args.output_folder, args.resize_factor)
