#!/usr/bin/env python

import cv2
import numpy as np
import os
import re

### 1 ###

def replace_color_in_masks(folder, old_color, new_color, range_val):
    # Iterate through all the images in the specified folder
    image_names = os.listdir(folder)
    l = len(image_names)
    for i, filename in enumerate(image_names):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            file_path = os.path.join(folder, filename)
            image = cv2.imread(file_path)

            # Convert the old color and new color into numpy arrays
            oldc = np.array(old_color, dtype=np.uint8)
            newc = np.array(new_color, dtype=np.uint8)

            # Convert BGR to RGB
            oldc = oldc[::-1]
            newc = newc[::-1]

            # Create a mask where the old color exists (module 255)
            oldc_min = np.array((max(0, oldc[0]-range_val), max(0, oldc[1]-range_val), max(0, oldc[2]-range_val)), dtype=np.uint8)
            oldc_max = np.array((min(255, oldc[0]+range_val), min(255, oldc[1]+range_val), min(255, oldc[2]+range_val)), dtype=np.uint8)
            mask = cv2.inRange(image, oldc_min, oldc_max)

            # Replace the old color with the new color
            image[mask != 0] = newc

            # Save the updated image back to the same path
            cv2.imwrite(file_path, image)

            printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)

### 2 ###
def remove_numbers_from_class_names(folder):
    # Regular expression to remove digits at the end of the first word
    pattern = re.compile(r'([a-zA-Z]+)\d*')

    # Process each file in the folder
    file_names = os.listdir(folder)
    l = len(file_names)
    for i, file_name in enumerate(file_names):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            
            # Read the file content
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Modify the content of each line
            new_lines = []
            for line in lines:
                parts = line.split()
                if parts:
                    # Replace the first word by removing numbers
                    parts[0] = re.sub(pattern, r'\1', parts[0])
                new_lines.append(' '.join(parts) + '\n')

            # Save the modified content back to the same file
            with open(file_path, 'w') as file:
                file.writelines(new_lines)
        
        printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)

### 3 ###

def label_colors_from_file(file_path):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Format is: R G B LABELNAME. See KITTI annotations format
            # (https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/)
            r, g, b, label = line.split()
            # STORE IN BGR FORMAT
            labels[label] = (int(b), int(g), int(r))    # includes background label
    return labels

def check_different_colors(input_folder, label_colors_path):
    label_colors = label_colors_from_file(label_colors_path)
    # Iterate through all the images in the specified folder
    file_names = os.listdir(input_folder)
    l = len(file_names)
    for i, filename in enumerate(file_names):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            masks = []
            for _, color in label_colors.items():
                # Create a mask where the color exists
                mask = cv2.inRange(image, np.array(color, dtype=np.uint8), np.array(color, dtype=np.uint8))
                masks.append(mask)
            
            # Combine all masks with an OR function
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Check if there are any pixels in the image that are not in the label colors
            if np.any(combined_mask == 0):
                print(f'Found a pixel with color not in the label colors file in image {filename}.')
                pixel = np.where(combined_mask == 0)
                print(f'Pixel color: {image[pixel[0][0], pixel[1][0]]}')
                return False

        printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)

    return True

### Console progress bar ###

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = '\r'):
    '''
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. '\r', '\r\n') (Str)
    '''
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

### MAIN FUNCTION ###

if __name__ == '__main__':
    while True:
        print('[1] Replace color in masks')
        print('[2] Remove numbers from class names')
        print('[3] Check different colors in masks')
        print('[4] Get unique colors in frame')
        print('[0] Exit')
        
        # Get user's choice
        choice = input('Enter your choice: ')
        if choice == '1':
            # Ask for the path to the folder with the masks
            folder_path = input('Enter the path to the folder containing the mask images: ')
            # Ask for the old and new colors
            old_color_str = input('Enter the old color (R,G,B) to replace: ')
            new_color_str = input('Enter the new color (R,G,B) to replace with: ')
            # Convert the input strings into tuples of integers for the colors
            old_color_str = old_color_str.replace(' ', '').replace('(', '').replace(')', '')
            new_color_str = new_color_str.replace(' ', '').replace('(', '').replace(')', '')
            old_color = tuple(map(int, old_color_str.split(',')))
            new_color = tuple(map(int, new_color_str.split(',')))
            # Ask for the value of range to replace
            print('Enter the range value to replace. Colors within this range will be replaced.')
            range_val = input('For example, 0 means exact match, 2 means +/- 2 in each channel: ')

            # Call the function to replace the color in all masks
            replace_color_in_masks(folder_path, old_color, new_color, int(range_val))
        
        elif choice == '2':
            folder_path = input('Enter the path to the folder containing the text files: ')
            remove_numbers_from_class_names(folder_path)
        elif choice == '3':
            # Ask for the path to the input folders
            folder_path = input('Enter the path to the folder containing the mask images: ')
            label_colors_path = input('Enter the path to the label colors file: ')
            if check_different_colors(folder_path, label_colors_path):
                print('All colors in the masks are the same as the label colors file.')
        elif choice == '4':
            # Ask for the input image
            image_path = input('Enter the path to the image mask: ')
            image = cv2.imread(image_path)
            unique_colors = np.unique(image.reshape(-1, image.shape[2]), axis=0)
            print('Unique colors in the image:')
            for color in unique_colors:
                print('(' + str(color[2]) + ', ' + str(color[1]) + ', ' + str(color[0]) + ')')
        elif choice == '0':
            print('Exiting the program.')
            break
        
        else:
            print('Invalid choice. Please select again.')
