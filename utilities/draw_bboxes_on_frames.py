#!/usr/bin/env python

import os
import cv2
import argparse

# Function to read the colors file
def label_colors_from_file(file_path):
    labels = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Format is: R G B LABELNAME. See KITTI annotations format
            # (https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/)
            r, g, b, label = line.split()
            if label != 'background':
                labels[label] = (int(r), int(g), int(b))
    return labels

# Function to read the bounding boxes from a text file
def read_bounding_boxes(bbox_file):
    bounding_boxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_name = ' '.join(parts[0:-5])
            instance_number = int(parts[-5])
            x, y, w, h = map(int, parts[-4:])
            bounding_boxes.append((class_name, instance_number, x, y, w, h))
    return bounding_boxes

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image, bounding_boxes, colors):
    for bbox in bounding_boxes:
        class_name, _, x, y, w, h = bbox
        # Default to white if class color is not found
        color = colors.get(class_name, (255, 255, 255))
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        # text = f'{class_name} {instance_number}'
        text = f'{class_name}'
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Main function to process all images
def process_images(images_dir, bboxes_dir, colors_file, output_dir, resize_factor):
    # Read the colors from the file
    colors = label_colors_from_file(colors_file)

    # Process each image
    image_names = os.listdir(images_dir)
    l = len(image_names)
    for i, image_name in enumerate(image_names):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):
            # Read the image
            image_path = os.path.join(images_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor)

            # Corresponding bounding boxes file
            txt_name = 'frame_' + os.path.splitext(image_name)[0] + '.txt'
            bbox_path = os.path.join(bboxes_dir, txt_name)

            if os.path.exists(bbox_path):
                # Read the bounding boxes and draw them on the image
                bounding_boxes = read_bounding_boxes(bbox_path)
                draw_bounding_boxes(image, bounding_boxes, colors)
                # Add frame number
                cv2.putText(image, f'Frame: {os.path.splitext(image_name)[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

                # Save the image with bounding boxes in the output directory
                output_path = os.path.join(output_dir, image_name)
                cv2.imwrite(output_path, image)
            else:
                print(f'No bounding boxes file found for: {image_name}')
        
        printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Draw bounding boxes on images using txt files.')
    parser.add_argument('--input_folder', required=True, help='Directory with the input images')
    parser.add_argument('--bboxes_folder', required=True, help='Directory with the bounding boxes files')
    parser.add_argument('--label_colors', required=True, help='Colors file (txt) in R G B class format')
    parser.add_argument('--output_folder', required=True, help='Directory where the processed images will be saved')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Resize factor for the frames')
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

    # Create the output directory if it does not exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # Process all images
    process_images(args.input_folder, args.bboxes_folder, args.label_colors, args.output_folder, args.resize_factor)
