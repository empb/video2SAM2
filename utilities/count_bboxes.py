#!/usr/bin/env python

import os
import argparse
from collections import defaultdict
import re

# Function to read bounding boxes from a text file
def read_bounding_boxes(bbox_file):
    bounding_boxes = []
    with open(bbox_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # In case the class name has more than one word
            class_name = parts[0]
            bounding_boxes.append(class_name)
    return bounding_boxes

# Function to count bounding boxes per class
def count_bboxes(bboxes_dir, check_duplicates):
    class_counts = defaultdict(int)  # Dictionary to store the count for each class

    # Iterate through each file in the bounding boxes directory
    nframes = 0
    bboxes_names = os.listdir(bboxes_dir)
    bboxes_names.sort()
    for bbox_file in bboxes_names:
        if bbox_file.endswith('.txt'):
            bbox_path = os.path.join(bboxes_dir, bbox_file)
            bounding_boxes = read_bounding_boxes(bbox_path)
            nframes += 1

            # Increment the count for each class
            for class_name in bounding_boxes:
                class_counts[class_name] += 1

            if check_duplicates:
                if len(bounding_boxes) != len(set(bounding_boxes)):
                    dup_class = [item for item in bounding_boxes if bounding_boxes.count(item) > 1]
                    for dup in set(dup_class):
                        print(f'! Duplicate {dup} in', bbox_file)
    print('Number of frames:', nframes)
    return class_counts

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Count bounding boxes per class from bounding box txt files.')
    parser.add_argument('--bboxes_folder', required=True, help='Directory with the bounding boxes files')
    parser.add_argument('--group_classes', action='store_true', help='Count classes with same name but different number as one class')
    parser.add_argument('--check_duplicates', action='store_true', help='Check if there is only one bbox per class per frame')
    return parser.parse_args()

# Main
if __name__ == '__main__':
    args = parse_arguments()

    # Count bounding boxes
    class_counts = count_bboxes(args.bboxes_folder, args.check_duplicates)

    # Print the result
    print('-'*32)
    print('Bounding boxes count per class:')
    class_names = sorted(class_counts.keys())
    for class_name in class_names:
        print('   ', class_name, class_counts[class_name])
    
    # In case of group classes with same name but different number
    sums = defaultdict(int)
    pattern = re.compile(r'([a-zA-Z_]+)')
    if args.group_classes:
        for key, value in class_counts.items():
            mmatch = pattern.match(key)
            if mmatch:
                prefix = mmatch.group(1)
                sums[prefix] += value
    
        print('-'*32)
        print('Bounding boxes grouped by class:')
        for prefix, total in sums.items():
            print(f'    {prefix}: {total}')

