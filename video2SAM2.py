#!/usr/bin/env python

##########################################################################
# > ./video2SAM2.py --input_folder video2/
##########################################################################

# Normal needed imports:
import os
import requests
import argparse
import cv2
import numpy as np
import time
import zipfile

# The following environment variable is needed because otherwise SAM pytorch model 
# present racing conditions on some CUDA kernel executions (06/07/2024):
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# This might help avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Temporary folder for the SAM video tracking
TEMP_FOLDER = 'temp/'


# Slow imports (only loaded if needed; take a few seconds to be loaded):
def make_slow_imports():
    global torch, build_sam2_video_predictor
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    # Some additional configuration
    if torch.cuda.is_available():
        # Use autocast for the script using bfloat16
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        autocast_context.__enter__()

        # Verify it the GPU is from Ampere Series or more recent
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
# Segment anything network initialization:
# model_size in ['tiny', 'small', 'base_plus', 'large']
def init_SAM_predictor(folder, model_size):
    print('> Initializing SAM 2 model...')
    url = 'https://dl.fbaipublicfiles.com/segment_anything_2/072824/'
    # Check if model is already downloaded
    file_path = os.path.join(folder, f'sam2_hiera_{model_size}.pt')
    print(f'  Model file path: {file_path}')    
    if not os.path.exists(file_path):
        print(f'  Downloading SAM 2 {model_size} model...')
        os.makedirs(folder, exist_ok=True)
        r = requests.get(f'{url}sam2_hiera_{model_size}.pt')
        with open(file_path, 'wb') as file:
            file.write(r.content)
            print(f'    Model downloaded and saved to {file_path}')
    else:
        print(f'    Model was already available in {file_path}.')
    model_cfg = 'sam2_hiera_' + model_size[0]
    if model_size == 'base_plus': model_cfg += '+.yaml'
    else : model_cfg += '.yaml'
    predictor = build_sam2_video_predictor(model_cfg, file_path)

    return predictor

##########################################################################
# Argument parsing and file management
##########################################################################

# Argument parsing:
def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Processes a video and annotates it with the SAM 2 model using the mouse.')
    # File parameters:
    parser.add_argument('--input_folder', type=str, required=True, help='Directory of JPEG frames with filenames like <frame_index>.jpg')
    parser.add_argument('--label_colors', type=str, default='label_colors.txt', help='File with label colors')
    parser.add_argument('--load_folder', type=str, default='annotations/', help='Folder with masks to load')
    parser.add_argument('--output_folder', type=str, default='annotations/', help='Output folder for masks')
    parser.add_argument('--backup_folder', type=str, default='backups/', help='Folder for backups')
    parser.add_argument('--resize_factor', type=float, default=1.0, help='Resize factor for the frames')
    parser.add_argument('--init_frame', type=int, default=0, help='Initial frame for the video (inclusive)')
    parser.add_argument('--end_frame', type=int, default=-1, help='End frame for the video (inclusive)')
    parser.add_argument('--sam_model_folder', type=str, default='models/', help='Folder to store/load the SAM 2 model')
    parser.add_argument('--model_size', type=str, choices=['tiny', 'small', 'base_plus', 'large'], default='large', help='Select the size of the SAM 2 model: tiny, small, base_plus, or large')
    args = parser.parse_args()
    return args

# Frame reading from folder. Returns frame names and frames:
def frames_from_folder(folder, resize_factor, init_frame, end_frame):
    # Scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    if end_frame == -1: end_frame = len(frame_names) - 1
    frame_names = frame_names[init_frame:end_frame+1]
    # Read all frames
    frames = []
    l = len(frame_names)
    for i, frame_name in enumerate(frame_names):
        frame = cv2.imread(os.path.join(folder, frame_name))
        if resize_factor != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        frames.append(frame)
        printProgressBar(i+1, l, prefix = 'Frames:     ', suffix = 'completed', length = 40)
    # Remove the extension from the frame names
    frame_names = [os.path.splitext(p)[0] for p in frame_names]
    return frame_names, frames

# Read colors from file:
# return a list with the order of the labels (to identify each label with an integer)
# and a dictionary where key is the label name and value is the RGB color 
def labels_colors_from_file(file_path):
    labels = {}
    label_order = []
    with open(file_path, 'r') as file:
        for line in file:
            # Format is: R G B LABELNAME. See KITTI annotations format
            # (https://docs.cvat.ai/docs/manual/advanced/formats/format-kitti/)
            r, g, b, label = line.split()
            if label != 'background':
                labels[label] = (int(r), int(g), int(b))
                label_order.append(label)
    return label_order, labels

# Load masks from folder:
def load_masks(folder, frame_shape, frame_names):
    semantic_folder = os.path.join(folder, 'semantic_rgb/')
    instance_folder = os.path.join(folder, 'instance/')
    if not os.path.exists(semantic_folder) or not os.path.exists(instance_folder):
        print(f'! Error: Folder {folder} does not contain semantic_rgb/ or instance/ subfolders'), 
        return None, None

    # Check if mask and frame shape are the same
    mask0 = cv2.imread(os.path.join(semantic_folder, f'frame_{frame_names[0]}.png'))
    inst0 = cv2.imread(os.path.join(instance_folder, f'frame_{frame_names[0]}.png'), cv2.IMREAD_GRAYSCALE)
    if mask0.shape[:2] != frame_shape:
        print(f'! Error: Frame shape {frame_shape} and mask shape {mask0.shape[:2]} do not match.')
        return None, None
    elif inst0.shape[:2] != frame_shape:
        print(f'! Error: Frame shape {frame_shape} and instance shape {inst0.shape[:2]} do not match.')
        return None, None
    
    # Load the masks and instances
    print(f'    Loading masks from {folder}... ')
    masks, instances = [], []
    l = len(frame_names)
    for i, file in enumerate(frame_names):
        sem_frame_path = os.path.join(semantic_folder, f'frame_{file}.png')
        if not os.path.exists(sem_frame_path):
            print(f'! Error: File {sem_frame_path} does not exist.')
            return None, None
        masks.append(cv2.cvtColor(cv2.imread(sem_frame_path), cv2.COLOR_BGR2RGB))
        # Instances must be (H, W, 1) and not (H, W, 3)
        inst_frame_path = os.path.join(instance_folder, f'frame_{file}.png')
        if not os.path.exists(inst_frame_path):
            print(f'! Error: File {inst_frame_path} does not exist.')
            return None, None
        instances.append(cv2.imread(inst_frame_path, cv2.IMREAD_GRAYSCALE))

        printProgressBar(i+1, l, prefix = 'Loaded masks:', suffix = 'completed', length = 40)
       
    # convert (H, W) to (H, W, 1)
    instances = [np.expand_dims(inst, axis=2) for inst in instances]
    print('    ... done!')
    return masks, instances

# Save masks in folder:
def save_masks(folder, frame_names, sem_masks, instances, bboxes, label_colors, is_backup=False):
    # If is a backup create a subfolder with time
    if is_backup:   
        folder = os.path.join(folder, time.strftime('%Y%m%d_%H%M%S') + '/')
    elif os.path.exists(folder):        # if is not a backup and folder exists, clear the files
        # Ask for confirmation
        answer = input(text_color(f' ! Remove folder {folder} before saving? (y/n): '))
        while answer.lower() != 'y' and answer.lower() != 'n':
            answer = input(text_color('    ! Please answer with \'y\' or \'n\': ', error_color=True))
        if answer.lower() == 'y':
            os.system(f'rm -r {folder}*')
    # Loop over masks and save them
    print(f'Saving masks in {folder}... ')
    # Save the semantic masks
    semantic_folder = os.path.join(folder, 'semantic_rgb/')
    if not os.path.exists(semantic_folder): os.makedirs(semantic_folder)
    l = len(sem_masks)
    i = 0
    for name, mask in zip(frame_names, sem_masks):
        cv2.imwrite(semantic_folder + f'frame_{name}.png', cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        printProgressBar(i+1, l, prefix = 'Segm. masks:', suffix = 'completed', length = 40)
        i += 1
    # Save the instance masks
    instance_folder = os.path.join(folder, 'instance/')
    if not os.path.exists(instance_folder): os.makedirs(instance_folder)
    l = len(instances)
    i = 0
    for name, mask in zip(frame_names, instances):
        # Convert from grayscale in 8 bits to 16 bits grayscale
        cv2.imwrite(instance_folder + f'frame_{name}.png', np.array(mask, dtype=np.uint16)*256)
        printProgressBar(i+1, l, prefix = 'Inst. masks:', suffix = 'completed', length = 40)
        i += 1
    # Save the bboxes
    folder_bboxes = os.path.join(folder, 'bboxes/')
    if not os.path.exists(folder_bboxes): os.makedirs(folder_bboxes)
    # Create a diccionary where key is the color and value is the label
    colors_labels = {v: k for k, v in label_colors.items()}
    l = len(bboxes)
    i = 0
    for name, bboxes_frame in zip(frame_names, bboxes):
        with open(folder_bboxes + f'frame_{name}.txt', 'w') as f:
            for bbox in bboxes_frame:
                (x, y, w, h), color, inst_id = bbox
                label = colors_labels[color]
                f.write(f'{label} {inst_id} {x} {y} {w} {h}\n')
        printProgressBar(i+1, l, prefix = 'Bboxes     :', suffix = 'completed', length = 40)
        i += 1
    print('    ... done!')

# Create a zip file with the KITTI format
def create_zip_kitti(kitti_folder, frames_per_zip, frame_names, sem_masks_rgb, colors_file, label_colors, label_order):
    readme_content = """\
This is a zip file with the KITTI format for importing to CVAT.
(06/08/2024): There is a bug in how CVAT uses the KITTI format. The instance masks are not used.
The instance folder is really the same as the semantic folder but with integers instead of colors."""
    # Number of zip files to create
    l = len(sem_masks_rgb)
    num_zips = l // frames_per_zip
    if l % frames_per_zip > 0:  num_zips += 1
    fframe = int(frame_names[0])
    for b in range(num_zips):
        min_f, max_f = b*frames_per_zip, min((b+1)*frames_per_zip, len(sem_masks_rgb))-1
        output_file = os.path.join(kitti_folder, f'kitti_frames_{min_f+fframe:06d}_to_{max_f+fframe:06d}.zip')
        # Create the zip file
        with zipfile.ZipFile(output_file, 'w') as zipf:
            # Go through the masks
            for i in range(min_f, max_f+1):
                sem_rgb = sem_masks_rgb[i]
                zipf.writestr(f'default/semantic_rgb/frame_{i+fframe:06d}.png', cv2.imencode('.png', cv2.cvtColor(sem_rgb, cv2.COLOR_RGB2BGR))[1].tobytes())
                # There is a bug in how CVAT uses the KITTI format. The instance masks are not used. 
                # The instance folder is really the same as the semantic folder but with integer instead of colors
                zipf.writestr(f'default/instance/frame_{i+fframe:06d}.png', cv2.imencode('.png', rgb_semantic_to_int(sem_rgb, label_colors, label_order))[1].tobytes())
                printProgressBar(i+1, l, prefix = f'    ZIP {b+1}/{num_zips}:', suffix = 'completed', length = 40)
            # Add the label colors file
            with open(colors_file, 'r') as file:
                zipf.writestr('label_colors.txt', file.read())
            # Add the readme file
            zipf.writestr('README.txt', readme_content)
    print(f'    Created {num_zips} ZIP files correctly.')

# Load bboxes from file:
def bboxes_from_file(folder, labels_colors, frame_names):
    folder = os.path.join(folder, 'bboxes/')
    print(f'    Loading bboxes from {folder}... ')
    bboxes = []
    for file in frame_names:
        bboxes.append([])
        with open(os.path.join(folder, f'frame_{file}.txt'), 'r') as f:
            for line in f:
                # Format is: class num_inst bbox_x bbox_y bbox_w bbox_h
                label, num_inst, x, y, w, h = line.split()
                bboxes[-1].append(((int(x), int(y), int(w), int(h)), labels_colors[label], int(num_inst)))
    print('    ... done!')
    return bboxes

# This folder is used for the inference state of SAM 2
# The SAM initialization for video tracking needs a path of a folder of images
# When working with long videos, CUDA may run out of memory
# So we deactivate/activate the video frames by adding/removing an underscore
# from the file extension name (.jpg or .jpg_)
def create_temp_folder(frames, frame_names, temp_folder):
    # Copy the input folder to the temp folder and rename the files
    # (to deactivate the frames)
    if not os.path.exists(temp_folder): os.makedirs(temp_folder)
    l = len(frames)
    for i, name in enumerate(frame_names):
        cv2.imwrite(os.path.join(temp_folder, name + '.jpg_'), frames[i])
        printProgressBar(i+1, l, prefix = 'Temp folder:', suffix = 'completed', length = 40)
    return

# Remove the temp folder
def remove_temp_folder(temp_folder):
    os.system(f'rm -r {temp_folder}')
    return

##########################################################################
# Console printing
##########################################################################

# Return text with codes for colors in console:
def text_color(text, tuple_color=(0, 174, 174), is_background=False, error_color=False):
    if error_color:
        r, g, b = 225, 0,  0
    else:
        r, g, b = tuple_color
    if is_background:
        return f'\033[48;2;{r};{g};{b}m' + '\033[38;2;0;0;0m' + text + '\033[0m'
    else:
        return f'\033[38;2;{r};{g};{b}m' + text + '\033[0m'

# Return text with bold code to print bold in console:
def text_bold(text):
    return f'\033[1m{text}\033[0m'

# Print text in console with color:
def print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length):
    print(chr(27) + '[2J')  # Clear screen
    print('='*70)
    print(text_color(' '*31 + text_bold('CONTROLS')))
    print("""    [left click]: Add positive point for SAM 2
    [right click]: Add negative point for SAM 2
    [l]: Change current label
    [c]: Clear current label points for current frame
    [a]: Clear all points for all frames
    [r]: Reset mask for current label and frame
    [n]: Reset mask for current label and all next frames
        
    [f]: Call SAM 2 for current frame
    [p]: Propagate mask for current label
    [+/-]: Increase/decrease by 1 the number of frames to
           propague segmentation when pressing [p]
    [*/_]: Same as before but by 30 instead of 1
    
    [,/.]: Go to previous/next frame
    [;/:]: Go 10 frames back/forward
          
    [v]: Show/hide mask
    [b]: Show/hide bboxes
    [i]: Swap between semantic and instance masks
    [k]: Create backup
    
    [q] or [ESC]: Quit""")
    print('-'*70)
    print(' '*24+ 'Current frame: ', end='')
    print(text_color(f'{current_frame}/{total_frames-1} ({init_video_frame+current_frame} in video)'))
    print(' '*24 +'Prompt progation: ', end='')
    print(text_color(f'{propagation_length}'))
    print('-'*70)
    print(text_color(' '*29 + text_bold('CURRENT LABEL')))
    for label, color in label_colors.items():
        if label == current_label:
            print(text_color('    ' + label + ' '*(66-len(label)), color, is_background=True))
        else:
            print(text_color('    ' + label, color))
    print('='*70)

# Print iterations progress
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

# Function for printing CUDA out of memory error
def print_out_of_memory_error(e):
    error_lines = str(e).strip().split('\n')
    last_line = error_lines[-1] if error_lines else 'No additional info available.'
    print(text_color(last_line, error_color=True))
    print(text_color('Try to reduce the \'Prompt progation\' length. In my case, using an 8GB GPU with the large model for 1080x1920 images and segmentation across 8 classes, a value of 250 for this parameter works well.'))

##########################################################################
# OpenCV window management
##########################################################################

# Frame navigation:
def navigate_frames(frames, frame_names, init_video_frame, label_colors, sam_predictor, backup_folder, temp_folder, masks, bboxes, instances):
    # PARAMETERS
    H_frame, W_frame = frames[0].shape[:2]
    current_frame, total_frames = 0, len(frames)
    show_mask, show_bboxes, show_instances = True, False, False
    labels_list = list(label_colors.keys())
    current_label = labels_list[0]
    # Possitive and negative points for SAM: points are dicts of {label : (x, y)}
    positive_points = [[] for _ in range(total_frames)]
    negative_points = [[] for _ in range(total_frames)]

    # Mouse callback function
    def click_event(event, x, y, flags, param):
        nonlocal current_frame, show_mask, H_frame, W_frame, current_label
        if event == cv2.EVENT_LBUTTONDOWN:
            if 0 <= x < W_frame and 0 <= y < H_frame:
                positive_points[current_frame].append(((x, y), current_label))
                # Update frame after adding a point
                update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if 0 <= x < W_frame and 0 <= y < H_frame:
                negative_points[current_frame].append(((x, y), current_label))
                # Update frame after adding a point
                update_frame(current_frame, show_mask, show_bboxes, show_instances)
    
    # Function to update the frame
    def update_frame(current_frame, show_mask, show_bboxes, show_instances):
        frame_copy = frames[current_frame].copy()
        if show_instances:
            mask = instances2rgb(instances[current_frame], instance_colors)
        else:
            mask = cv2.cvtColor(masks[current_frame], cv2.COLOR_RGB2BGR)
        # Draw the points on the frame
        for point in positive_points[current_frame]:
            cv2.circle(frame_copy, point[0], 12, (0, 255, 0), -1)
            color = (label_colors[point[1]][2], label_colors[point[1]][1], label_colors[point[1]][0])
            cv2.circle(frame_copy, point[0], 9, color, -1)
        for point in negative_points[current_frame]:
            cv2.circle(frame_copy, point[0], 12, (0, 0, 255), -1)
            color = (label_colors[point[1]][2], label_colors[point[1]][1], label_colors[point[1]][0])
            cv2.circle(frame_copy, point[0], 9, color, -1)
        # Draw the bboxes on the frame
        if show_bboxes:
            for bbox in bboxes[current_frame]:
                (x, y, w, h), color, _ = bbox
                cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (color[2], color[1], color[0]), 3)
                if show_mask:
                    cv2.rectangle(mask, (x, y), (x+w, y+h), (color[2], color[1], color[0]), 3)
        # Show the frame
        if not show_mask:
            cv2.imshow('Video', frame_copy)
        else:   # Stack horzontally frame and mask
            frame_masked = add_frame_mask(frame_copy, mask)
            cv2.imshow('Video', np.hstack((frame_masked, mask)))
    
    # Function to append the mask, instances and bboxes in the lists
    def append_mask_in_lists(bool_mask, label, current_frame):
        # Save the mask with color
        masks[current_frame][bool_mask] = label_colors[label]
        # Add instance to the mask
        instance_number = len(np.unique(instances[current_frame]))
        instances[current_frame][bool_mask] = instance_number
        # Save the bbox ((x, y, w, h), color, instance_id)
        mask = np.zeros_like(bool_mask, dtype=np.uint8)
        mask[bool_mask] = 1
        bboxes[current_frame].append((cv2.boundingRect(np.array(mask, dtype=np.uint8)), label_colors[label], instance_number))

    # Create window and set callback
    cv2.namedWindow('Video', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('Video', click_event)
    # Create 500 random colors for the instance masks
    instance_colors = np.random.randint(0, 255, (500, 3))
    update_frame(0, True, False, False)

    # Screen loop
    propagation_length, propagation_changed = 250, False
    inf_state_init_frame, inference_state = 0, None
    print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
    while True:
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or key == 27: break # 'q' or 'ESC' key
        elif key == ord(','):   # Previous frame
            current_frame = max(0, current_frame - 1)
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('.'):   # Next frame
            current_frame = min(total_frames - 1, current_frame + 1)
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord(';'):   # Backward 10 frames
            current_frame = max(0, current_frame - 10)
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord(':'):  # Forward 10 frames
            current_frame = min(total_frames - 1, current_frame + 10)
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('r') or key == ord('n'):  # Reset masks
            if key == ord('r'): processed_frames = [current_frame]
            else: processed_frames = list(range(current_frame, total_frames))
            l = len(processed_frames)
            for i in processed_frames:
                # Remove instances from the mask
                instances2delete = [bbox[2] for bbox in bboxes[i] if bbox[1] == label_colors[current_label]]
                printProgressBar(i+1-processed_frames[0], l, prefix = 'Reseting:   ', suffix = 'completed', length = 40)
                if instances2delete != []:
                    for inst in instances2delete:
                        instances[i][instances[i] == inst] = 0
                    # Adjust the instances numbers
                    instances_copy = instances[i].copy()
                    for inst in instances2delete:
                        instances[i][instances_copy > inst] -= 1
                    # When color is equal to label_color[current_label], set it to (0, 0, 0)
                    mask_bool = np.all(masks[i] == label_colors[current_label], axis=2)
                    masks[i][mask_bool] = (0, 0, 0)
                    # Remove bboxes too
                    new_bboxes = []
                    for bbox in bboxes[i]:
                        if bbox[1] != label_colors[current_label]: # if bbox is not from the current label
                            # Upadate the instance id
                            nbbox = bbox
                            for inst in instances2delete:
                                if bbox[2] > inst:
                                    nbbox = ((nbbox[0][0], nbbox[0][1], nbbox[0][2], nbbox[0][3]), nbbox[1], nbbox[2]-1)
                            new_bboxes.append(nbbox)
                    bboxes[i] = new_bboxes
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('a'):  # Clear all points for all frames
            positive_points = [[] for _ in range(total_frames)]
            negative_points = [[] for _ in range(total_frames)]
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('c'):  # Clear all points for current frame and current label
            positive_points[current_frame] = [ p for p in positive_points[current_frame] if p[1] != current_label]
            negative_points[current_frame] = [ p for p in negative_points[current_frame] if p[1] != current_label]
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('l'):  # Change label
            current_label = labels_list[(labels_list.index(current_label) + 1) % len(labels_list)]
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
        elif (key == ord('f') or key == ord('p')) and positive_points[current_frame] != []:   # Call SAM 2
            # 1. REDFINE THE INFERENCE STATE IF NECESSARY
            # If it is not defined / if current frame is not in the range / 
            # if state initial frame is not equal the current frame
            if (inference_state is None or
                (key == ord('f') and not (inf_state_init_frame <= current_frame < inf_state_init_frame + propagation_length)) 
                or (key == ord('p') and (inf_state_init_frame != current_frame or propagation_changed))):
                # Redefine the inference state
                inf_state_init_frame = current_frame
                t0 = time.time()
                propagation_changed = False
                print('> Establishing inference state for SAM 2...')
                del inference_state # for releasing space
                try:
                    inference_state = create_SAM_inference_state(sam_predictor, temp_folder, current_frame, 
                            min(current_frame + propagation_length, total_frames-1), init_video_frame)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print_out_of_memory_error(e)
                        inference_state = None
                        torch.cuda.empty_cache()
                        continue
                    else:   raise e     # Another error
                print(f'  Inference state established in {time.time() - t0:.2f} seconds.')
            # 2. ADD THE NEW POINTS TO THE PREDICTOR
            print('> Calling SAM 2 for frame ' + str(current_frame) + '...')
            t0 = time.time()
            boolean_masks, objects_list = sam2_add_new_points(sam_predictor, inference_state, label_colors, 
                current_frame-inf_state_init_frame, positive_points[current_frame], negative_points[current_frame])
            print(f'  Done! Frame segmented in {time.time() - t0:.2f} seconds.')
            # 3.1 IF ONLY FOR CURRENT FRAME, SAVE THE MASKS
            if key == ord('f') and boolean_masks != None:
                # Save the masks
                for label in boolean_masks.keys():
                    append_mask_in_lists(boolean_masks[label], label, current_frame)
                # Clear the points
                positive_points[current_frame] = []
                negative_points[current_frame] = []
            elif boolean_masks != None:   # 3.2 IF PROPAGATION, SHOW THE MASKS TO CONFIRM PROPAGATION
                # Create the RGB mask
                rgb_mask = np.zeros((H_frame, W_frame, 3), dtype=np.uint8)
                for label in boolean_masks.keys():
                    rgb_mask[boolean_masks[label]] = label_colors[label]
                # Show it in new window ABOVE the main window
                x, y, width, height = cv2.getWindowImageRect('Video')
                cv2.namedWindow('Press any key to confirm, [ESC] to cancel', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
                cv2.moveWindow('Press any key to confirm, [ESC] to cancel', x, y)
                cv2.resizeWindow('Press any key to confirm, [ESC] to cancel', width, height)
                bgr_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
                frame_masked = add_frame_mask(frames[current_frame].copy(), bgr_mask)
                cv2.imshow('Press any key to confirm, [ESC] to cancel', np.hstack((frame_masked, bgr_mask)))
                print(text_color('> Do you want to propagate the masks?'))
                print(text_color('  Press any key to confirm, [ESC] to cancel'))
                conf_key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow('Press any key to confirm, [ESC] to cancel')
                if conf_key != 27:  # If any other key, save the masks
                    t0 = time.time()
                    # Call the SAM 2 propagation function
                    video_segments = sam2_prompt_propagation(sam_predictor, inference_state, objects_list)
                    print(f'  Done! {propagation_length} frames segmented in {time.time() - t0:.2f} seconds.')
                    if video_segments != None:
                        print('  Extracting masks...')
                        l = len(video_segments.keys())
                        for i, num_frame in enumerate(video_segments.keys()):
                            boolean_masks = video_segments[num_frame]
                            for label in boolean_masks.keys():
                                append_mask_in_lists(boolean_masks[label], label, num_frame+inf_state_init_frame)
                            printProgressBar(i+1, l, prefix = 'Masks:      ', suffix = 'completed', length = 40)
                        # Remove ALL the points
                        positive_points = [[] for _ in range(total_frames)]
                        negative_points = [[] for _ in range(total_frames)]
                else:   # If ESC, cancel the propagation
                    print(text_color('  Propagation canceled. Discarding the masks.', error_color=True))
            # Reset the inference state (just for reseting tracking)
            sam_predictor.reset_state(inference_state)
            # Clear torch cache
            torch.cuda.empty_cache()
            # Update the frame
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('v'):   # Show/hide mask
            show_mask = not show_mask
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('b'):   # Show/hide bboxes
            show_bboxes = not show_bboxes
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('i'):   # Swap between semantic and instance masks
            show_instances = not show_instances
            update_frame(current_frame, show_mask, show_bboxes, show_instances)
        elif key == ord('k'):   # Create backup
            save_masks(backup_folder, frame_names, masks, instances, bboxes, label_colors, is_backup=True)
        elif key == ord('+'):   # Increase propation length by 1
            propagation_length += 1
            propagation_changed = True
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
        elif key == ord('-'):   # Decrease propation length by 1
            propagation_length = max(1, propagation_length - 1)
            propagation_changed = True
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
        elif key == ord('*'):   # Increase propation length by 30
            propagation_length += 30
            propagation_changed = True
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
        elif key == ord('_'):   # Decrease propation length by 30
            propagation_length = max(1, propagation_length - 30)
            propagation_changed = True
            print_console(current_label, label_colors, current_frame, init_video_frame, total_frames, propagation_length)
        elif key == ord('d'):   # Just for debugging
            print(np.unique(instances[current_frame]))

    cv2.destroyAllWindows()
    return masks, instances, bboxes

##########################################################################
# SAM functions
##########################################################################

# Function to create the inference state for SAM 2 processing
def create_SAM_inference_state(sam_predictor, temp_folder, init_frame, last_frame, init_video_frame):
    # Deactivate the inference state for the frames that are not in the range
    frames_name = os.listdir(temp_folder)
    frames_name.sort()
    for name in frames_name:
        extension = os.path.splitext(name)[-1]
        if extension in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.jpg_', '.jpeg_', '.JPG_', '.JPEG_']:
            frame_number = int(os.path.splitext(name)[0]) - init_video_frame
            # Activate the frames within the range (changing the extension from .jpg_ to .jpg)
            if frame_number >= init_frame and frame_number <= last_frame:
                if extension[-1] == '_':
                    old_name = os.path.join(temp_folder, name)
                    os.rename(old_name, old_name[:-1])
            elif extension[-1] != '_': # Deactivate the frames outside the range
                old_name = os.path.join(temp_folder, name)
                os.rename(old_name, old_name + '_')

    # Initialize the inference state on the activated frames
    inference_state = sam_predictor.init_state(video_path=temp_folder)
    return inference_state

# Fucntion to call SAM 2 for a frame, return a dict of the boolean masks, where keys are 
# the string label and value are the mask. Returns also a list with the order of the objects
def sam2_add_new_points(sam_predictor, inference_state, label_colors, frame_idx, positive_points_frame, negative_points_frame):
    # For each label, add the points
    # The objects are the labels and the index in the list are the objects ids
    objects_list = list(label_colors.keys())
    out_obj_ids = None
    for nl, label in enumerate(objects_list):
        pp = [ point[0] for point in positive_points_frame if point[1] == label ]
        if pp != []:
            pn = [point[0] for point in negative_points_frame if point[1] == label]
            # Calling the SAM 2 predictor
            try:
                _, out_obj_ids, out_mask_logits = sam_predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=nl,  # obj_id is the index of the label in the list of labels
                    points=np.array(pp + pn),
                    labels=np.array([1]*len(pp) + [0]*len(pn)))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print_out_of_memory_error(e)
                    torch.cuda.empty_cache()
                    return None, None
                else:   raise e     # Another error
                
    # The result of the last call is a list of masks for each object
    if out_obj_ids:
        bool_masks = {}
        h, w = inference_state['video_height'], inference_state['video_width']
        for i, out_obj_id in enumerate(out_obj_ids):
            bool_mask = (out_mask_logits[i] > 0.0).cpu().numpy().reshape(h, w, 1).squeeze()
            bool_masks[objects_list[out_obj_id]] = bool_mask
        return bool_masks, objects_list
    else:
        return None, None

# Function to propagate the mask in the video. Returns a dict of boolean mask per frame.
# The boolean mask are stored in a dict, where keys are string labels and values are the mask
def sam2_prompt_propagation(sam_predictor, inference_state, objects_list):
    video_segments = {}  # Contains the per-frame segmentation results
    h, w = inference_state['video_height'], inference_state['video_width']
    try:
        # Clear torch CUDA memory
        torch.cuda.empty_cache()
        # Call the SAM 2 propagation function
        for out_frame_idx, out_obj_ids, out_mask_logits in sam_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    objects_list[out_obj_id]: (out_mask_logits[i] > 0.0).cpu().numpy().reshape(h, w, 1).squeeze()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print_out_of_memory_error(e)
            torch.cuda.empty_cache()
            return None
        else:   raise e     # Another error
    return video_segments


##########################################################################
# Image processing functions
##########################################################################

# Add the mask to the frame. 
# Mask must be in BGR color space and the result is in BGR too
def add_frame_mask(frame, mask):
    # Convert mask to 3 channels if needed
    if mask.shape[:2] == 1:
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask == 1] = (255, 255, 255)
    else:
        colored_mask = mask
    # Add transparency to the mask  
    colored_mask_rgba = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2BGRA)
    colored_mask_rgba[:, :, 3] = 200
    # Add the mask to the frame
    rbga_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    cv2.addWeighted(rbga_frame, 1, colored_mask_rgba, 0.5, 0, rbga_frame)
    return cv2.cvtColor(rbga_frame, cv2.COLOR_BGRA2BGR)

# Convert instances map to RGB
def instances2rgb(instance_map, instance_colors):
    # Take the values of the instance map
    instances = [ inst for inst in np.unique(instance_map) if inst != 0 ]
    # Create the RGB mask and color every instance
    rgb_instance_map = np.zeros((instance_map.shape[0], instance_map.shape[1], 3), dtype=np.uint8)
    instance_map = instance_map.squeeze()
    for i, instance in enumerate(instances):
        mask = (instance_map == instance)
        rgb_instance_map[mask] = instance_colors[i % len(instance_colors)]
    return rgb_instance_map

# Create bboxes from masks
def bboxes_from_masks(sem_masks, instances):
    print(f'    Computing bboxes from masks...')
    bboxes = [[] for _ in range(len(sem_masks))]
    l = len(sem_masks)
    for i in range(len(sem_masks)):
        unique_instances = np.unique(instances[i])[1:] # 0 is the background
        for inst_id in unique_instances:
            # Find where the instance is equal to the actual
            y, x = np.where(np.squeeze(instances[i]) == inst_id)
            # Bounding box
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            # Extract the region of interest
            roi = sem_masks[i][y_min:y_max+1, x_min:x_max+1]
            # Convert roi to list of colors and counts
            colors, counts = np.unique(roi.reshape(-1, 3), axis=0, return_counts=True)
            # Get the most common color
            fcolor = colors[np.argmax(counts)]
            # If this is the background color, take the second most common
            if np.all(fcolor == (0, 0, 0)):
                fcolor = colors[np.argsort(counts)[-2]]
            # Save the bbox
            bboxes[i].append(((x_min, y_min, x_max-x_min+1, y_max-y_min+1), (int(fcolor[0]), int(fcolor[1]), int(fcolor[2])), inst_id))
        printProgressBar(i+1, l, prefix = 'Bboxes:     ', suffix = 'completed', length = 40)
    print('    ... done!')
    return bboxes

# Transforms the RGB semantic mask to an integer mask according to the order of the labels
def rgb_semantic_to_int(mask, label_colors, label_order):
    mask_int = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for i, label in enumerate(label_order):
        mask_int[ np.all(mask == label_colors[label], axis=-1) ] = i+1
    return np.array(mask_int, dtype=np.uint16)*256

##########################################################################
# Main
##########################################################################
if __name__ == '__main__':
    # 0. Parse input arguments:
    args = parse_arguments()

    # 1. Read video:
    print(f'> Opening video {args.input_folder}...')
    if args.resize_factor <= 0 or args.resize_factor > 1:
        print(text_color(f'! Error: Resize factor must be in the range (0, 1].', error_color=True))
        exit(1)
    elif args.init_frame < 0 or (args.end_frame != -1 and args.init_frame > args.end_frame):
        print(text_color(f'! Error: Initial frame and end frame must be positive and init_frame <= end_frame.', error_color=True))
        exit(1)

    frame_names, frames = frames_from_folder(args.input_folder, args.resize_factor, 
                                        args.init_frame, args.end_frame)
    
    if frames is None:
        print(text_color(f'! Error opening video {args.input_folder}', error_color=True))
        exit(1)
    total_frames = len(frames)
    print(f'    Video opened with {total_frames} frames of size {frames[0].shape[1]}x{frames[0].shape[0]}.')

    # 2. Read labels (dict where key is label name and value is RGB color)
    label_order, label_colors = labels_colors_from_file(args.label_colors)

    # 3. Load masks from folder:
    loaded_masks, loaded_bboxes, loaded_instances = None, None, None
    if os.path.exists(args.load_folder):
        answer = input(text_color(f'> Do you want to load masks from {args.load_folder}? (y/n): '))
        while answer.lower() != 'y' and answer.lower() != 'n':
            answer = input(text_color('    ! Please answer with \'y\' or \'n\': ', error_color=True))
        if answer.lower() == 'y':
            loaded_masks, loaded_instances = load_masks(args.load_folder, frames[0].shape[:2], frame_names)

            if loaded_masks is None:    # If the masks could not be loaded
                print(text_color(f'! Error: Masks could not be loaded from {args.load_folder}', error_color=True))
                exit(1)
                
            # If the folder load_folder/bboxes exitsts and its not empty
            bboxes_path = os.path.join(args.load_folder, 'bboxes/')
            if os.path.exists(bboxes_path) and os.listdir(bboxes_path):
                loaded_bboxes = bboxes_from_file(args.load_folder, label_colors, frame_names)
            else:
                loaded_bboxes = bboxes_from_masks(loaded_masks, loaded_instances)

    # If the masks are not loaded, create empty lists
    if loaded_masks is None:
        loaded_masks = [np.zeros((frames[0].shape[0], frames[0].shape[1], 3), dtype=np.uint8) for _ in range(total_frames)]
    if loaded_bboxes is None:
        loaded_bboxes = [[] for _ in range(total_frames)]
    if loaded_instances is None:
        loaded_instances = [np.zeros((frames[0].shape[0], frames[0].shape[1], 1), dtype=np.uint8) for _ in range(total_frames)]

    # 4. Initialize the SAM model:
    make_slow_imports()
    sam_predictor = init_SAM_predictor(args.sam_model_folder, args.model_size)
    # Create the temporal folder for the inference state
    create_temp_folder(frames, frame_names, TEMP_FOLDER)

    # 5. Navigate through frames and click the points
    sem_masks, instances, bboxes = navigate_frames(frames, frame_names, args.init_frame, label_colors, sam_predictor, 
            args.backup_folder, TEMP_FOLDER, loaded_masks, loaded_bboxes, loaded_instances)

    # 6. Ask for saving the masks
    answer = input(text_color('> Do you want to save the masks? (y/n): '))
    while answer.lower() != 'y' and answer.lower() != 'n':
        answer = input(text_color('    ! Please answer with \'y\' or \'n\': ', error_color=True))
    if answer.lower() == 'y':
        save_masks(args.output_folder, frame_names, sem_masks, instances, bboxes, label_colors)
        # 7. Ask for exporting to KITTI format
        answer = input(text_color('> Do you want to export in KITTI format (for importing to CVAT)? (y/n): '))
        while answer.lower() != 'y' and answer.lower() != 'n':
            answer = input(text_color('    ! Please answer with \'y\' or \'n\': ', error_color=True))
        if answer.lower() == 'y':
            kitti_folder = input('    Output file for the KITTI exportation [default: cvat/]: ')
            # Default folder
            if kitti_folder == '':  kitti_folder = 'cvat/'
            # Ask for the number of frames per zip file
            frames_per_zip = input(f'    Frames per ZIP file [default: 300, total: {total_frames}]: ')
            while frames_per_zip != '' and not frames_per_zip.isdigit():
                frames_per_zip = input(text_color('    ! Please enter a number: ', error_color=True))
            # Default value
            if frames_per_zip == '':    frames_per_zip = 300
            else: frames_per_zip = int(frames_per_zip)
            # Check the number of frames per zip
            if frames_per_zip > total_frames:
                frames_per_zip = total_frames
            # If folder does not exist, create it
            if not os.path.exists(kitti_folder): os.makedirs(kitti_folder)
            # Save the masks in KITTI format
            create_zip_kitti(kitti_folder, frames_per_zip, frame_names, sem_masks, args.label_colors, label_colors, label_order)

    # Remove the temporal folder and exit
    remove_temp_folder(TEMP_FOLDER)
    exit(0)
    
