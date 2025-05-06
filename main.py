import sys
import os
import argparse
import time
import datetime
import cv2
import numpy as np
import pyautogui
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm  # Import tqdm for progress bar

# from stopwatch import StopWatch
# from ImagePlayerFunction import ImagePlay
from gaze_tracker import GazeTracker
from calibration import calibrate
from screen import Screen

# Add an argument to accept the input image file
parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, help='Path to the input image file')


# Use the input image path provided by the user
input_image_path = r"C:\Users\shema\Downloads\pic0.png"

URL = 0

RES_SCREEN = pyautogui.size()  # RES_SCREEN[0] -> width
# RES_SCREEN[1] -> height
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def nothing(val):
    pass

def GaussianMask(sizex, sizey, sigma=33, center=None, fix=1):
    """
    sizex  : mask width
    sizey  : mask height
    sigma  : gaussian Sd
    center : gaussian mean
    fix    : gaussian max
    return gaussian mask
    """
    x = np.arange(0, sizex, 1, float)
    y = np.arange(0, sizey, 1, float)
    x, y = np.meshgrid(x, y)
    
    if center is None:
        x0 = sizex // 2
        y0 = sizey // 2
    else:
        if np.isnan(center[0]) == False and np.isnan(center[1]) == False:            
            x0 = center[0]
            y0 = center[1]        
        else:
            return np.zeros((sizey, sizex))

    return fix * np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / sigma**2)

def Fixpos2Densemap(fix_arr, width, height, imgfile, alpha=0.5, threshold=10):
    """
    fix_arr   : fixation array number of subjects x 3 (x, y, fixation)
    width     : output image width
    height    : output image height
    imgfile   : image file (optional)
    alpha     : merge rate imgfile and heatmap (optional)
    threshold : heatmap threshold (0~255)
    return heatmap 
    """
    heatmap = np.zeros((height, width), np.float32)
    for n_subject in tqdm(range(fix_arr.shape[0])):
        heatmap += GaussianMask(width, height, 33, (fix_arr[n_subject, 0], fix_arr[n_subject, 1]),
                                fix_arr[n_subject, 2])

    # Normalization
    heatmap = heatmap / np.amax(heatmap)
    heatmap = heatmap * 255
    heatmap = heatmap.astype("uint8")
    
    if imgfile.any():
        # Resize heatmap to imgfile shape 
        h, w, _ = imgfile.shape
        heatmap = cv2.resize(heatmap, (w, h))
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create mask
        mask = np.where(heatmap <= threshold, 1, 0)
        mask = np.reshape(mask, (h, w, 1))
        mask = np.repeat(mask, 3, axis=2)

        # Merge images
        merged_image = imgfile * mask + heatmap_color * (1 - mask)
        merged_image = merged_image.astype("uint8")
        merged_image = cv2.addWeighted(imgfile, 1 - alpha, merged_image, alpha, 0)
        return merged_image

    else:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

def main():
    # Load the input image
    img = cv2.imread(input_image_path)

    # Initialize the camera (you may need to adjust this based on your camera setup)
    camera = cv2.VideoCapture(URL)

    gaze_tracker = GazeTracker()
    screen = Screen(SCREEN_WIDTH, SCREEN_HEIGHT)

    cv2.namedWindow("frame")
    cv2.createTrackbar('threshold', 'frame', 0, 255, nothing)
    cv2.setTrackbarPos('threshold', 'frame', 1)

    screen.clean()
    screen.show()

    os.makedirs('images', exist_ok=True)

    gaze_regions = defaultdict(int)
    gaze_data = []

    while True:
        _, frame = camera.read()

        start = time.time()

        gaze_tracker.update(frame)

        end = time.time()

        cv2.namedWindow("frame")
        dec_frame = gaze_tracker.eye_tracker.decorate_frame()

        dec_frame = cv2.resize(dec_frame, (int(FRAME_WIDTH / 2), int(FRAME_HEIGHT / 2)))

        cv2.moveWindow("frame", 0, 0)
        cv2.imshow('frame', dec_frame)

        try:
            gaze = gaze_tracker.get_gaze()
        except:
            gaze = None
            screen.print_message("CALIBRATION REQUIRED!")
            screen.show()
            print("CALIBRATION REQUIRED!")

        print("GAZE: {}".format(gaze))

        if gaze:
            screen.update(gaze)
            screen.refresh()
            try:
                pyautogui.moveTo(gaze[0] + ((RES_SCREEN[0] - screen.width) // 2), gaze[1] + 25)
            except:
                pass

            # Update gaze time in regions of interest
            x, y = gaze
            region_x = x // (SCREEN_WIDTH // 3)
            region_y = y // (SCREEN_HEIGHT // 3)
            region = (region_x, region_y)
            gaze_regions[region] += 1

            # Collect gaze data for heatmap
            gaze_data.append((gaze[0], gaze[1], end - start))

        print("TIME: {:.3f} ms".format(end * 1000 - start * 1000))       

        k = cv2.waitKey(1) & 0xff
        if k == 1048603 or k == 27:  # esc to quit
            break
        if k == ord('c'):  # c to calibrate
            screen.mode = "calibration"
            screen.draw_center()
            calibrate(camera, screen, gaze_tracker)

    camera.release()
    cv2.destroyAllWindows()

    # Create a bar diagram to visualize gaze time in different regions
    regions = list(gaze_regions.keys())
    gaze_times = list(gaze_regions.values())

    plt.bar([str(region) for region in regions], gaze_times)
    plt.xlabel('Gaze Region')
    plt.ylabel('Gaze Time')
    plt.title('Gaze Time in Different Regions')
    plt.show()   
    num_subjects = 40
    fix_arr = np.random.randn(num_subjects, 3)
    fix_arr -= fix_arr.min()
    fix_arr /= fix_arr.max()
    fix_arr[:, 0] *= SCREEN_WIDTH
    fix_arr[:, 1] *= SCREEN_HEIGHT

    # Create heatmap
    heatmap = Fixpos2Densemap(fix_arr, img.shape[1], img.shape[0], img, 0.7, 5)

    # Display the image with the heatmap
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.axis('off')
    plt.title('Fixation Heatmap on Image')
    plt.show()

if __name__ == '__main__':
    main()
