import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Based on
# http://nghiaho.com/uploads/code/videostab.cpp

# In frames. The larger the more stable the video, but less reactive to sudden panning
WINDOW_SIZE = 30
# dx, dy, dangle
N_MEASUREMENTS = 3


def box_filter_convolve(y, window_size):
    #[n_frames, ]
    box_filter = np.ones(window_size)/window_size
    y_pad = np.lib.pad(y, (window_size, window_size), 'edge')
    y_smooth = np.convolve(y_pad, box_filter, mode='same')
    y_smooth = y_smooth[window_size:-window_size]
    assert y.shape == y_smooth.shape, print(y.shape, y_smooth.shape)
    return y_smooth


def plot_curves(trajectory, smoothed_trajectory):
    x = range(trajectory.shape[0])

    y1 = trajectory[:,0]
    y2 = smoothed_trajectory[:,0]
    plt.figure(figsize=(20//2, 10//2))
    plt.plot(x, y1, 'g')
    plt.plot(x, y2, 'y')
    plt.savefig('data/curves_x.png', bbox_inches='tight')

    y1 = trajectory[:,1]
    y2 = smoothed_trajectory[:,1]
    plt.figure(figsize=(20//2, 10//2))
    plt.plot(x, y1, 'g')
    plt.plot(x, y2, 'y')
    plt.savefig('data/curves_y.png', bbox_inches='tight')


def stabilize_video(input_video_filepath, output_video_filepath):
    cap = cv2.VideoCapture(input_video_filepath)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_filepath, fourcc, fps, (w, h))

    frame_transforms = np.zeros((n_frames-1, N_MEASUREMENTS), np.float32)
    _, prev = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i in range(n_frames-1):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                           maxCorners=200,
                                           qualityLevel=0.01,
                                           minDistance=30,
                                           blockSize=3)

        _, curr = cap.read()
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        # Default settings
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        assert prev_pts.shape == curr_pts.shape

        # Take only valid points
        idx = np.where(status==1)[0]
        _prev_pts = prev_pts[idx]
        _curr_pts = curr_pts[idx]

        m = cv2.estimateRigidTransform(_prev_pts, _curr_pts, fullAffine=False)

        # Q : Scale also included?
        dx = m[0,2]
        dy = m[1,2]
        dangle = np.arctan2(m[1,0], m[0,0])

        frame_transforms[i] = [dx,dy,dangle]

        prev_gray = curr_gray

    trajectory = np.cumsum(frame_transforms, axis=0)

    smoothed_trajectory = np.copy(trajectory)
    for i in range(N_MEASUREMENTS):
        smoothed_trajectory[:,i] = box_filter_convolve(trajectory[:,i], window_size=WINDOW_SIZE)

    diff = smoothed_trajectory - trajectory
    frame_transforms_fixed = frame_transforms + diff

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Write 1st frame unchanged
    _, frame = cap.read()
    out.write(frame)
    # Write n_frames-1 transformed
    for i in range(n_frames-1):
        _, frame = cap.read()

        dx = frame_transforms_fixed[i,0]
        dy = frame_transforms_fixed[i,1]
        dangle = frame_transforms_fixed[i,2]

        m = np.zeros((2,3), np.float32)
        m[0,0] = np.cos(dangle)
        m[0,1] = -np.sin(dangle)
        m[1,0] = np.sin(dangle)
        m[1,1] = np.cos(dangle)
        m[0,2] = dx
        m[1,2] = dy

        frame_tr = cv2.warpAffine(frame, m, (w,h))

        out.write(frame_tr)

    plot_curves(trajectory, smoothed_trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest="input_video_filepath", required=True)
    parser.add_argument('-o', dest="output_video_filepath", required=True)
    args = parser.parse_args()

    input_video_filepath = args.input_video_filepath
    output_video_filepath = args.output_video_filepath

    stabilize_video(input_video_filepath, output_video_filepath)