import os
import glob
import shutil
import argparse
import numpy as np
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
from concurrent.futures import ThreadPoolExecutor
import cv2
import math

# Initialize a global image store
imagestore = []

# Argument parsing
parser = argparse.ArgumentParser(description='Source Video path')
parser.add_argument('source_vid_path', type=str, help="Path to the source video directory")
parser.add_argument('fps', type=int, help="Frames per second to extract from videos (only for video files)")
parser.add_argument('output_file', type=str, help="Output file name for the numpy array (without extension)")
args = parser.parse_args()

video_source_path = args.source_vid_path
fps = args.fps
output_file = args.output_file


def create_dir(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def remove_old_images(path):
    """Removes all .jpg and .png images in the specified directory."""
    filelist = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
    for f in filelist:
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error deleting file {f}: {e}")


def is_video_file(filename):
    """Checks if the file is a video based on its extension."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    return filename.lower().endswith(video_extensions)


def extract_frames(video, video_source_path, framepath, fps):
    """Extracts frames from a video using OpenCV."""
    cap = cv2.VideoCapture(os.path.join(video_source_path, video))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % math.floor(frame_rate / fps) == 0:
            filename = os.path.join(framepath, f"{video}_frame{count:03d}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
    cap.release()


def process_videos(video_source_path, fps, framepath):
    """Processes all videos in the directory concurrently."""
    videos = [v for v in os.listdir(video_source_path) if is_video_file(v)]
    print(f"Found {len(videos)} videos")

    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit to 2 threads for testing
        for video in videos:
            print(f"Starting processing of video: {video}")
            executor.submit(extract_frames, video, video_source_path, framepath, fps)
            print(f"Finished submitting video: {video}")


def process_img(img_name, read_path, write=True, write_path=None, res_shape=(227, 227)):
    """Processes an image, resizes it, converts it to 8-bit grayscale if needed, and saves it if specified."""
    if write and write_path is None:
        raise TypeError(
            'The value of argument cannot be `None` when, `write` is set to True. Provide a valid path, '
            'where processed image should be stored!')

    # Load the image as-is, without forcing a color mode
    img = cv2.imread(read_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Warning: Unable to load image: {read_path}. Skipping.")
        return None

    # Check if image is 64-bit and convert to 8-bit if necessary
    if img.dtype == np.float64 or img.dtype == np.uint16:
        print(f"Warning: Image {read_path} is {img.dtype}-bit. Converting to 8-bit.")
        img = np.clip((img / img.max()) * 255, 0, 255).astype(np.uint8)

    # If the image is already single channel (grayscale), we can skip RGB conversion
    if len(img.shape) == 2:
        gray = img
    else:
        # Convert to grayscale if not already
        rgb_weights = [0.2989, 0.5870, 0.1140]
        gray = np.dot(img[..., :3], rgb_weights)

    # Resize the image
    gray = cv2.resize(gray, res_shape)

    if write:
        os.makedirs(write_path, exist_ok=True)
        cv2.imwrite(os.path.join(write_path, img_name), gray)

    return gray


def process_images(root_path, frames_ext='.jpg'):
    """Processes all extracted frames or images in the directory and its subdirectories."""
    img_list = []
    # Recursively find all images in subdirectories
    for subdir, _, _ in os.walk(root_path):
        images = glob.glob(os.path.join(subdir, f'*{frames_ext}'))
        print(f"Looking for images in: {subdir}")
        print(f"Found {len(images)} images with extension {frames_ext}")

        with ThreadPoolExecutor() as executor:
            for image in images:
                img_name = os.path.basename(image)
                write_path = os.path.join(subdir, 'Processed')
                gray = executor.submit(process_img, img_name, image, write=True, write_path=write_path)
                result = gray.result()
                if result is not None:
                    img_list.append(result)
    return img_list


def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes."""
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    return clips


def get_training_set(video_source_path):
    """ Generates training sequences. """
    clips = []
    for f in sorted(os.listdir(video_source_path)):
        directory_path = os.path.join(video_source_path, f)
        if os.path.isdir(directory_path):
            all_frames = []
            for c in sorted(os.listdir(directory_path)):
                img_path = os.path.join(directory_path, c)
                if str(img_path).endswith(".tif"):
                    img = Image.open(img_path).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    return clips


def global_normalization(img_list, name=None, path='Train_Data', save_data=True):
    """Normalizes images, reshapes them, and optionally saves them as a NumPy array."""
    img_arr = np.array(img_list)

    # Assuming img_arr has the shape (num_clips, sequence_size, height, width, channels)
    num_clips, sequence_size, height, width, channels = img_arr.shape

    # Reshape the array to merge sequence size and num_clips for normalization
    img_arr = img_arr.reshape(-1, height, width, channels)

    # Normalize the images
    img_arr = (img_arr - img_arr.mean()) / img_arr.std()
    img_arr = np.clip(img_arr, 0, 1)

    # Reshape it back to its original shape
    img_arr = img_arr.reshape(num_clips, sequence_size, height, width, channels)

    if save_data:
        if name is None:
            raise TypeError(
                'The value of the `name` argument cannot be `None` type, when `save_data` is set to True. Provide '
                'value with `str` datatype.')
        if '.npy' not in name:
            name += '.npy'
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, name), img_arr)
        print(f'\nData Saved Successfully at this path: {path}\n')

    return img_arr


def main():
    if any(is_video_file(f) for f in os.listdir(video_source_path)):
        framepath = os.path.join(video_source_path, 'frames')
        create_dir(framepath)
        remove_old_images(framepath)
        process_videos(video_source_path, fps, framepath)
        frames_ext = '.jpg'
    else:
        framepath = video_source_path
        frames_ext = '.tif'
        img_list = process_images(framepath, frames_ext)

    if 'img_list' not in locals():
        img_list = process_images(framepath, frames_ext)

    if len(img_list) == 0:
        print("No images were processed. Please check the video/image files and paths.")
        exit()

    # Normalize and save the processed images
    clips = get_training_set(video_source_path)
    global_normalization(clips, name=output_file, path='Processed_Data', save_data=True)


if __name__ == "__main__":
    main()
