import os 
import sys 
import numpy as np 
from glob import glob
from tqdm import tqdm, trange 
from collections import defaultdict 
import tensorflow as tf 
import imageio
import cv2
import pandas as pd 


INPUT_DIR = "/<path_to_data>/bridgev2_processed"
OUTPUT_DIR = "/<path_to_data>/bridge_webvid_format"


PROTO_TYPE_SPEC = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "next_observations/images0": tf.uint8,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
        "language": tf.string, 
    }



def load_trajectories(file_path):
    trajectories = []

    loaded_data = load_tfrecord_file(file_path)

    for i, record in enumerate(loaded_data):
        # Access your features here, e.g., record['feature1'], record['feature2']

        image = record["images"]
        proprio = record["proprio"]
        language = record["language"]
        actions = record["actions"]
        terminals = record["terminals"]

        image = np.array(image)
        proprio = np.array(proprio)
        language = np.array(language).item().decode("utf-8").replace(",", "").strip("\n\t ") # have to get rid of extra commas
        actions = np.array(actions)
        terminals = np.array(terminals)

        trajectories.append({"frames":image, "language":language})

    return trajectories

def process_bridge_data():

    videoid = 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_file = open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w")
    metadata_file.write("page_dir,videoid,name\n")

    filepaths = list(glob(os.path.join(INPUT_DIR, "**", "*.tfrecord"), recursive=True))
    

    train_filepaths = list(glob(os.path.join(INPUT_DIR, "**", "train/*.tfrecord"), recursive=True))
    train_filepaths = sorted(train_filepaths)
    val_filepaths = list(glob(os.path.join(INPUT_DIR, "**", "val/*.tfrecord"), recursive=True))
    val_filepaths = sorted(val_filepaths)

    assert len(train_filepaths) + len(val_filepaths) == len(filepaths)

    n_skipped_no_language = 0

    for file_path in tqdm(filepaths):   
        trajectories = load_trajectories(file_path)
        for traj in trajectories:
            frames = traj["frames"]
            language = traj["language"] 
            page = file_path.split("bridgev2_processed")[-1].strip("/").split("/")[0]
            video_path = os.path.join(OUTPUT_DIR, "videos", page, f"{videoid}.mp4")
            save_video(video_path, frames, resize=(256, 256))

            metadata_file.write(f"{page},{videoid},{language}\n")

            videoid += 1

    metadata_file.close()

    print(f"Processed {videoid - 1} total train trajectories.")
    print("Number of trajectories skipped because there is no language annotation:", n_skipped_no_language)


def load_tfrecord_file(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = dataset.map(_decode_example)
    return parsed_dataset

def _decode_example(example_proto, load_language=True):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {
        key: tf.io.parse_tensor(parsed_features[key], dtype)
        for key, dtype in PROTO_TYPE_SPEC.items()
    }
    # restructure the dictionary into the downstream format
    return {
        "images":parsed_tensors["observations/images0"], 
        "proprio": parsed_tensors["observations/state"],
        **({"language": parsed_tensors["language"][0]} if load_language else {}), 
        "actions": parsed_tensors["actions"],
        "terminals": tf.expand_dims(parsed_tensors["terminals"], axis=-1), 
    }

def save_video(output_file, frames, fps=30, codec='libx264', resize=None):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    # Create the video writer object
    with imageio.get_writer(output_file, fps=fps, codec=codec) as writer:
        for frame in frames:
            if resize is not None:
                frame = cv2.resize(frame, (resize))
            writer.append_data(frame)


def save_video_cv2(output_video_file, frames):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

if __name__ == "__main__":
    process_bridge_data()