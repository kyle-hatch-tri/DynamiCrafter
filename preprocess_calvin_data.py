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


INPUT_DIR = "/<path_to_data>/calvin_data_processed/language_conditioned/training/"
TRAIN_SCENES = ["A", "B", "C"]
OUTPUT_DIR = "/<path_to_data>/calvin_webvid_format"

PROTO_TYPE_SPEC = {
        "actions": tf.float32,
        "proprioceptive_states": tf.float32,
        "image_states": tf.uint8,
        "language_annotation": tf.string, 
    }




def load_trajectory(file_path):
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

        assert i < 1, f"i: {i}"

    return image, language

def process_calvin_data():

    videoid = 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata_file = open(os.path.join(OUTPUT_DIR, "metadata.csv"), "w")
    metadata_file.write("page_dir,videoid,name\n")

    get_scene_from_path = lambda x:x.split("/")[-2]
    get_traj_no_from_path = lambda x:int(x.split("/")[-1].split(".")[0].replace("traj", ""))
    

    filepaths = []
    for scene in TRAIN_SCENES:
        scene_paths = glob(os.path.join(INPUT_DIR, scene, "traj*.tfrecord"))
        if len(scene_paths) > 0:
            scene_paths = sorted(scene_paths, key=get_traj_no_from_path)
            filepaths += scene_paths


    for file_path in tqdm(filepaths):
        
        frames, language = load_trajectory(file_path)

        page = get_scene_from_path(file_path) # page = suite number 
        assert page != "D", f"file_path: {file_path}"

        video_path = os.path.join(OUTPUT_DIR, "videos", page, f"{videoid}.mp4")
        save_video(video_path, frames, resize=(256, 256))

        metadata_file.write(f"{page},{videoid},{language}\n")

        videoid += 1

    

    metadata_file.close()

    print(f"Processed {videoid - 1} total trajectories.")
    

def load_tfrecord_file(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = dataset.map(_decode_example)
    return parsed_dataset

def _decode_example(example_proto):
    # decode the example proto according to PROTO_TYPE_SPEC
    features = {
        key: tf.io.FixedLenFeature([], tf.string)
        for key in PROTO_TYPE_SPEC.keys()
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    parsed_tensors = {}
    for key, dtype in PROTO_TYPE_SPEC.items():
        if dtype == tf.string:
            parsed_tensors[key] = parsed_features[key]
        else:
            parsed_tensors[key] = tf.io.parse_tensor(parsed_features[key], dtype)
    # restructure the dictionary into the downstream format
    return {
        "images":parsed_tensors["image_states"],
        "proprio": parsed_tensors["proprioceptive_states"],
        **({"language": parsed_tensors["language_annotation"]} if True else {}),
        "actions": parsed_tensors["actions"],
        "terminals": tf.zeros_like(parsed_tensors["actions"][:, 0:1], dtype=tf.bool)
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
    process_calvin_data()