import asyncio
import io
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

SAM2APIOutput: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]
# GATEWAY_URL = "https://dream-gateway.livepeer.cloud"
GATEWAY_URL = "http://0.0.0.0:8935"
THREAD_POOL = 10

# Initialize session.
session = requests.Session()
session.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        pool_maxsize=THREAD_POOL,
        max_retries=3,
        pool_block=True,
    ),
)


def detect_masks(
    image_pil: Image.Image,
    point_coords: list[list[int]],
    bbox_xyxys: tuple[int, int, int, int],
    session: requests.Session,
) -> SAM2APIOutput:
    """
    Detects faces in an image using Amazon Rekognition.
    """
    try:
        image_buffer = io.BytesIO()
        image_pil.save(image_buffer, format="JPEG")
        image_buffer.seek(0)  # Move to the start of the file-like object.

        # Define the API endpoint.
        url = f"{GATEWAY_URL}/segment-anything-2"

        # Prepare the payload for the request
        headers = {
            "Authorization": "Bearer 50D28E87-3902-49EC-8DF0-0EA5A93EB62F",
        }

        data = {
            "model_id": "facebook/sam2-hiera-large",
            "point_coords": str(point_coords),
            "point_labels": str(
                [1 for _ in point_coords]
            ),  # point_labels should be inside bounding box
            "box": str([np.array(bbox_xyxys).tolist()]),
            "multimask_output": "True",
        }

        # Since you already have the image in memory, prepare the files dict.
        files = {"image": ("test.jpeg", image_buffer, "image/jpeg")}

        response = session.post(
            url, headers=headers, data=data, files=files, timeout=120
        )

        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            print(f"Response content: {response.text}")
            # Return a default value or None to indicate failure
            return None, None, None

        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON response: {e}")
            print(f"Response content: {response.text}")
            raise

        masks = np.array(json.loads(response_data["masks"]))
        logits = np.array(json.loads(response_data["logits"]))
        scores = np.array(json.loads(response_data["scores"]))

        return masks, logits, scores

    except Exception as e:
        print(f"An error occurred: {e}")
        # Return a default value or None to indicate failure
        return None, None, None


async def detect_masks_sam2_async(
    image_pils: list[Image.Image],
    point_coords: list[list[list[int]]],
    bbox_xyxys: list[tuple[int, int, int, int]],
) -> list[SAM2APIOutput]:
    """
    Detect masks features in a list of images using LivePeer SAM2 API.
    """
    if not (len(image_pils) == len(point_coords) == len(bbox_xyxys)):
        raise ValueError("Length of inputs must be the same!")

    with ThreadPoolExecutor(max_workers=THREAD_POOL) as executor:
        # Initialize the event loop
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor, detect_masks, *(image_pil, point_coord, bbox_xyxy, session)
            )
            for image_pil, point_coord, bbox_xyxy in zip(
                image_pils, point_coords, bbox_xyxys
            )
        ]

        results = await asyncio.gather(*tasks)
        return results


def load_bounding_boxes(pkl_file_path):
    """
    Load bounding boxes from a pickle file.

    Args:
        pkl_file_path (str): Path to the pickle file.

    Returns:
        list: List of bounding boxes.
    """
    with open(pkl_file_path, "rb") as file:
        return pickle.load(file)


def load_point_coords(pkl_file_path):
    """
    Load point coordinates from a pickle file.

    Args:
        pkl_file_path (str): Path to the pickle file.

    Returns:
        list: List of point coordinates.
    """
    with open(pkl_file_path, "rb") as file:
        return pickle.load(file)


async def main():
    """
    Run object segmentation using Livepeer AI SAM2 pipeline synchronously.
    """
    # Retrieve all image files from the example_data/images folder.
    image_files = glob(os.path.join("example_data", "images", "*.jpg"))
    image_pils = [Image.open(image_file) for image_file in image_files]

    # Load point coords and bounding boxes from the pickle file.
    point_coords = load_point_coords("example_data/point_coords_snippet.pkl")
    bbox_xyxys = load_bounding_boxes("example_data/bbox_snippet.pkl")

    # Run the detection.
    results = await detect_masks_sam2_async(image_pils, point_coords, bbox_xyxys)
    for masks, logits, scores in results:
        print("Masks:", masks)
        print("Logits:", logits)
        print("Scores:", scores)


# Run the async function.
asyncio.run(main())
