import asyncio
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image

SAM2APIOutput: TypeAlias = tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]

THREAD_POOL = 64

# Example input data
image_pils = [
    Image.open("example_data/image_1.jpg"),
    Image.open("example_data/image_2.jpg"),
    Image.open("example_data/image_3.jpg"),
]
point_coords = [
    [[10, 20], [30, 40]],  # Points for image 1
    [[50, 60], [70, 80]],  # Points for image 2
    [[15, 25], [35, 45]],  # Points for image 3
]
bbox_xyxys = [
    (0, 0, 100, 100),  # Bounding box for image 1
    (10, 10, 200, 200),  # Bounding box for image 2
    (20, 20, 150, 150),  # Bounding box for image 3
]

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
    image_buffer = io.BytesIO()
    image_pil.save(image_buffer, format="JPEG")
    image_buffer.seek(0)  # Move to the start of the file-like object.

    # Define the API endpoint.
    url = "https://dream-gateway.livepeer.cloud/segment-anything-2"

    # Prepare the payload for the request.
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

    # Since you already have the image in memory, prepare the files dict
    files = {"image": ("test.jpeg", image_buffer, "image/jpeg")}

    response = session.post(url, headers=headers, data=data, files=files, timeout=120)
    response_data = response.json()

    masks = np.array(json.loads(response_data["masks"]))
    logits = np.array(json.loads(response_data["logits"]))
    scores = np.array(json.loads(response_data["scores"]))

    return masks, logits, scores


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


async def main():
    results = await detect_masks_sam2_async(image_pils, point_coords, bbox_xyxys)
    for masks, logits, scores in results:
        print("Masks:", masks)
        print("Logits:", logits)
        print("Scores:", scores)


# Run the async function.
asyncio.run(main())
