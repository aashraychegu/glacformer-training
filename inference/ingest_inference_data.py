import pathlib as pl
from PIL import Image
import datasets
from datasets import Dataset
from datasets import Image as DSImage
import numpy as np
import cv2
import matplotlib.pyplot as plt
# files = list((pl.Path(__file__).parent / "inference_inputs").glob("*"))
files = list((pl.Path(".").parent / "inference_inputs").glob("*"))
print(files)
# exit()

def window_with_remainder(length, overlap, input_size):
    testarray = np.arange(0, input_size)
    return np.vstack(
        (
            testarray[0:length],
            np.lib.stride_tricks.sliding_window_view(
                testarray[len(testarray) % length :], length
            )[::overlap],
        )
    )[:, [0, -1]] + [0, 1]
# Converts an image from cv2 to PIL
def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# Converts an image from PIL to cv2
def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(src=np.array(img, dtype=np.uint8), code=cv2.COLOR_RGB2BGR)
# Splits an longer image into a series of smaller images with a given overlap
# to reduce memory costs it takes in the length, input size and returns a list of valid indices
def window_with_remainder(length, overlap, input_size):
    testarray = np.arange(0, input_size)
    return np.vstack(
        (
            testarray[0:length],
            np.lib.stride_tricks.sliding_window_view(
                testarray[len(testarray) % length :], length
            )[::overlap],
        )
    )[:, [0, -1]] + [0, 1]
# does canny edge detection on an image
def canny_image(image: Image):
    return PIL.ImageOps.autocontrast(PIL.ImageOps.equalize(image))

image_paths = []
def ingest_inference_data():
    for c, i in enumerate(files):
        img = Image.open(i)
        img = img.crop((0, 430, img.size[0], 1790))
        img = Image.fromarray(np.array(img) / 255)
        img = img.convert("L")
        cropping_coordinates = window_with_remainder(512, 128, img.size[0])
        file_path = pl.Path("./inference_raw_imgs") / i.stem
        file_path.mkdir(exist_ok=True)
        for count, (start, end) in enumerate(cropping_coordinates):
            save_path = file_path / f"{count}-{start}-{end}.png"
            cimg = img.crop((start, 0, end, img.size[1])).resize((512, 512))
            resized = canny_image(cimg)
            resized.save(save_path)    
            print(f"finished {save_path}")
            image_paths.append(str(save_path.resolve()))
            # break
        print("\n--\n")

    dataset = Dataset.from_dict({"image": sorted(image_paths)})
    dataset = dataset.cast_column("image", DSImage())
    # prints dataset information
    print(dataset)
    print(dataset.info)
    print(dataset.features)
    dataset.save_to_disk("./inference_dataset/inference_dataset.hf")
    return dataset

if __name__ == "__main__":
    ingest_inference_data()