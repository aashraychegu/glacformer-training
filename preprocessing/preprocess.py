import pathlib as pl
import argparse
import pprint
from PIL import Image
import pathlib as pl
import numpy as np
import pandas as pd
from PIL import ImageDraw
import itertools
from set_offset import set_offset
import cv2
import numpy as np
from datasets import Dataset
from datasets import Image as DSImage
import shutil
import os
import warnings
import time
import shutil


# Disable the decompression bomb warning you get when opening up the .tiff file
warnings.filterwarnings("ignore")

# Helper methods


# Replaces one color with another in an image
def replace_color_numpy(image, orig_color, replacement_color):
    data = np.array(image.convert("RGB"))
    data[(data == orig_color).all(axis=-1)] = replacement_color
    return Image.fromarray(data, mode="RGB")


# helps visualize the pixels of an image easier
def visualize_pixels(image):
    image = replace_color_numpy(image, [0, 0, 0], [255, 0, 0])
    image = replace_color_numpy(image, [1, 1, 1], [0, 255, 0])
    image = replace_color_numpy(image, [2, 2, 2], [0, 0, 255])

    return image


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
    return image
    # Read the image

    # Convert the image to grayscale
    gray = cv2.cvtColor(convert_from_image_to_cv2(image), cv2.COLOR_BGR2GRAY)

    combined_image = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    green = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_32F, 7))
    blue = cv2.convertScaleAbs(cv2.Scharr(blurred, cv2.CV_32F, 0, 1))

    combined_image[:, :, 0] = gray
    combined_image[:, :, 1] = np.clip(np.round(green + gray), 0, 255)
    blend = 0
    combined_image[:, :, 2] = np.clip(
        np.round(blue * blend + gray * (1 - blend)), 0, 255
    )

    return convert_from_cv2_to_image(combined_image)


# Combines two images side by side for easier viewing
def combine_images_side_by_side(image1, image2):
    # Get the dimensions of the images
    width1, height1 = image1.size
    width2, height2 = image2.size
    # Create a new image with a width that is the sum of both images' widths and the height of the tallest image
    new_image = Image.new("RGB", (width1 + width2, max(height1, height2)))
    # Paste the images into the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    return new_image


# Converts a csv file of glacier surface coordinates and glacial bed coordinates to an image mask the same size as the input image
# uses an offset to correct for issues with the data
def csv_to_mask(img, raw_data, offset):
    # adds the offset to fix the mask
    csv = raw_data[["x_surface", "y_surface", "x_bed", "y_bed"]] + offset
    # flips the csv file to ensure the mask is the correct way up
    csv = csv[::-1].reset_index(drop=True)
    # adds the start and end points to the csv file to ensure the mask stretches from the edges of the images
    top = pd.DataFrame(
        {
            "x_surface": 0,
            "y_surface": csv.iloc[0]["y_surface"],
            "x_bed": 0,
            "y_bed": csv.iloc[0]["y_bed"],
        },
        index=[0],
    )
    bottom = pd.DataFrame(
        {
            "x_surface": img.size[0],
            "y_surface": csv.iloc[-1]["y_surface"],
            "x_bed": img.size[0],
            "y_bed": csv.iloc[-1]["y_bed"],
        },
        index=[0],
    )
    csv = pd.concat([top, csv, bottom], ignore_index=True)

    # Create a draw object for the image for drawing the polygons
    mask_base = img.copy()
    draw = ImageDraw.Draw(mask_base)

    # Loop over the rows of the csv file
    for i in range(len(csv) - 1):
        # interpolates between the current row and the next row for the edges of the polygons
        crow = csv.iloc[i]
        nrow = csv.iloc[i + 1]

        # Define the coordinates for the sky, bed, and bottom polygons
        skycooords = [
            (crow["x_surface"], 0),
            (nrow["x_surface"], 0),
            (nrow["x_surface"], nrow["y_surface"]),
            (crow["x_surface"], crow["y_surface"]),
        ]
        bedcoords = [
            (crow["x_surface"], crow["y_surface"]),
            (nrow["x_surface"], nrow["y_surface"]),
            (nrow["x_bed"], nrow["y_bed"]),
            (crow["x_bed"], crow["y_bed"]),
        ]
        btmcoords = [
            (crow["x_bed"], crow["y_bed"]),
            (nrow["x_bed"], nrow["y_bed"]),
            (nrow["x_bed"], img.size[1]),
            (crow["x_bed"], img.size[1]),
        ]
        draw.polygon(skycooords, fill="#000000")
        draw.polygon(bedcoords, fill="#010101")
        draw.polygon(btmcoords, fill="#020202")
    del draw
    return mask_base


# adds command line arguments for the script
p = pprint.PrettyPrinter(indent=4)
parser = argparse.ArgumentParser(description="Preprocessing script")
parser.add_argument(
    "--raw_data_directories_file",
    type=str,
    help="Path to the file containing the raw data directories",
    default="@NA",
)
parser.add_argument(
    "--raw_data_folder", type=str, help="Path to the raw data folder", default=""
)
parser.add_argument(
    "--images_folder", type=str, help="Path to the temporary images folder"
)
parser.add_argument(
    "--token",
    type=str,
    help="The auth token for pushing the dataset",
    default="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",
)
parser.add_argument(
    "--overwrite_offsets", type=bool, help="Overwrite offsets.csv file", default=False
)
parser.add_argument(
    "--skip_existing", type=bool, help="Skip existing folders", default=True
)
parser.add_argument(
    "--dataset_folder",
    type=str,
    help="Path to the dataset folder",
    default=pl.Path(__file__).parent.parent / "dataset",
)
parser.add_argument(
    "--offsets_file",
    type=str,
    help="Path to the offsets_file",
    default=pl.Path(__file__).parent / "offsets.csv",
)
parser.add_argument(
    "--delete_old_datasets", type=bool, help="Delete old offsets", default=False
)
parser.add_argument("--debug", type=bool, help="Debug mode", default=False)
parser.add_argument("--use_new_data", type=bool, help="Use new data", default=True)
parser.add_argument("--push_to_hub", type=bool, help="Push to hub", default=True)
args = parser.parse_args()

# Check if the raw data directories file is provided
# If it is, read the file and move all the csvs and tiffs to a new directory
if args.raw_data_directories_file != "@NA":
    with open(args.raw_data_directories_file, "r") as f:
        raw_data_directories = f.read().splitlines()
    raw_data_directories = [pl.Path(x) for x in raw_data_directories]
    collated_directory = pl.Path(args.raw_data_directories).parent / "raw_combined_data"
    collated_directory.mkdir(parents=True, exist_ok=True)
    collated_csvs = collated_directory / "csvs"
    collated_tiffs = collated_directory / "tiffs"
    collated_csvs.mkdir(parents=True, exist_ok=True)
    collated_tiffs.mkdir(parents=True, exist_ok=True)
    for raw_data_directory in raw_data_directories:
        shutil.move((raw_data_directory / "csvs").glob("*"), collated_csvs)
        shutil.move((raw_data_directory / "tiffs").glob("*"), collated_tiffs)
    args.raw_data_folder = collated_directory
else:
    if args.raw_data_folder == "":
        raise ValueError("Please provide the path to the raw data folder")

# converts given paths to pathlib objects for easier use
raw_data_folder = pl.Path(args.raw_data_folder)
images_folder = pl.Path(args.images_folder)
images_folder.mkdir(parents=True, exist_ok=True)
dataset_folder = pl.Path(args.dataset_folder)
dataset_folder.mkdir(parents=True, exist_ok=True)

# gets all proided csvs and images
csvs = list((raw_data_folder / "csvs").glob("*.csv"))
imgs = list((raw_data_folder / "tiffs").glob("*.tiff"))

print(f"Found {len(csvs)} CSV files and {len(imgs)} images")

# creates a dictionary of the csvs and images with the name of the file, and the path to the file
clean_csv_names = {}
for x in csvs:
    clean_csv_names[str(x.resolve()).split("\\")[-1].replace(".csv", "")] = x

clean_img_names = {}
for x in imgs:
    clean_img_names[str(x.resolve()).split("\\")[-1].replace(".tiff", "")] = x

# finds the common names between the csvs and images
matched_pairs = sorted(
    list(set(clean_csv_names.keys()).intersection(set(clean_img_names.keys())))
)
p.pprint(matched_pairs)
print(f"Found {len(matched_pairs)} pairs of images and CSV files")


# creates a new offsets file
def new_offsets_file(offsets_file):
    offsets_data = pd.DataFrame(columns=["Folder_Name", "Offset"])
    offsets_data["Folder_Name"] = matched_pairs
    offsets_data["Offset"] = list(itertools.repeat(-1, len(matched_pairs)))
    offsets_data.to_csv(offsets_file, index=False)
    print(f"Created offsets.csv in {images_folder}")


# creates offsets.csv if it doesn't exist or if the user wants to overwrite it
offsets_file = pl.Path(__file__).parent / "offsets.csv"
if not offsets_file.exists() or args.overwrite_offsets:
    new_offsets_file(offsets_file)
    print(f"Created offsets.csv in {images_folder}")
else:
    offsets_data = pd.read_csv(offsets_file)
    print(
        f"reading existing offsets.csv in {images_folder/'offsets.csv'} with length {len(offsets_data)}"
    )

    existing_offsets = offsets_data[offsets_data["Offset"] != -1]

    if (len(existing_offsets) != len(matched_pairs)) or args.use_new_data:

        new_offsets_data = pd.DataFrame(columns=["Folder_Name", "Offset"])
        new_offsets_data["Folder_Name"] = matched_pairs
        new_offsets_data["Offset"] = list(itertools.repeat(-1, len(matched_pairs)))
        # Helps merge existing offsets with any new data
        new_offsets_data.loc[
            new_offsets_data["Folder_Name"].isin(existing_offsets["Folder_Name"]),
            "Offset",
        ] = offsets_data.loc[
            offsets_data["Folder_Name"].isin(new_offsets_data["Folder_Name"]), "Offset"
        ]
        new_offsets_data["Offset"] = new_offsets_data["Offset"].fillna(-1)
        new_offsets_data.to_csv(offsets_file, index=False)
        print(f"Combining offsets.csv with new data")
        offsets_data = new_offsets_data.fillna(-1)

# helps store image and mask paths for creating the dataset
image_paths = []
mask_paths = []

# loops over all the matched pairs of images and csvs
for count, matched_pair in enumerate(matched_pairs):
    pair_folder = images_folder / matched_pair
    # reads offset
    curr_offset = offsets_data[offsets_data["Folder_Name"] == matched_pair][
        "Offset"
    ].values[0]
    # skips the folder if it already exists and the user wants to skip existing folders
    if (pair_folder.exists() and args.skip_existing) and not (
        curr_offset == -1 or curr_offset == float("nan")
    ):
        print(f"Adding all files from {matched_pair}")
        chunked_image_folder = pair_folder / "chunked_images"
        chunked_mask_folder = pair_folder / "chunked_masks"
        image_paths.extend(
            [str(x.resolve()) for x in chunked_image_folder.glob("*.png")]
        )
        mask_paths.extend([str(x.resolve()) for x in chunked_mask_folder.glob("*.png")])
        continue
    # reads the raw image and csv
    img_source = clean_img_names[matched_pair]
    csv_source = clean_csv_names[matched_pair]
    raw_img = Image.open(img_source)
    raw_data = pd.read_csv(csv_source)

    # crops the image to the relevant section
    img = raw_img.copy().crop((0, 430, raw_img.size[0], 1790))
    img = Image.fromarray(np.divide(np.array(img), 2**8 - 1)).convert("L")

    # sets the offset if it doesn't exist
    if curr_offset == -1 or curr_offset == float("nan"):
        # generates the mask for setting the offset
        mask_base = csv_to_mask(img, raw_data, 0)
        print(
            f"Processing {matched_pair} | {count+1} / {len(matched_pairs)} with the default offset of -1"
        )
        offset = int(
            set_offset(
                img.crop((0, 0, 500, img.size[1])),
                visualize_pixels(mask_base.crop((0, 0, 500, mask_base.size[1]))),
                os.getpid(),
            )
        )
        offsets_data.loc[offsets_data["Folder_Name"] == matched_pair, "Offset"] = offset
        offsets_data.to_csv(offsets_file, index=False)

    else:
        offset = offsets_data[offsets_data["Folder_Name"] == matched_pair][
            "Offset"
        ].values[0]
    print(
        f"Processing {matched_pair} {count+1} / {len(matched_pairs)} with offset {offset}"
    )

    # generates the mask
    mask = csv_to_mask(img, raw_data, offset)
    # creates the folder for the image/csv pair
    pair_folder.mkdir(parents=True, exist_ok=True)
    img = canny_image(img)
    print("Mask Generation and Canny Processing complete. \n Chunking Started")

    # can show the generated mask and image for debugging purposes
    if args.debug:
        combine_images_side_by_side(
            img.crop((0, 0, 750, img.size[1])),
            visualize_pixels(mask.crop((0, 0, 750, mask.size[1]))).transpose(
                Image.FLIP_LEFT_RIGHT
            ),
        ).show()

    # starts cropping the image

    # gets coordinates for cropping the image
    cropping_coordinates = window_with_remainder(512, 128, mask.size[0])

    # creates folder for the saved chunks
    chunked_image_folder = pair_folder / "chunked_images"
    chunked_mask_folder = pair_folder / "chunked_masks"
    chunked_image_folder.mkdir(parents=True, exist_ok=True)
    chunked_mask_folder.mkdir(parents=True, exist_ok=True)

    # loops over the cropping coordinates and saves the cropped chunks
    for count, (start, end) in enumerate(cropping_coordinates):
        current_chunk_path = chunked_image_folder / f"cimg_{count}_{start}_{end}.png"
        current_mask_path = chunked_mask_folder / f"cmask_{count}_{start}_{end}.png"
        image_paths.append(str(current_chunk_path.resolve()))
        mask_paths.append(str(current_mask_path.resolve()))
        resized = img.crop((start, 0, end, img.size[1])).resize((512, 512))
        resized.save(current_chunk_path)
        mask.crop((start, 0, end, mask.size[1])).resize((512, 512)).point(
            lambda p: min(p, 2)
        ).save(current_mask_path)
        print(
            f"{count+1}/{len(cropping_coordinates)} | {matched_pair.split('-')[0]} | {start}-{end}",
            end="\r",
        )
    print("\nFinished", "\n" * 3)

# Define a function to create a dataset from image and label paths
# Create a Dataset object from a dictionary of image and label paths
print(f"{len(image_paths)} images and {len(mask_paths)} masks")
dataset = Dataset.from_dict({"image": sorted(image_paths), "label": sorted(mask_paths)})
dataset = dataset.cast_column("image", DSImage())
dataset = dataset.cast_column("label", DSImage())
dataset_save_path = dataset_folder / time.strftime(r"%Y_%m_%d-%H_%M_%S")
# prints dataset information
print(dataset)
print(dataset.info)
print(dataset.features)

# can delete old datasets
if args.delete_old_datasets:
    shutil.rmtree(dataset_folder)

# ensures there is a dataset folder
dataset_folder.mkdir(parents=True, exist_ok=True)
dataset.save_to_disk(dataset_save_path)

# pushes the dataset to the hub if the user wants
if args.push_to_hub or not args.token:
    dataset.push_to_hub("glacierscopessegmentation/scopes_test", token=args.token)
