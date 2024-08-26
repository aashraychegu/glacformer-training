# Create the argument parser
import argparse
import pathlib
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import evaluate
import huggingface_hub
import albumentations as A
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import (
    SegformerImageProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

# Get the path of the parent directory for this file
parent_dir = pathlib.Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load_from",
    type=str,
    help="Path to load the pre-trained model from",
    default=parent_dir / "glacformer",
)
parser.add_argument(
    "--token",
    type=str,
    help="The throwaway auth token",
    default="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",
)
parser.add_argument(
    "--save_to",
    type=int,
    help="where to save the model to once training is over",
    default="@source",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    help="The initial learning rate for Adam",
    default=6e-5,
)
parser.add_argument(
    "--num_epochs",
    type=int,
    help="Total number of training epochs to perform",
    default=1,
)
args = parser.parse_args()
load_from = args.load_from
token = args.token
save_to = args.save_to
learning_rate = args.learning_rate
num_epochs = args.num_epochs


torch.backends.cuda.matmul.allow_tf32 = True


hf_model_name = "glacierscopessegmentation/glacier_segmentation_transformer"
huggingface_hub.login(token=token)
ds = load_dataset(
    "glacierscopessegmentation/scopes",
    keep_in_memory=True,
)
train_ds = ds["train"].shuffle(seed=42)
test_ds = ds["test"]
id2label = {
    "0": "sky",  # This is given by the rgb value of 00 00 00 for the mask
    "1": "surface-to-bed",  # This is given by the rgb value of 01 01 01 for the mask
    "2": "bed-to-bottom",  # This is given by the rgb value of 02 02 02 for the mask
}
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
len(train_ds), len(test_ds)

test_image_processor = SegformerImageProcessor.from_pretrained("nvidia/MiT-b0")

if load_from == "new":
    test_config = SegformerConfig(
        num_channels=3,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        depths=[2, 3, 4, 3],
        hidden_sizes=[64, 128, 320, 512],
        decoder_hidden_size=256 * 3,
    )
    testmodel = SegformerForSemanticSegmentation(test_config)
else:
    testmodel = SegformerForSemanticSegmentation.from_pretrained(
        load_from,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True,
    )

# Load the image processor for the test model from the pre-trained checkpoint


transform = A.Compose(
    [
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.Perspective(p=0.3),
    ],
    additional_targets={"mask": "mask"},
)

# creates transforms for data augumentation, and using albumentations allows me apply the same transform to the image and the mask at the same time


# Define a function to apply transformations to a batch of training examples
def train_transforms(example_batch):
    imagesandmasks = [
        transform(image=np.array(image.convert("RGB")), mask=np.array(mask))
        for image, mask in zip(example_batch["image"], example_batch["label"])
    ]
    # applies the transform to the image and mask, but the data is stored as a list of dictionaries, so the next lines separate out the dicts into 2 different lists
    images = [i["image"] for i in imagesandmasks]
    masks = [i["mask"] for i in imagesandmasks]
    inputs = test_image_processor(images, masks)
    return inputs


# Define a function to apply transformations to a batch of validation examples
def val_transforms(example_batch):
    # Convert each image in the batch to RGB
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [x for x in example_batch["label"]]
    inputs = test_image_processor(images, labels)
    return inputs


# this makes the transforms happen when a batch is loaded
train_ds.set_transform(train_transforms)
test_ds.set_transform(val_transforms)


# Load the "mean_iou" metric for evaluating semantic segmentation models
metric = evaluate.load("mean_iou")


# Define a function to compute metrics for evaluation predictions
# Here, the metric is mean intersection over union
def compute_metrics(eval_pred):
    # Ensure that gradient computation is turned off, as it is not needed for evaluation
    with torch.no_grad():
        # This computes the final logits tensor by interpolating the output logits to the size of the labels tensor from an input of size (batch_size, num_labels, height, width)
        # This is input that has gone through the model's forward pass
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits).cpu()

        logits_tensor = logits_tensor.argmax(dim=1)
        logits_tensor = logits_tensor.unsqueeze(1).to(float).detach().cpu().numpy()
        # this can lead to very high ram usage for the upscaling
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        logits_tensor = torch.squeeze(logits_tensor, dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        # Computes metrics
        metrics = metric.compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            reduce_labels=False,
            ignore_index=255,
        )
        for key, value in metrics.items():
            if type(value) is np.ndarray:
                metrics[key] = value.tolist()
        # Return the computed metrics
        return metrics


# Define the training arguments
training_args = TrainingArguments(
    output_dir="glacformer",  # The output directory for the model predictions and checkpoints
    learning_rate=learning_rate,  # The initial learning rate for Adam
    num_train_epochs=num_epochs,  # Total number of training epochs to perform
    auto_find_batch_size=True,  # Whether to automatically find an appropriate batch size
    save_total_limit=3,  # Limit the total amount of checkpoints and delete the older checkpoints
    # eval_accumulation_steps=1,  # Number of steps to accumulate gradients before performing a backward/update pass
    evaluation_strategy="epoch",  # The evaluation strategy to adopt during training
    save_strategy="epoch",  # The checkpoint save strategy to adopt during training
    save_steps=1,  # Number of updates steps before two checkpoint saves
    eval_steps=1,  # Number of update steps before two evaluations
    # logging_steps=30,  # Number of update steps before logging learning rate and other metrics
    remove_unused_columns=False,  # Whether to remove columns not used by the model when using a dataset
    fp16=True,  # Whether to use 16-bit float precision instead of 32-bit for saving memory
    tf32=True,  # Whether to use tf32 precision instead of 32-bit for saving memory
    gradient_accumulation_steps=4,  # Number of updates steps to accumulate before performing a backward/update pass for saving memory
    hub_model_id=hf_model_name,  # The model ID on the Hugging Face model hub
)

# Define the trainer
trainer = Trainer(
    model=testmodel,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()

if save_to == "@source":
    save_to = load_from
trainer.model.save_pretrained(save_to)
