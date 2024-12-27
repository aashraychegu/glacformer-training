# Create the argument parser

def main():
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
    import pathlib as pl
    import datetime
    import os
    import accelerate
    import shutil
    import torch.nn.functional as F
    os.environ["WANDB_PROJECT"] = "glacformer_training" 
    #os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_DEBUG"] = "INFO"
    torch.backends.cuda.matmul.allow_tf32 = True
    parent_dir = pathlib.Path(__file__).resolve().parent


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--continue_training",
        type=bool,
        help="continues training from the last checkpoint",
        default=False,
    )

    parser.add_argument(
        "--load_from",
        type=str,
        help="load a new model or from a checkpoint or a path. if 'new' or a path, set continue_training to false",
        default="new",
    )

    parser.add_argument(
        "--token",
        type=str,
        help="The throwaway auth token",
        default="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",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The initial learning rate for Adam",
        default=18e-5,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Total number of training epochs to perform",
        default=1,
    )

    args = parser.parse_args()
    token = args.token
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    load_from = args.load_from

    if args.continue_training == "True":
        load_from = "checkpoint"
        print("Continuing training from checkpoint")
    else:
        print("Starting new training")
        
    hf_model_name = "glacierscopessegmentation/glacier_segmentation_transformer"
    # huggingface_hub.login(token=token,add_to_git_credential=True)
    print(huggingface_hub.auth_list())

    ds = load_dataset("glacierscopessegmentation/scopes", keep_in_memory=True, cache_dir=pl.Path(__file__).parent / "data")
    train_ds = ds["train"]
    test_ds = ds["test"]
    ds.cleanup_cache_files()
    del ds

    id2label = {
        "0": "sky",  # This is given by the rgb value of 00 00 00 for the mask
        "1": "surface-to-bed",  # This is given by the rgb value of 01 01 01 for the mask
        "2": "bed-to-bottom",  # This is given by the rgb value of 02 02 02 for the mask
    }

    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)
    print(len(train_ds), len(test_ds))
    train_ds.cleanup_cache_files()
    test_ds.cleanup_cache_files()
    test_image_processor = SegformerImageProcessor.from_pretrained("nvidia/MiT-b0")

    if load_from == "new" or load_from == "checkpoint":
        test_config = SegformerConfig(
            num_channels=3,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            depths=[3, 4, 6, 3],
            hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=128 * 6,
        )
        testmodel = SegformerForSemanticSegmentation(test_config)
    else:
        testmodel = SegformerForSemanticSegmentation.from_pretrained(
            load_from,
            id2label=id2label,
            label2id=label2id,
            local_files_only=True,
        )

    transform = A.Compose(
        [
            A.GridDistortion(p=0.75, distort_limit=1, num_steps=10),
            A.RandomBrightnessContrast(p=0.8,brightness_limit=.5,contrast_limit=.5,ensure_safe_range=True),
            A.RandomToneCurve(p=0.8,scale=.8),
            A.RandomResizedCrop(p=.8,size=[512,512],scale=[.4,1],ratio=[0.05,100])
        ],
        additional_targets={"mask": "mask"},
    )

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
    train_ds = train_ds.with_transform(train_transforms)
    test_ds = test_ds.with_transform(val_transforms)

    # Load the "mean_iou" metric for evaluating semantic segmentation models
    metric = evaluate.load("mean_iou",cache_dir=pl.Path(__file__).parent / "eval_cache")

    # Define a function to compute metrics for evaluation predictions
    # Here, the metric is mean intersection over union
    # from memory_profiler import profile 
    # @profile
    def compute_metrics(eval_pred, compute_result=True):
        with torch.no_grad():
            logits, labels = eval_pred
            pred_labels = F.interpolate(
                logits.cpu().argmax(dim=1).unsqueeze(1).float(),
                size=labels.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1).cpu()
            
            if not compute_result:
                return {"predictions": pred_labels, "labels": labels}
            
            metrics = metric.compute(
                predictions=pred_labels,
                references=labels,
                num_labels=num_labels,
                reduce_labels=False,
                ignore_index=255,
            )
            
            m2 = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    for c, v in enumerate(list(value)):
                        m2[key + "_" + id2label[c]] = v
                else:
                    m2[key] = value
            
            del metrics
            return m2

    # Define the training arguments
    if load_from == "checkpoint":
        run_name = sorted(list((parent_dir.parent / "glacformer").glob("*")))[-1].name
    else: run_name = datetime.datetime.now().strftime("SherlockCluster--%Y-%m-%d--%H-%M-%S-%Z")

    training_args = TrainingArguments(
        output_dir = "glacformer/" + run_name,
        num_train_epochs=num_epochs,  # Total number of training epochs to perform
        auto_find_batch_size=True,  # Whether to automatically find an appropriate batch size
        save_total_limit=3,  # Limit the total amount of checkpoints and delete the older checkpoints
        eval_strategy="epoch",  # The evaluation strategy to adopt during training
        save_strategy="epoch",  # The checkpoint save strategy to adopt during training
        save_steps=100,  # Number of update steps before two checkpoint saves
        eval_steps=3,  # Number of update steps before two evaluations
        remove_unused_columns=False,  # Whether to remove columns not used by the model when using a dataset
        fp16=True,  # Whether to use 16-bit float precision instead of 32-bit for saving memory
        tf32=True,  # Whether to use tf32 precision instead of 32-bit for saving memory
        hub_model_id=hf_model_name,  # The model ID on the Hugging Face model hub
        report_to="wandb",
        run_name=run_name,
        per_device_train_batch_size=116,
        ignore_data_skip=True,
        push_to_hub=True,
        ddp_find_unused_parameters=False,
        batch_eval_metrics=True,
        hub_strategy="end",
        optim = "adamw_torch_fused",
        learning_rate=learning_rate,  # The initial learning rate for Adam
        lr_scheduler_type = "cosine_with_restarts",
        # - Try this during winter quarter - torch_compile=True,
        # - Try this during winter quarter - # deepspeed=./config.json
    )


    # Define the trainer
    trainer = Trainer(
        model=testmodel,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    if load_from == "checkpoint": trainer.train(resume_from_checkpoint=True)
    else: trainer.train()

if __name__ == "__main__":
    main()
