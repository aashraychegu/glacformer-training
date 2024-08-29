# glacformer_training_script.py
This is a training script for the glacformer model. It uses huggingface's Transformers library for training the transformer model and the Dataset library for loading the dataset.

On machines with multiple GPU's, it is possible to use Huggingface accelerate to speed up training times.

Here are the relevant commands:
 - [Simple Accelerating with Accelerate](https://huggingface.co/docs/transformers/perf_train_gpu_many?select-gpu=Accelerate#gpu-selection)
 - [Fully Sharded Data Parallel for more advanced use cases](https://huggingface.co/docs/transformers/fsdp)