import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
import datasets
from transformers import (
    ViTFeatureExtractor,
    TFViTForImageClassification,
    DefaultDataCollator,
    create_optimizer,
)
from transformers.keras_callbacks import PushToHubCallback
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback,EarlyStopping


def create_image_folder_dataset(root_path):
  """creates `Dataset` from image folder structure"""
  
  # get class names by folders names
  _CLASS_NAMES= os.listdir(root_path)
  # defines `datasets` features`
  features=datasets.Features({
                      "img": datasets.Image(),
                      "label": datasets.features.ClassLabel(names=_CLASS_NAMES),
                  })
  # temp list holding datapoints for creation
  img_data_files=[]
  label_data_files=[]
  # load images into list for creation
  for img_class in os.listdir(root_path):
    for img in os.listdir(os.path.join(root_path,img_class)):
      path_=os.path.join(root_path,img_class,img)
      img_data_files.append(path_)
      label_data_files.append(img_class)
  # create dataset
  ds = datasets.Dataset.from_dict({"img":img_data_files,"label":label_data_files},features=features)
  return ds



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay_rate", type=float, default=0.01)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--fp16", type=bool, default=True)

    # Data, model, and output directories
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    args, _ = parser.parse_known_args()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load Dataset from local path
    dataset_path="/opt/ml/input/data/dataset"
    eurosat_ds = create_image_folder_dataset(dataset_path)
    img_class_labels = eurosat_ds.features["label"].names


    feature_extractor = ViTFeatureExtractor.from_pretrained(args.model_id)

    # basic processing (only resizing)
    def process(examples):
        examples.update(feature_extractor(examples['img'], ))
        return examples

    # we are also renaming our label col to labels to use `.to_tf_dataset` later
    eurosat_ds = eurosat_ds.rename_column("label", "labels")
    processed_dataset = eurosat_ds.map(process, batched=True)


    test_size=.15

    processed_dataset = processed_dataset.shuffle().train_test_split(test_size=test_size)

    # convert to TF datasets
    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DefaultDataCollator(return_tensors="tf")

    # converting our train dataset to tf.data.Dataset
    tf_train_dataset = processed_dataset["train"].to_tf_dataset(
    columns=['pixel_values'],
    label_cols=["labels"],
    shuffle=True,
    batch_size=args.train_batch_size,
    collate_fn=data_collator)

    # converting our test dataset to tf.data.Dataset
    tf_eval_dataset = processed_dataset["test"].to_tf_dataset(
    columns=['pixel_values'],
    label_cols=["labels"],
    shuffle=True,
    batch_size=args.eval_batch_size,
    collate_fn=data_collator)

    # Prepare model labels - useful in inference API
    id2label = {str(i): label for i, label in enumerate(img_class_labels)}
    label2id = {v: k for k, v in id2label.items()}

    # enable mixed precision training
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # create optimizer wight weigh decay
    num_train_steps = len(tf_train_dataset) * args.num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=args.weight_decay_rate,
        num_warmup_steps=args.num_warmup_steps,
    )

    # load pre-trained ViT model
    model = TFViTForImageClassification.from_pretrained(
        args.model_id,
        num_labels=len(img_class_labels),
        id2label=id2label,
        label2id=label2id,
    )

    # define loss
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # define metrics 
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(3, name="top-3-accuracy"),
    ]

    # compile model
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics
                )

    callbacks = []
    callbacks.append(TensorboardCallback(log_dir=os.path.join(args.model_dir, "logs")))
    callbacks.append(EarlyStopping(monitor="val_accuracy",patience=1))

    # TODO: add with new DLC supporting Transformers 4.14.1
    # if args.hub_token:
    #     callbacks.append(
    #         PushToHubCallback(
    #             output_dir=args.model_dir,
    #             tokenizer=tokenizer,
    #             hub_model_id=args.hub_model_id,
    #             hub_token=args.hub_token,
    #         )
    #     )

    # Training
    logger.info("*** Train ***")
    model.fit(
        tf_train_dataset,
        validation_data=tf_eval_dataset,
        callbacks=callbacks,
        epochs=args.num_train_epochs,
    )

    # Save result
    model.save_pretrained(args.model_dir)
    feature_extractor.save_pretrained(args.model_dir)
