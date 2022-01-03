import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    TFAutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    create_optimizer,
)
from transformers.keras_callbacks import PushToHubCallback
from tensorflow.keras.callbacks import TensorBoard as TensorboardCallback

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--dataset_id", type=str)
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Load DatasetDict
    dataset = load_dataset(args.dataset_id)
    ner_labels = dataset["train"].features["ner_tags"].feature.names

    # Preprocess train dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            # get a list of tokens their connecting word id (for words tokenized into multiple chunks)
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to the current
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # define tokenizer_columns
    # tokenizer_columns is the list of keys from the dataset that get passed to the TensorFlow model
    pre_tokenizer_columns = set(dataset["train"].features)
    tokenizer_columns = list(set(tokenized_datasets["train"].features) - pre_tokenizer_columns)

    # test size will be 15% of train dataset
    test_size = 0.15
    processed_dataset = tokenized_datasets["train"].shuffle().train_test_split(test_size=test_size)

    # convert to TF datasets
    # Data collator that will dynamically pad the inputs received, as well as the labels.
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="tf")

    # converting our train dataset to tf.data.Dataset
    tf_train_dataset = processed_dataset["train"].to_tf_dataset(
        columns=tokenizer_columns,
        shuffle=False,
        batch_size=args.train_batch_size,
        collate_fn=data_collator,
    )

    # converting our test dataset to tf.data.Dataset
    tf_eval_dataset = processed_dataset["test"].to_tf_dataset(
        columns=tokenizer_columns,
        shuffle=False,
        batch_size=args.eval_batch_size,
        collate_fn=data_collator,
    )

    # Prepare model labels - useful in inference API
    id2label = {str(i): label for i, label in enumerate(ner_labels)}
    label2id = {v: k for k, v in id2label.items()}

    # enable mixed precision training
    if args.fp16:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    num_train_steps = len(tf_train_dataset) * args.num_train_epochs
    optimizer, lr_schedule = create_optimizer(
        init_lr=args.learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=args.weight_decay_rate,
        num_warmup_steps=args.num_warmup_steps,
    )

    model = TFAutoModelForTokenClassification.from_pretrained(
        args.model_id,
        id2label=id2label,
        label2id=label2id,
    )

    model.compile(optimizer=optimizer)

    callbacks = []
    callbacks.append(TensorboardCallback(log_dir=os.path.join(args.model_dir, "logs")))

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

    metric = load_metric("seqeval")

    def evaluate(model, dataset, ner_labels):
        all_predictions = []
        all_labels = []
        for batch in dataset:
            logits = model.predict(batch)["logits"]
            labels = batch["labels"]
            predictions = np.argmax(logits, axis=-1)
            for prediction, label in zip(predictions, labels):
                for predicted_idx, label_idx in zip(prediction, label):
                    if label_idx == -100:
                        continue
                    all_predictions.append(ner_labels[predicted_idx])
                    all_labels.append(ner_labels[label_idx])
        return metric.compute(predictions=[all_predictions], references=[all_labels])

    results = evaluate(model, tf_eval_dataset, ner_labels=list(model.config.id2label.values()))
    logger.info(results)
    # Save result
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
