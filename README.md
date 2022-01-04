# End-to-End Image Classification example using Keras and Hugging Face Transformers

* [Notebook](./image-classification.ipynb)


Welcome to this end-to-end Image Classification example using Keras and Hugging Face Transformers. In this demo, we will use the Hugging Faces `transformers` and `datasets` library together with `Tensorflow` & `Keras` to fine-tune a pre-trained vision transformer for image classification.

We are going to use the [EuroSAT](https://paperswithcode.com/dataset/eurosat) dataset for land use and land cover classification. The dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.

More information for the dataset can be found at the [repository](https://github.com/phelber/eurosat).


We are going to use all of the great Feature from the Hugging Face ecosystem like model versioning and experiment tracking as well as all the great features of Keras like Early Stopping and Tensorboard.
