# Flower Classifier

This is a Python script that uses a pre-trained TensorFlow model to classify the type of flower in an input image. This is a Python script that uses a pre-trained TensorFlow model to classify the type of flower in an input image.

## Prerequisites

Before running the script, ensure that you have the following dependencies installed:

- Python (version 3.6 or higher)
- TensorFlow (version 2.x)
- PIL (Python Imaging Library)
- NumPy
- argparse
- tabulate
- tensorflow_hub

### Alternately, you can set the same environment using Anaconda

```shell
conda env create -f environment.yml
conda activate <environment_name>
```
## Usage

To classify a flower image, run the following command:

```shell
python predict.py image_path [--saved_model MODEL_FILE] [--k TOP_K_CLASSES] [--cat_names CATEGORY_NAMES_FILE]
```

* `image_path`: Path to the input image file.
* `--saved_model` MODEL_FILE (optional): Path to the saved model file (default: ./flower_classifier_model/model_1.h5).
* `--k` TOP_K_CLASSES (optional): Number of top predicted classes to display (default: 5).
* `--cat_names` CATEGORY_NAMES_FILE (optional): Path to the JSON file mapping labels to flower names (default: ./flower_classifier_model/label_map.json).

The script will display the top predicted classes for the given image along with their probabilities.

## Example

```shell
python predict.py test_images/rose.jpg --saved_model model.h5 --k 3 --cat_names label_map.json
```

