import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # To avoid warnings being displayed
warnings.filterwarnings("ignore")

import argparse
from tabulate import tabulate
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import tensorflow_hub as hub


def print_table(*columns, headers):
    table_data = list(zip(*columns))
    table = tabulate(table_data, headers=headers, tablefmt="grid")
    print(table)
    print()

def process_image(img):
    """
    Preprocesses the input image.

    Args:
        img: Image as a NumPy array.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Resize the image to a fixed size
    image_size = 224
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (image_size, image_size))
    img /= 255  # Normalize pixel values between 0 and 1
    img = np.asarray(img)
    return img

def predict(image_path, model, k):
    """
    Performs flower classification on the input image using the provided model.

    Args:
        image_path (str): Path to the input image file.
        model (tf.keras.Model): Trained TensorFlow model for flower classification.
        k (int): Number of top predictions to retrieve.

    Returns:
        tuple: Tuple containing the top probabilities and corresponding class indices.
    """
    # Open and preprocess the input image
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    img = np.expand_dims(img, axis=0)

    # Make predictions using the model
    ps = model.predict(img)

    # Get the top k predictions and their corresponding class indices
    top_probabilities = np.sort(ps)[0, :-(k + 1):-1]
    top_classes = np.argsort(ps)[0, :-(k + 1):-1]

    return top_probabilities, top_classes

def main(img_filepath, model_filepath, k, class_labels):
    """
    Main function to classify the flower in the input image.

    Args:
        img_filepath (str): Path to the input image file.
        model_filepath (str): Path to the saved model file.
        k (int): Number of top predictions to display.
        class_labels (str): Path to the JSON file mapping labels to flower names.
    """

    # Load pre-trained tensorflow model
    reloaded_model = tf.keras.models.load_model(model_filepath, custom_objects={'KerasLayer':hub.KerasLayer})

    # Load list of labels from the json file
    with open(class_labels, 'r') as f:
        label_map = json.load(f)
    
    # Perform prediction on the input image
    probs, classes = predict(img_filepath, reloaded_model, k)

    # Get the flower names corresponding to the predicted classes
    labels = [label_map[str(c + 1)] for c in classes]

    print('\nThe top {} predicted classes for the given image are:\n'.format(k))
    headers = ['No', 'Name', 'Probability']

    print_table(list(range(1, len(probs)+1)), labels, [f'{p:.3f}' for p in probs], headers=headers)

if __name__ == '__main__':
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(prog='Flower Classifier', description='Identify the flower')
    parser.add_argument('image', type=str)
    parser.add_argument('--saved_model', type=str,
                        default=os.path.join('.','flower_classifier_model','model_1.h5'))        
    parser.add_argument('--k', metavar='top_k_classes', type=int,
                        help='Number of highest probable flower names',                        
                        default=5)
    parser.add_argument('--cat_names', metavar='Category names list', type=str,
                        help='Path to a JSON file mapping labels to flower names',
                        default=os.path.join('.','flower_classifier_model','label_map.json'))
    args = parser.parse_args()

    image = args.image
    model = args.saved_model
    k = args.k
    category_names= args.cat_names

    # Run the main function with the provided arguments
    main(image, model, k, category_names)