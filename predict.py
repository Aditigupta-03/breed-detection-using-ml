import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir
import bounding_box

H = 512
W = 512

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("test_images/mask")

    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    data_x = glob("test_images/image/*")
    for path in tqdm(data_x, total=len(data_x)):
        name = path.split("/")[-1].split(".")[0]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        x = cv2.resize(image, (W, H))
        x = x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)[0]
        y = cv2.resize(y, (w, h))
        y = np.expand_dims(y, axis=-1)
        masked_image = image * y
        line = np.ones((h, 10, 3)) * 128
        cv2.imwrite(f"test_images/mask/{name}.png", line)
