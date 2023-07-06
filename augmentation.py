import os
import imutils
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate # augmentation library
from bounding_box import mask_to_bbox, mask_to_border, parse_mask

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)



def load_data(path, split= 0.5):
    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)
    train_x, test_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(Y, test_size=split_size, random_state=42)
    return (train_x, train_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment= True):
    H = 512
    W = 512
    for x,y in tqdm(zip(images,masks),total=len(images)):
        name = x.split("\\")[-1].split(".")[0]
        # print(name)
        """read the images and masks"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)
        """Augmentation"""
        if augment == True:
            aug = HorizontalFlip(p= 1.0)
            augmented = aug(image = x, mask =y)
            x1 = augmented["image"]
            y1 = augmented["mask"]
            x2 = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
            y2 = y

            aug = ChannelShuffle(p=1)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = CoarseDropout(p=1, min_holes=3, max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']
            X = [x, x1, x2, x3, x4]
            Y = [y, y1, y2, y3, y4]
        else:
            X = [x]
            Y = [y]
        index = 0
        for i, m in zip(X,Y):
            i = cv2.resize(i, (512,512))
            m = cv2.resize(m, (512,512))
            temp_image_name = f"{name}_{index}.png"
            temp_mask_name = f"{name}_{index}.png"
            image_path = os.path.join(save_path, "images", temp_image_name)
            mask_path =  os.path.join(save_path, "masks", temp_mask_name)
            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            index +=1


if __name__ == "__main__":
    """Seeding"""
    np.random.seed(42)
    data_path = "data"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)
    print(f'Train: {len(train_x)}, {len(train_y)}')
    print(f'Test: {len(test_x)}, {len(test_y)}')
    """save file of augmented data"""
    create_dir("new_data/train/images/")
    create_dir("new_data/train/masks/")
    create_dir("new_data/test/images/")
    create_dir("new_data/test/masks/")
    """augmentation apply: """
    augment_data(train_x, train_y, "new_data/train/", augment= True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
    images = sorted(glob(os.path.join("new_data/train", "images", "*")))
    masks = sorted(glob(os.path.join("new_data/train", "masks", "*")))
