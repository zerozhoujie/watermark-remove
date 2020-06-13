from pathlib import Path
import random
import numpy as np
import cv2
from keras.utils import Sequence

def random_wm(img,imgsize):
    TRANSPARENCY = random.randint(80, 120)
    img = Image.fromarray(img)
    root = './logo/'
    wmpath = root + str(random.randint(1,37)) + '.png'
    wmimg = Image.open(wmpath).convert("RGBA")
    w,h = wmimg.size
    if w > imgsize:
        new_w = imgsize
        new_h = int(h/(w/imgsize))
    else:
        new_w = w
        new_h = h
    height = np.random.randint(image_size-new_h)
    wmimg = wmimg.resize((new_w, new_h), Image.BILINEAR)
    paste_mask = wmimg.split()[3].point(lambda i: i * TRANSPARENCY / 100.)
    img.paste(wmimg, (0, height), mask=paste_mask)
    # img.show()
    return img

class NoisyImageGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, image_size=64):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_num // self.batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 3), dtype=np.uint8)
        sample_id = 0

        while True:
            image_path = random.choice(self.image_paths)
            x_image = cv2.imread(str(image_path))
            y_image = cv2.imread(str(image_path).replace('x_train','y_train'))
            h, w, _ = x_image.shape
            #image = Image.open(str(image_path)).convert("RGB")
            #w, h = image.size
            x_image = np.array(x_image)
            y_image = np.array(y_image)             

            if h >= image_size and w >= image_size:
                #h, w, _ = x_image.shape
                i = np.random.randint(h - image_size + 1)
                j = np.random.randint(w - image_size + 1)
                x_ = x_image[i:i + image_size, j:j + image_size]
                y_ = y_image[i:i + image_size, j:j + image_size]
                #x_ = random_wm(y_,image_size)
                x[sample_id] = x_
                y[sample_id] = y_

                sample_id += 1

                if sample_id == batch_size:
                    return x, y


class ValGenerator(Sequence):
    def __init__(self, image_dir):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(image_paths)
        self.data = []

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

        for image_path in image_paths:
            x = cv2.imread(str(image_path))
            h, w, _ = x.shape
            x = x[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
            y = cv2.imread(str(image_path).replace('x_test','y_test'))
            h, w, _ = y.shape
            y = y[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)

            self.data.append([np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)])

    def __len__(self):
        return self.image_num

    def __getitem__(self, idx):
        return self.data[idx]
