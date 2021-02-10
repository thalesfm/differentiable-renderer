import json
import numpy as np
import matplotlib.pyplot as plt


def loadimg(fname):
    with open(fname) as fp:
        img = json.load(fp)
    return np.array(img)


def preprocess(img, brightness, gamma=2.2):
    img = brightness * img
    img = np.clip(img, 0, 1)
    return img**(1/gamma)
