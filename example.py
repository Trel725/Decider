import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from BSFilter2 import BSFilter
import glob
from Decider import Decider
import os
import re
import natsort


def cuda_memgrowth():
    # needed to initialize CUDA
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


cuda_memgrowth()


def norm_func(x, a=0, b=1):
    # function, applied to each spectrum
    return ((b - a) * (x - min(x))) / (max(x) - min(x)) + a


def read_preproc(file):
    sample = pd.read_csv(file)
    sample = sample.apply(norm_func, axis=1).values[:, :2089]
    return sample


def load_by_parts(fname):
    # loads array, splitted into multiple files
    # needseded due to GitHub limit on file size
    parts = []
    for idx in range(100):
        file = f"{fname}_{idx:02d}.npy"
        if os.path.isfile(file):
            p = np.load(file)
            parts.append(p)
        else:
            break
    if len(parts) == 0:
        raise ValueError("No such files found!")

    return np.concatenate(parts, axis=0)


nnmodel = load_model("./data/model.h5",
                     custom_objects={"BSFilter": BSFilter})
x_train = load_by_parts("./data/X_train")
y_train = np.load("./data/Y_train.npy")

decider = Decider(nnmodel,
                  x_train,
                  y_train,
                  epsilon=0.075,
                  prior_positive=0.5,
                  n_samples_prediction=200,
                  n_samples_reference=20)


for dirct in natsort.natsorted(glob.glob("./data/unknown/*")):
    if os.path.isdir(dirct):
        idx = re.findall(r'\d+', os.path.basename(dirct))[0]
        idx = int(idx)
        print(f"Reading unknown #{idx}")
        samples = []
        for idx, file in enumerate(glob.glob(dirct + "/*.csv")):
            print(f"  Found sample {idx}, {file}")
            sample = read_preproc(file)
            samples.append(sample)

        decider.decide_samples(samples)
        decider.visualize_sample()
        plt.show()
        print("-" * 10 + "\n")
