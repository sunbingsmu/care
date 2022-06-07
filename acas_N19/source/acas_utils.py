import os
import time

import numpy as np
import random
import tensorflow
from tensorflow import set_random_seed

random.seed(123)
np.random.seed(123)
set_random_seed(123)

import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_backdoor.fix_gpu_memory()
    start_analysis()

    pass


if __name__ == '__main__':
    #sys.stdout = open('file', 'w')
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
    #sys.stdout.close()