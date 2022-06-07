import os
import time

import numpy as np
import random
import tensorflow
import keras
import h5py
from tensorflow import set_random_seed
random.seed(123)
np.random.seed(123)
set_random_seed(123)

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from causal_inference import causal_analyzer

import utils_backdoor

import sys
#from tensorflow.keras.models import model_from_json

##############################
#        PARAMETERS          #
##############################

DEVICE = '3'  # specify which GPU to use

INPUT_SHAPE = (5,)

MODEL_DIR = '../models'  # model directory
MODEL_FILENAME = 'ACASXU_3_3.h5'  # model file
PROPERTY = 'p2_n33'

DATA_DIR = '../data'

RESULT_DIR = '../results'  # directory for storing results

# parameters for optimization
BATCH_SIZE = 32  # batch size used for optimization

DRAWNDOWN_SIZE = BATCH_SIZE * 313
COUNTEREG_SIZE = BATCH_SIZE * 313

GEN_INPUT_SHAPE = (DRAWNDOWN_SIZE, 5, 1, 1)

MINI_BATCH = DRAWNDOWN_SIZE // BATCH_SIZE
##############################
#      END PARAMETERS        #
##############################
#property 2
#    "bounds": "[(0.6,0.6799), (-0.5,0.5), (-0.5,0.5), (0.45,0.5), (-0.5,-0.45)]",
#y!0
def __generate_x():
    x_0 = np.random.uniform(low=0.6, high=0.6799, size=(BATCH_SIZE, 1))
    x_1 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_2 = np.random.uniform(low=-0.5, high=0.5, size=(BATCH_SIZE, 1))
    x_3 = np.random.uniform(low=0.45, high=0.5, size=(BATCH_SIZE, 1))
    x_4 = np.random.uniform(low=-0.5, high=-0.45, size=(BATCH_SIZE, 1))
    x = np.array([x_0, x_1, x_2, x_3, x_4])
    return np.transpose(x.reshape(5,32))

def property_satisfied(pre_y):
    #pre_y = np.argmin(y, axis=1)
    if pre_y != 0:
        return True
    return False

def gen_data_set(model, data_path):
    """generate drawndown and counter example data set"""
    # property 8
    n_dd = 0
    n_cex = 0
    dd = []
    cex = []
    dd_saved = 0
    cex_saved = 0

    while True:
        x = __generate_x()

        y = model.predict(x.reshape(BATCH_SIZE,5))
        pre_y =  np.argmin(y, axis=1)

        for i in range (0, BATCH_SIZE):
            print('n_dd:{}'.format(n_dd))
            print('n_cex:{}'.format(n_cex))
            if (property_satisfied(pre_y[i])):
                if n_dd < DRAWNDOWN_SIZE:
                    dd.append(x[i])
                    n_dd = n_dd + 1
            else:
                if n_cex < COUNTEREG_SIZE:
                    cex.append(x[i])
                    n_cex = n_cex + 1

        if n_dd >= DRAWNDOWN_SIZE and dd_saved == 0:
            dd_saved = 1
            with h5py.File(data_path + '/drawndown.h5', 'w') as f:
                f.create_dataset("drawndown", data=np.array(dd))

        if n_cex >= COUNTEREG_SIZE and cex_saved == 0:
            cex_saved = 1
            with h5py.File(data_path + '/counterexample.h5', 'w') as f:
                f.create_dataset("counterexample", data=np.array(cex))

        if dd_saved == 1 and cex_saved == 1:
            break


    return n_dd, n_cex


def load_dataset(data_path):

    with h5py.File(data_path + '/drawndown_' + PROPERTY + '.h5', 'r') as hf:
        dd = hf['drawndown'][:]

    with h5py.File(data_path + '/counterexample_' + PROPERTY +'.h5', 'r') as hf:
        cex = hf['counterexample'][:]

    print('Dawndown shape %s' % str(dd.shape))
    print('Counterexample shape %s' % str(cex.shape))

    return dd, cex



def build_data_loader(X):
    X = X.reshape(GEN_INPUT_SHAPE)
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, batch_size=BATCH_SIZE)

    return generator


def trigger_analyzer(analyzer, dd_gen, cex_gen):

    visualize_start_time = time.time()

    # execute reverse engineering
    analyzer.analyze(dd_gen, cex_gen)

    visualize_end_time = time.time()
    print('visualization cost %f seconds' %
          (visualize_end_time - visualize_start_time))

    return

def start_analysis():
    '''
    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)
    print('generating datasets')
    dd, cex = gen_data_set(model, DATA_DIR)
    print ('n_dd: {}, n_cex: {}'.format(dd, cex))
    '''

    print('loading dataset')
    dd, cex = load_dataset(DATA_DIR)
    # transform numpy arrays into data generator
    dd_generator = build_data_loader(dd)
    cex_generator = build_data_loader(cex)

    print('loading model')
    model_file = '%s/%s' % (MODEL_DIR, MODEL_FILENAME)
    model = load_model(model_file)

    # initialize analyzer
    analyzer = causal_analyzer(
        model,
        dd_generator, cex_generator,
        input_shape=INPUT_SHAPE,
        mini_batch=MINI_BATCH,
        batch_size=BATCH_SIZE, verbose=2)

    trigger_analyzer(analyzer, dd_generator, cex_generator)
    #'''
    pass


def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
    utils_backdoor.fix_gpu_memory()
    for i in range (0, 3):
        print(i)
        start_analysis()

    pass


if __name__ == '__main__':
    #sys.stdout = open('file', 'w')
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print('elapsed time %s s' % elapsed_time)
    #sys.stdout.close()