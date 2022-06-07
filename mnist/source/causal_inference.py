#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2021-06-12 16:27:19
# @Author  : Sun Bing

import numpy as np
from keras import backend as K

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import UpSampling2D, Cropping2D
from keras.layers import Input
from keras import Model
import pyswarms as ps

import utils_backdoor

from decimal import Decimal

import os
import sys
import time
from keras.preprocessing import image

##############################
#        PARAMETERS          #
##############################

RESULT_DIR = '../results'  # directory for storing results
IMG_FILENAME_TEMPLATE = 'mnist_visualize_%s_label_%d.png'  # image filename template for visualization results

# input size
IMG_ROWS = 28
IMG_COLS = 28
IMG_COLOR = 1
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
MASK_SHAPE = (IMG_ROWS, IMG_COLS)

NUM_CLASSES = 10  # total number of classes in the model

CALSAL_STEP = 4

TEST_ONLY = 1

class causal_analyzer:

    # upsample size, default is 1
    UPSAMPLE_SIZE = 1
    # pixel intensity range of image and preprocessing method
    # raw: [0, 255]
    # mnist: [0, 1]
    # imagenet: imagenet mean centering
    # inception: [-1, 1]
    INTENSITY_RANGE = 'mnist'
    # type of regularization of the mask
    REGULARIZATION = 'l1'
    # threshold of attack success rate for dynamically changing cost
    ATTACK_SUCC_THRESHOLD = 0.99
    # patience
    PATIENCE = 10
    # multiple of changing cost, down multiple is the square root of this
    COST_MULTIPLIER = 1.5,
    # if resetting cost to 0 at the beginning
    # default is true for full optimization, set to false for early detection
    RESET_COST_TO_ZERO = True
    # min/max of mask
    MASK_MIN = 0
    MASK_MAX = 1
    # min/max of raw pixel intensity
    COLOR_MIN = 0
    COLOR_MAX = 1
    # number of color channel
    IMG_COLOR = 1
    # whether to shuffle during each epoch
    SHUFFLE = True
    # batch size of optimization
    BATCH_SIZE = 32
    # verbose level, 0, 1 or 2
    VERBOSE = 1
    # whether to return log or not
    RETURN_LOGS = True
    # whether to save last pattern or best pattern
    SAVE_LAST = False
    # epsilon used in tanh
    EPSILON = K.epsilon()
    # early stop flag
    EARLY_STOP = True
    # early stop threshold
    EARLY_STOP_THRESHOLD = 0.99
    # early stop patience
    EARLY_STOP_PATIENCE = 2 * PATIENCE
    # save tmp masks, for debugging purpose
    SAVE_TMP = False
    # dir to save intermediate masks
    TMP_DIR = 'tmp'
    # whether input image has been preprocessed or not
    RAW_INPUT_FLAG = False

    SPLIT_LAYER = 6

    REP_N          = 5

    def __init__(self, model, generator, input_shape,
                 init_cost, steps, mini_batch, lr, num_classes,
                 upsample_size=UPSAMPLE_SIZE,
                 patience=PATIENCE, cost_multiplier=COST_MULTIPLIER,
                 reset_cost_to_zero=RESET_COST_TO_ZERO,
                 mask_min=MASK_MIN, mask_max=MASK_MAX,
                 color_min=COLOR_MIN, color_max=COLOR_MAX, img_color=IMG_COLOR,
                 shuffle=SHUFFLE, batch_size=BATCH_SIZE, verbose=VERBOSE,
                 return_logs=RETURN_LOGS, save_last=SAVE_LAST,
                 epsilon=EPSILON,
                 early_stop=EARLY_STOP,
                 early_stop_threshold=EARLY_STOP_THRESHOLD,
                 early_stop_patience=EARLY_STOP_PATIENCE,
                 save_tmp=SAVE_TMP, tmp_dir=TMP_DIR,
                 raw_input_flag=RAW_INPUT_FLAG,
                 rep_n=REP_N):


        self.model = model
        self.input_shape = input_shape
        self.gen = generator
        self.init_cost = init_cost
        self.steps = 1  #steps
        self.mini_batch = mini_batch
        self.lr = lr
        self.num_classes = num_classes
        self.upsample_size = upsample_size
        self.patience = patience
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5
        self.reset_cost_to_zero = reset_cost_to_zero
        self.mask_min = mask_min
        self.mask_max = mask_max
        self.color_min = color_min
        self.color_max = color_max
        self.img_color = img_color
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.verbose = verbose
        self.return_logs = return_logs
        self.save_last = save_last
        self.epsilon = epsilon
        self.early_stop = early_stop
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.save_tmp = save_tmp
        self.tmp_dir = tmp_dir
        self.raw_input_flag = raw_input_flag

        self.rep_n = rep_n       # number of neurons to repair
        self.r_weight = None
        self.target = 3
        self.alpha = 0.2

        # split the model for causal inervention
        '''
        self.model1 = Model(inputs=self.model.inputs, outputs=self.model.layers[5].output)
        model2_input = Input(self.model.layers[6].input_shape[1:])
        self.model12 = model2_input
        for layer in self.model.layers[6:]:
            self.model12 = layer(self.model12)
        self.model12 = Model(inputs=model2_input, outputs=self.model12)
        '''
        self.model1, self.model2 = self.split_keras_model(self.model, self.SPLIT_LAYER)

        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    def get_perturbed_input(self, x):

        mask_flatten = []
        pattern_flatten = []

        mask = []
        pattern = []

        y_label = self.target
        mask_filename = IMG_FILENAME_TEMPLATE % ('mask', y_label)
        if os.path.isfile('%s/%s' % (RESULT_DIR, mask_filename)):
            img = image.load_img(
                '%s/%s' % (RESULT_DIR, mask_filename),
                color_mode='grayscale',
                target_size=INPUT_SHAPE)
            mask = image.img_to_array(img)
            mask /= 255
            mask = mask[:, :, 0]

        pattern_filename = IMG_FILENAME_TEMPLATE % ('pattern', y_label)
        if os.path.isfile('%s/%s' % (RESULT_DIR, pattern_filename)):
            img = image.load_img(
                '%s/%s' % (RESULT_DIR, pattern_filename),
                color_mode='rgb',
                target_size=INPUT_SHAPE)
            pattern = image.img_to_array(img)

        pattern = pattern[:, :, 1] / 255.

        filtered = np.multiply(x, np.expand_dims(np.subtract(np.ones((MASK_SHAPE)), mask), axis=2))

        fusion = np.expand_dims(np.multiply(pattern, mask),axis=2)

        x_out = np.add(filtered, fusion)

        #test
        #'''
        utils_backdoor.dump_image(x[0]* 255,
                                  '../results/ori_img0.png',
                                  'png')
        utils_backdoor.dump_image(x_out[0] * 255,
                                  '../results/img0.png',
                                  'png')
        
        utils_backdoor.dump_image(np.expand_dims(mask, axis=2) * 255,
                                  '../results/mask_test.png',
                                  'png')
        utils_backdoor.dump_image(np.expand_dims(pattern,axis=2)* 255, '../results/pattern_test.png', 'png')

        fusion = np.expand_dims(np.multiply(pattern, mask), axis=2)

        utils_backdoor.dump_image(fusion, '../results/fusion_test.png', 'png')
        #'''
        return x_out

    def injection_func(self, mask, pattern, adv_img):
        return mask * pattern + (1 - mask) * adv_img

    def reset_opt(self):

        K.set_value(self.opt.iterations, 0)
        for w in self.opt.weights:
            K.set_value(w, np.zeros(K.int_shape(w)))

        pass

    def reset_state(self, pattern_init, mask_init):

        print('resetting state')

        # setting cost
        if self.reset_cost_to_zero:
            self.cost = 0
        else:
            self.cost = self.init_cost
        K.set_value(self.cost_tensor, self.cost)

        # setting mask and pattern
        mask = np.array(mask_init)
        pattern = np.array(pattern_init)
        mask = np.clip(mask, self.mask_min, self.mask_max)
        pattern = np.clip(pattern, self.color_min, self.color_max)
        mask = np.expand_dims(mask, axis=2)

        # convert to tanh space
        mask_tanh = np.arctanh((mask - 0.5) * (2 - self.epsilon))
        pattern_tanh = np.arctanh((pattern / 255.0 - 0.5) * (2 - self.epsilon))
        print('mask_tanh', np.min(mask_tanh), np.max(mask_tanh))
        print('pattern_tanh', np.min(pattern_tanh), np.max(pattern_tanh))

        K.set_value(self.mask_tanh_tensor, mask_tanh)
        K.set_value(self.pattern_tanh_tensor, pattern_tanh)

        # resetting optimizer states
        self.reset_opt()

        pass

    def save_tmp_func(self, step):

        cur_mask = K.eval(self.mask_upsample_tensor)
        cur_mask = cur_mask[0, ..., 0]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_mask_step_%d.png' % step))
        utils_backdoor.dump_image(np.expand_dims(cur_mask, axis=2) * 255,
                                  img_filename,
                                  'png')

        cur_fusion = K.eval(self.mask_upsample_tensor *
                            self.pattern_raw_tensor)
        cur_fusion = cur_fusion[0, ...]
        img_filename = (
            '%s/%s' % (self.tmp_dir, 'tmp_fusion_step_%d.png' % step))
        utils_backdoor.dump_image(cur_fusion, img_filename, 'png')

        pass

    def analyze(self, gen):
        alpha_list = [0.9]

        for alpha in alpha_list:
            self.alpha = alpha
            print('alpha: {}'.format(alpha))
            for i in range(0, 1):
                print('iteration: {}'.format(i))
                self.analyze_each(gen)

    def analyze_each(self, gen):
        #'''
        ana_start_t = time.time()
        # find hidden range
        for step in range(self.steps):
            min = []
            min_p = []
            max = []
            max_p = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)
                min_i, max_i = self.get_h_range(X_batch)
                min.append(min_i)
                max.append(max_i)

                min_i, max_i = self.get_h_range(X_batch_perturbed)
                min_p.append(min_i)
                max_p.append(max_i)

                p_prediction = self.model.predict(X_batch_perturbed)
                ori_predict = self.model.predict(X_batch)
                np.savetxt("../results/p_prediction.txt", p_prediction, fmt="%s")
                np.savetxt("../results/ori_predict.txt", ori_predict, fmt="%s")
                predict = np.argmax(p_prediction, axis=1)
                ori_predict = np.argmax(ori_predict, axis=1)
                labels = np.argmax(Y_batch, axis=1)


            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)

            min_p = np.min(np.array(min_p), axis=0)
            max_p = np.max(np.array(max_p), axis=0)
        #'''
        # loop start

        for step in range(self.steps):
            #'''
            ie_batch = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch, _ = gen.next()

                #X_batch_perturbed = self.get_perturbed_input(X_batch)

                # find hidden neuron interval

                # find
                #ie_batch.append(self.get_ie_do_h(X_batch, np.minimum(min_p, min), np.maximum(max_p, max)))
                ie_batch.append(self.get_tie_do_h(X_batch, self.target, np.minimum(min_p, min), np.maximum(max_p, max)))

            ie_mean = np.mean(np.array(ie_batch),axis=0)

            np.savetxt("../results/ori.txt", ie_mean, fmt="%s")
            #return
            # ie_mean dim: 512 * 43
            # find tarted class: diff of each column
            col_diff = np.max(ie_mean, axis=0) - np.min(ie_mean, axis=0)
            col_diff = np.transpose([np.arange(len(col_diff)), col_diff])
            ind = np.argsort(col_diff[:, 1])[::-1]
            col_diff = col_diff[ind]

            np.savetxt("../results/col_diff.txt", col_diff, fmt="%s")

            row_diff = np.max(ie_mean, axis=1) - np.min(ie_mean, axis=1)
            row_diff = np.transpose([np.arange(len(row_diff)), row_diff])
            ind = np.argsort(row_diff[:, 1])[::-1]
            row_diff = row_diff[ind]

            np.savetxt("../results/row_diff.txt", row_diff, fmt="%s")

            ana_start_t = time.time() - ana_start_t
            print('fault localization time: {}s'.format(ana_start_t))
            #'''
            rep_t = time.time()
            # row_diff contains sensitive neurons: top self.rep_n
            # index
            self.rep_index = []
            result, acc = self.pso_test([], self.target)
            print("before repair: attack SR: {}, BE acc: {}".format(result, acc))
            #'''
            self.rep_index = row_diff[:,:1][:self.rep_n,:]
            print("repair index: {}".format(self.rep_index.T))
            #'''
            #self.rep_index = [1563, 1552, 1547, 1331, 1541]

            print("repair index: {}".format(self.rep_index))
            #'''

            self.repair()

            rep_t = time.time() - rep_t

            result, acc = self.pso_test(self.r_weight, self.target)
            print("after repair: attack SR: {}, BE acc: {}".format(result, acc))
            print('PSO time: {}s'.format(rep_t))

    pass

    def analyze_gradient(self, gen):
        #'''
        ana_start_t = time.time()
        # find hidden range
        for step in range(self.steps):
            min = []
            min_p = []
            max = []
            max_p = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)
                min_i, max_i = self.get_h_range(X_batch)
                min.append(min_i)
                max.append(max_i)

                min_i, max_i = self.get_h_range(X_batch_perturbed)
                min_p.append(min_i)
                max_p.append(max_i)

                p_prediction = self.model.predict(X_batch_perturbed)
                ori_predict = self.model.predict(X_batch)
                np.savetxt("../results/p_prediction.txt", p_prediction, fmt="%s")
                np.savetxt("../results/ori_predict.txt", ori_predict, fmt="%s")
                predict = np.argmax(p_prediction, axis=1)
                ori_predict = np.argmax(ori_predict, axis=1)
                labels = np.argmax(Y_batch, axis=1)


            min = np.min(np.array(min), axis=0)
            max = np.max(np.array(max), axis=0)

            min_p = np.min(np.array(min_p), axis=0)
            max_p = np.max(np.array(max_p), axis=0)
        #'''
        # loop start

        for step in range(self.steps):
            #'''
            ie_batch = []
            #self.mini_batch = 2
            for idx in range(self.mini_batch):
                X_batch, _ = gen.next()

                #X_batch_perturbed = self.get_perturbed_input(X_batch)

                # find hidden neuron interval

                # find
                #ie_batch.append(self.get_ie_do_h(X_batch, np.minimum(min_p, min), np.maximum(max_p, max)))
                ie_batch.append(self.get_gradient(X_batch, self.target, min, max))

            ie_mean = np.mean(np.array(ie_batch),axis=0)

            np.savetxt("../results/ori.txt", ie_mean, fmt="%s")

            row_diff = ie_mean
            row_diff = np.transpose([np.arange(len(row_diff)), row_diff])
            ind = np.argsort(row_diff[:, 1])[::-1]
            row_diff = row_diff[ind]

            np.savetxt("../results/row_diff.txt", row_diff, fmt="%s")

            ana_start_t = time.time() - ana_start_t
            print('fault localization time: {}s'.format(ana_start_t))
            #'''
            rep_t = time.time()
            # row_diff contains sensitive neurons: top self.rep_n
            # index
            self.rep_index = []
            result, acc = self.pso_test([], self.target)
            print("before repair: attack SR: {}, BE acc: {}".format(result, acc))
            #'''
            self.rep_index = row_diff[:,:1][:self.rep_n,:]
            print("repair index: {}".format(self.rep_index.T))
            '''
            self.rep_index = [1563, 1552, 1547, 1331, 1541]
            print("repair index: {}".format(self.rep_index))
            '''

            self.repair()

            rep_t = time.time() - rep_t

            #self.rep_index = [461, 395, 491, 404, 219]
            #self.r_weight = [-0.13325777,  0.08095828, -0.80547224, -0.59831971, -0.23067632]

            result, acc = self.pso_test(self.r_weight, self.target)
            print("after repair: attack SR: {}, BE acc: {}".format(result, acc))
            print('PSO time: {}s'.format(rep_t))

    pass

    # return
    def get_ie_do_h(self, x, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                ie_i.append(np.mean(pre_final,axis=0))
            ie.append(np.mean(np.array(ie_i),axis=0))
        return np.array(ie)

    # get ie of targeted class
    def get_tie_do_h(self, x, t_dix, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                ie_i.append(np.mean(pre_final,axis=0)[t_dix])
            ie.append(np.array(ie_i))

        return np.array(ie)

    # get ie of targeted class
    def get_gradient_ie_do_h(self, x, t_dix, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):
            ie_i = []
            last_ie_i = 0
            first_loop = 0
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = _pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden.reshape(l_shape))
                this_ie_i = np.mean(pre_final,axis=0)[t_dix]

                #delta
                if first_loop == 0:
                    first_loop = 1
                    last_ie_i = this_ie_i
                    last_val = h_val
                    continue
                if (h_val - last_val) != 0.0:
                    delta = (this_ie_i - last_ie_i) / (h_val - last_val)
                else:
                    delta = (this_ie_i - last_ie_i)

                ie_i.append(delta)
                last_ie_i = this_ie_i
                last_val = h_val

            ie.append(np.array(ie_i))

        return np.array(ie)

    def get_gradient(self, x, t_dix, min, max):
        pre_layer5 = self.model1.predict(x)
        l_shape = pre_layer5.shape
        ie = []

        hidden_min = min.reshape(-1)
        hidden_max = max.reshape(-1)
        num_step = CALSAL_STEP

        _pre_layer5 = np.reshape(pre_layer5, (len(pre_layer5), -1))

        for i in range (len(_pre_layer5[0])):

            pre_final = self.model.predict(x)
            last_ie_i = np.mean(pre_final, axis=0)[t_dix]

            do_hidden = _pre_layer5.copy()
            last_val = do_hidden[:, i]
            delta_x = np.ones(do_hidden[:, i].shape)
            do_hidden[:, i] = np.add(last_val, delta_x)
            #do_hidden[:, i] = last_val * 1.05
            pre_final = self.model2.predict(do_hidden.reshape(l_shape))
            this_ie_i = np.mean(pre_final, axis=0)[t_dix]

            #delta = np.divide((this_ie_i - last_ie_i), (last_val * 0.05))
            delta = this_ie_i - last_ie_i

            ie.append(np.array(delta))
        #return (np.array(np.mean(np.array(ie),axis=1)))
        return np.array(ie)

    # return
    def get_die_do_h(self, x, x_p, min, max):
        pre_layer5 = self.model1.predict(x)
        pre_layer5_p = self.model1.predict(x_p)

        ie = []

        hidden_min = min
        hidden_max = max
        num_step = 16

        for i in range (len(pre_layer5[0])):
            ie_i = []
            for h_val in np.linspace(hidden_min[i], hidden_max[i], num_step):
                do_hidden = pre_layer5.copy()
                do_hidden[:, i] = h_val
                pre_final = self.model2.predict(do_hidden)
                pre_final_ori = self.model2.predict(pre_layer5_p)
                ie_i.append(np.mean(np.absolute(pre_final - pre_final_ori),axis=0))
            ie.append(np.mean(np.array(ie_i),axis=0))
        return np.array(ie)

    # return
    def get_final(self, x, x_p, min, max):
        return np.mean(self.model.predict(x),axis=0)

    def get_h_range(self, x):
        pre_layer5 = self.model1.predict(x)

        max = np.max(pre_layer5,axis=0)
        min = np.min(pre_layer5, axis=0)

        return min, max

    def repair(self):
        # repair
        print('Start reparing...')
        print('alpha: {}'.format(self.alpha))
        options = {'c1': 0.41, 'c2': 0.41, 'w': 0.8}
        #'''# original
        optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=self.rep_n, options=options,
                                            bounds=([[-10.0] * self.rep_n, [10.0] * self.rep_n]),
                                            init_pos=np.ones((20, self.rep_n), dtype=float), ftol=1e-3,
                                            ftol_iter=10)
        #'''

        # Perform optimization
        best_cost, best_pos = optimizer.optimize(self.pso_fitness_func, iters=100)

        # Obtain the cost history
        # print(optimizer.cost_history)
        # Obtain the position history
        # print(optimizer.pos_history)
        # Obtain the velocity history
        # print(optimizer.velocity_history)
        #print('neuron to repair: {} at layter: {}'.format(self.r_neuron, self.r_layer))
        #print('best cost: {}'.format(best_cost))
        #print('best pos: {}'.format(best_pos))

        self.r_weight = best_pos

        return best_pos

    # optimization target perturbed sample has the same label as clean sample
    def pso_fitness_func(self, weight):

        result = []
        for i in range (0, int(len(weight))):
            r_weight =  weight[i]

            cost = self.pso_test_rep(r_weight)

            #print('cost: {}'.format(cost))

            result.append(cost)

        #print(result)

        return result


    def pso_test_rep(self, r_weight):

        #result = []
        result = 0.0
        tot_count = 0
        correct = 0
        # per particle
        for idx in range(self.mini_batch):
            X_batch, Y_batch = self.gen.next()
            X_batch_perturbed = self.get_perturbed_input(X_batch)

            p_prediction = self.model1.predict(X_batch_perturbed)
            o_prediction = self.model1.predict(X_batch)
            l_shape = p_prediction.shape

            _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
            _o_prediction = np.reshape(o_prediction, (len(o_prediction), -1))

            do_hidden = _p_prediction.copy()
            o_hidden = _o_prediction.copy()

            for i in range (0, len(self.rep_index)):
                rep_idx = int(self.rep_index[i])
                do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
                o_hidden[:, rep_idx] = (r_weight[i]) * _o_prediction[:, rep_idx]

            p_prediction = self.model2.predict(do_hidden.reshape(l_shape))
            o_prediction = self.model2.predict(o_hidden.reshape(l_shape))

            # cost is the difference
            #cost = np.abs(p_prediction - Y_batch)
            #cost = np.mean(cost,axis=0)
            #result.append(cost)

            labels = np.argmax(Y_batch, axis=1)
            predict = np.argmax(p_prediction, axis=1)
            o_predict = np.argmax(o_prediction, axis=1)

            o_correct = np.sum(labels == o_predict)
            correct = correct + o_correct

            o_target = (labels == self.target * np.ones(predict.shape))
            pre_target = (predict == self.target * np.ones(predict.shape))

            attack_success = np.sum(predict == self.target * np.ones(predict.shape)) - np.sum(o_target & pre_target)

            #cost = np.sum(labels != predict)
            result = result + attack_success
            tot_count = tot_count + len(labels)

        result = result / tot_count
        correct = correct / tot_count

        cost = (1.0 - self.alpha) * result + self.alpha * (1 - correct)

        return cost

    def pso_test(self, r_weight, target):
        result = 0.0
        correct = 0.0
        tot_count = 0
        if len(self.rep_index) != 0:

            # per particle
            for idx in range(self.mini_batch):
                X_batch, Y_batch = self.gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)

                o_prediction = self.model1.predict(X_batch)
                p_prediction = self.model1.predict(X_batch_perturbed)

                _p_prediction = np.reshape(p_prediction, (len(p_prediction), -1))
                _o_prediction = np.reshape(o_prediction, (len(o_prediction), -1))

                l_shape = p_prediction.shape

                do_hidden = _p_prediction.copy()
                o_hidden = _o_prediction.copy()

                for i in range (0, len(self.rep_index)):
                    rep_idx = int(self.rep_index[i])
                    do_hidden[:, rep_idx] = (r_weight[i]) * _p_prediction[:, rep_idx]
                    o_hidden[:, rep_idx] = (r_weight[i]) * _o_prediction[:, rep_idx]

                p_prediction = self.model2.predict(do_hidden.reshape(l_shape))
                o_prediction = self.model2.predict(o_hidden.reshape(l_shape))

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(p_prediction, axis=1)
                o_predict = np.argmax(o_prediction, axis=1)

                # cost is the difference
                o_target = (labels == target * np.ones(predict.shape))
                pre_target = (predict == target * np.ones(predict.shape))

                attack_success = np.sum(predict == target * np.ones(predict.shape)) - np.sum(o_target & pre_target)
                #diff = np.sum(labels != predict)
                result = result + attack_success
                tot_count = tot_count + len(labels)

                o_correct = np.sum(labels == o_predict)
                correct = correct + o_correct

            result = result / tot_count
            correct = correct / tot_count
        else:
            # per particle
            for idx in range(self.mini_batch):
                X_batch, Y_batch = self.gen.next()
                X_batch_perturbed = self.get_perturbed_input(X_batch)

                o_prediction = np.argmax(self.model.predict(X_batch), axis=1)
                p_prediction = self.model.predict(X_batch_perturbed)

                labels = np.argmax(Y_batch, axis=1)
                predict = np.argmax(p_prediction, axis=1)

                #o_target = (labels == target * np.ones(predict.shape))
                #pre_target = (predict == target * np.ones(predict.shape))

                # cost is the difference
                #attack_success = np.sum(predict == target * np.ones(predict.shape)) - np.sum(o_target & pre_target)
                attack_success = np.sum(predict == target * np.ones(predict.shape))
                #diff = np.sum(labels != predict)
                result = result + attack_success

                o_correct = np.sum(labels == o_prediction)
                correct = correct + o_correct
                tot_count = tot_count + len(labels)
            result = result / tot_count
            correct = correct / tot_count
        return result, correct
