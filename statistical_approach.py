import os
import time, sys
import math
import random
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

import tensorflow as tf
import numpy as np
import torch

class HParams(object):
    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)

    def update_config(self, in_string):
        pairs = in_string.split(",")
        pairs = [pair.split("=") for pair in pairs]
        for key, val in pairs:
            self.dict_[key] = type(self.dict_[key])(val)
        self.__dict__.update(self.dict_)
        return self

    def __getitem__(self, key):
        return self.dict_[key]

    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)


def get_default_hparams():
    return HParams(
        image_dim = 28,
        epoch_num = 50,
        batch_size = 1000,
        learning_rate = 1e-3,
        next_batch = 0)

def conv_pool(inputs, filters=32, kernel_size=[5,5], pool_size=[2,2]):
    conv = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.relu)
    pool = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_size, strides=2)
    return pool

def dense_layer(inputs, reshape=(7,7,64), units=1024, drop_out=True, drop_out_rate=0.4, training=True):
    inputs = tf.reshape(inputs, [-1, reshape[0]*reshape[1]*reshape[2]])
    dense = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=drop_out_rate, training=training)
    return dropout

def cnn_model(inputs, training=True):
    with tf.variable_scope("cnn_model", reuse=tf.AUTO_REUSE):
        inputs = tf.expand_dims(inputs, 3)
        conv1 = conv_pool(inputs)
        conv2 = conv_pool(conv1, filters=64)
        dense = dense_layer(conv2, training=training)
        logits = tf.layers.dense(inputs=dense, units=10)
    return logits

def get_batch(data, batch_num, batch_size=100):
    if batch_size == 0:
        raise ValueError("batch_size cannot be zero")
    elif batch_num < 0:
        raise ValueError("batch_num must be larger than or equal to zero")
    elif data is np.ndarray:
        raise TypeError("input data should be numpy.ndarray")
    elif data.shape[0] < batch_size:
        raise TypeError("please reduce the batch_size less than the data number")
    number_of_batch = data.shape[0]//batch_size
    if batch_num >= number_of_batch:
        end_of_batch = True
    else:
        end_of_batch = False
    batch_num = batch_num % number_of_batch
    return data[batch_num*batch_size:(batch_num+1)*batch_size], end_of_batch

def training(train_x, train_y, hps, 
             restore=False, save=False, clean_graph=False, 
             sess=tf.Session(), path="./trained_model", filename="model.ckpt"):
    inputs = tf.placeholder(tf.float32, shape=(None, hps.image_dim, hps.image_dim))
    labels = tf.placeholder(tf.int32, shape=(None))
    logits = cnn_model(inputs)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate).minimize(loss)
    saver = tf.train.Saver()

    sess.run([tf.global_variables_initializer()])
    if restore:
        saver.restore(sess, "{0}/{1}".format(path, filename))

    for epoch in range(hps.epoch_num):
        next_batch = hps.next_batch
        while True:
            train_data, end_of_batch = get_batch(train_x, batch_size=hps.batch_size, batch_num=next_batch)
            train_labels, _ = get_batch(train_y, batch_size=hps.batch_size, batch_num=next_batch)
            next_batch += 1
            sess.run(optimizer, feed_dict={inputs: train_data, labels: train_labels})
            loss_run_time = sess.run(loss, feed_dict={inputs: train_data, labels: train_labels})
            if end_of_batch:
                break
        print("Epoch: {}, Loss:{}".format(epoch, loss_run_time))
    if save:
        save_path = saver.save(sess, "{0}/{1}".format(path, filename))
        print("Model is saved to {0}".format(save_path))
        
def evaluation(eval_x, eval_y, hps, 
               sess=tf.Session(),
               path="./trained_model", filename="model.ckpt", print_output=True):
    inputs = tf.placeholder(tf.float32, shape=(None, hps.image_dim, hps.image_dim))
    labels = tf.placeholder(tf.int32, shape=(None))
    logits = cnn_model(inputs, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    saver = tf.train.Saver()
    sess.run([tf.global_variables_initializer()])
    saver.restore(sess, "./trained_model/model.ckpt")
    loss_run_time, logits_run_time = sess.run([loss, logits], feed_dict={inputs: eval_x, labels: eval_y})
    accuracy = 100 * np.sum(np.argmax(logits_run_time, axis=1) == eval_y) / eval_y.shape[0]
    if print_output:
        print("Loss:{}, Accuracy:{}%".format(loss_run_time, accuracy))
    return logits_run_time, eval_y

def evaluation_model(hps, 
               sess=tf.Session(),
               path="./trained_model", 
               filename="model.ckpt"):
    inputs = tf.placeholder(tf.float32, shape=(None, hps.image_dim, hps.image_dim))
    labels = tf.placeholder(tf.int32, shape=(None))
    logits = cnn_model(inputs, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    saver = tf.train.Saver()
    sess.run([tf.global_variables_initializer()])
    saver.restore(sess, "./trained_model/model.ckpt")
    return inputs, labels, logits, loss

def uniform_perturbation(sample, x_min, x_max, n=1, sigma=1, seed=0, image_size=28):
    import torch
    import torch.distributions as dist
    sample = torch.tensor(sample).view(-1, image_size*image_size)
    if isinstance(x_min, (int, float, complex)) and isinstance(x_max, (int, float, complex)):
        prior = dist.Uniform(low=torch.max(sample-sigma, torch.tensor([x_min])), high=torch.min(sample+sigma, torch.tensor([x_max])))
    elif isinstance(x_min, torch.Tensor) and isinstance(x_max, torch.Tensor):
        prior = dist.Uniform(low=torch.max(sample-sigma, x_min), high=torch.min(sample+sigma, x_max))
    else:
        raise ValueError('Type of x_min and x_max {0} is not supported'.format(type(x_min)))
    x = prior.sample(torch.Size([n])).view(-1, image_size, image_size)
    return x.numpy(), prior

def compute_property_function(logits, labels):
    equal_to_labels = logits.argmax(axis=1) == labels
    correct_logits = np.array([logits[idx][label] for idx, label in enumerate(labels)]) 
    z_c = np.exp(correct_logits)/np.exp(logits).sum(axis=1)
    logits_rankings = np.argsort(logits, axis=1)
    best_logits_without_labels = np.array([logit[logits_ranking[-2]] if equal_to_label else logit[logits_ranking[-1]] 
                                           for logit, logits_ranking, equal_to_label in zip(logits, logits_rankings, equal_to_labels)])
    z_i = np.exp(best_logits_without_labels)/np.exp(logits).sum(axis=1)
    return z_i - z_c

def get_s_xn(samples, input_labels):
    tf.reset_default_graph()
    with tf.Session() as sess:
        inputs, labels, logits, loss = evaluation_model(get_default_hparams(), sess=sess)
        loss_run_time, logits_run_time = sess.run([loss, logits], feed_dict={inputs: samples, labels: input_labels})
        accuracy = 100 * np.sum(np.argmax(logits_run_time, axis=1) == perturbated_labels) / perturbated_labels.shape[0]
        s_xn = compute_property_function(logits_run_time, perturbated_labels)
    return s_xn

def mh_update(samples, sample_labels, width_proposal, prior, l_k, inputs, labels, 
              width_inc=1.02, width_dec=0.5, update_steps=250, sess=tf.Session()):
    #Take reference to the original code from the paper, which provides a efficient method for MH process
    import torch
    import torch.distributions as dist
    sample_size = samples.shape[0]
    x = torch.tensor(samples).view(-1, 28*28)
    acc_ratio = torch.zeros(sample_size)
    for i in range(update_steps):
        g_bottom = dist.Uniform(low=torch.max(x - width_proposal.unsqueeze(-1), prior.low), high=torch.min(x + width_proposal.unsqueeze(-1), prior.high))
        x_new = g_bottom.sample()
        loss_run_time, logits_run_time = sess.run([loss, logits], feed_dict={inputs: x_new.view(-1, 28, 28).numpy(), 
                                                                             labels: sample_labels})
        s_xn = compute_property_function(logits_run_time, sample_labels)
        
        g_top = dist.Uniform(low=torch.max(x_new - width_proposal.unsqueeze(-1), prior.low), high=torch.min(x_new + width_proposal.unsqueeze(-1), prior.high))
        lg_alpha = (prior.log_prob(x_new) - prior.log_prob(x)+ g_top.log_prob(x) - g_bottom.log_prob(x_new)).sum(dim=1)
        acceptance = torch.min(lg_alpha, torch.zeros_like(lg_alpha))
        
        log_u = torch.log(torch.rand_like(acceptance))
        acc_idx = torch.tensor((log_u <= acceptance).numpy() & (s_xn >= l_k))
        acc_ratio += acc_idx.float()
        x = torch.where(acc_idx.unsqueeze(-1), x_new, x)
    width_proposal = torch.where(acc_ratio > 0.124, width_proposal*width_inc, width_proposal)
    width_proposal = torch.where(acc_ratio < 0.124, width_proposal*width_dec, width_proposal)  
    return x.view(-1, 28, 28).numpy(), width_proposal
  
result_dict = dict()
    
((train_x, train_y),(eval_x, eval_y)) = tf.keras.datasets.mnist.load_data()
train_x = train_x/np.float32(255)
eval_x = eval_x/np.float32(255)    

# The bounds in NN-space
x_min = (0-0.1307)/0.3081
x_max = (1-0.1307)/0.3081

sample_idx = random.randrange(0, eval_x.shape[0])
sample_x, sample_y = eval_x[sample_idx], eval_y[sample_idx]

result_dict['sample_idx'] = sample_idx

sigmas = [0.5, 0.8, 1.1, 1.4]
M = [100, 250]

#Naive MC
result_dict['naive'] = dict()
for sigma in sigmas:
    tf.reset_default_graph()
    accuracy_list = []
    step = 0
    sample_size = 10000
    repeat = 3000
    with tf.Session() as sess:
        inputs, labels, logits, loss = evaluation_model(get_default_hparams(), sess=sess)
        for i in range(repeat):
            perturbated_samples, _ = uniform_perturbation(sample_x, x_min, x_max, n=sample_size, sigma=sigma)
            perturbated_labels = np.repeat(sample_y, sample_size)
            loss_run_time, logits_run_time = sess.run([loss, logits], feed_dict={inputs: perturbated_samples, labels: perturbated_labels})
            accuracy = 100 * np.sum(np.argmax(logits_run_time, axis=1) == perturbated_labels) / perturbated_labels.shape[0]
            accuracy_list += [accuracy/100]
    error_list = (1 - np.array(accuracy_list))
    log_i = np.log(error_list.mean())
    result_dict['naive'][sigma] = log_i


#Naive AMLS
result_dict['AMLS'] = dict()
for sigma in sigmas:
    for m in M:
        sample_size = 1000
        perturbated_samples, prior = uniform_perturbation(sample_x, x_min, x_max, n=sample_size, sigma=sigma)
        perturbated_labels = np.repeat(sample_y, sample_size)
        quantile = 0.1
        log_p_min = -250
        l_k = float('-inf')
        l_prev = float('-inf')
        log_i = 0
        k = 0
        width_proposal = sigma*torch.ones(sample_size)/30

        tf.reset_default_graph()
        log_i_record = []
        with tf.Session() as sess:
            inputs, labels, logits, loss = evaluation_model(get_default_hparams(), sess=sess)
            while l_k < 0:
                loss_run_time, logits_run_time = sess.run([loss, logits], feed_dict={inputs: perturbated_samples, labels: perturbated_labels})
                accuracy = 100 * np.sum(np.argmax(logits_run_time, axis=1) == perturbated_labels) / perturbated_labels.shape[0]
                s_xn = compute_property_function(logits_run_time, perturbated_labels)
                l_k = min(0, np.quantile(s_xn, quantile, interpolation='lower'))
                if l_k == l_prev:
                    break
                l_prev = l_k
                p_k = np.where(s_xn>= l_k)[0].shape[0] / sample_size
                log_i += math.log(p_k)
                if log_i < log_p_min:
                    break
                perturbated_samples = perturbated_samples[s_xn>=l_k]
                s_xn = s_xn[s_xn>=l_k]
                resample_idx = np.random.choice(s_xn.shape[0], sample_size)
                perturbated_samples = perturbated_samples[resample_idx]
                s_xn = s_xn[resample_idx]
                logits_run_time = logits_run_time[resample_idx]
                perturbated_samples, width_proposal = mh_update(perturbated_samples, perturbated_labels, width_proposal, prior, 
                                                                l_k, inputs, labels, update_steps=m, sess=sess)
              
        result_dict['AMLS'][(sigma, m)] = l_k

with open('./result.pkl', 'wb') as f:
    pickle.dump(result_dict, f)

