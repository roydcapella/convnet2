"""
jsaavedr, 2020
Modified by rcapella, 2020
Before using this program, set the path where the folder "covnet2"  is stored.
To use train.py, you will require to send the following parameters :
 * -config : A configuration file where a set of parameters for data construction and trainig is set.
 * -name: A section name in the configuration file.
 * -mode: [train, test] for training, testing, or showing  variables of the current model. By default this is set to 'train'
 * -save: Set true for saving the model
"""

import sys
import os
import tensorflow as tf

from models import resnet
from models import simple
from models import alexnet
import datasets.data as data
import utils.configuration as conf
import utils.losses as losses
import utils.imgproc as imgproc
import utils.metrics as metrics
import numpy as np
import argparse
import os
import matplotlib as plt
import pickle

##imp
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


sys.path.append(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__' :        
    parser = argparse.ArgumentParser(description = "Train many model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)
    parser.add_argument("-mode", type=str, choices=['train', 'test', 'predict'],  help=" train or test", required = False, default = 'train')
    parser.add_argument("-arch", type=str, choices=['resnet', 'alexnet'],  help=" resnet or alexnet", required = False, default = 'resnet')
    parser.add_argument("-method", type=str, choices=['sgd', 'adam', 'tl', ],  help="sgd, adam or tl", required = False, default = 'sgd')
    parser.add_argument("-save", type= bool,  help=" True to save the model", required = False, default = False)    
    pargs = parser.parse_args()     
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)                   
    now = datetime.now().strftime("%Y%m%d-%H%M")

    if pargs.mode == 'train' :
        tfr_train_file = os.path.join(configuration.get_data_dir(), "train.tfrecords")
    if pargs.mode == 'train' or  pargs.mode == 'test':    
        tfr_test_file = os.path.join(configuration.get_data_dir(), "test.tfrecords")
    if configuration.use_multithreads() :
        if pargs.mode == 'train' :
            tfr_train_file=[os.path.join(configuration.get_data_dir(), "train_{}.tfrecords".format(idx)) for idx in range(configuration.get_num_threads())]
        if pargs.mode == 'train' or  pargs.mode == 'test':    
            tfr_test_file=[os.path.join(configuration.get_data_dir(), "test_{}.tfrecords".format(idx)) for idx in range(configuration.get_num_threads())]        
    sys.stdout.flush()

    saved_to = os.path.join(configuration.get_data_dir(), "models", pargs.arch, pargs.method , now)
    checkpoints_path = os.path.join(saved_to, "checkpoints")
    mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
    shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")

    input_shape = np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)
    number_of_classes = configuration.get_number_of_classes()
    print ("Initializing {} with {} in {} in mode {} ".format(pargs.name, pargs.arch, pargs.method, pargs.mode))
    
    # loading tfrecords into dataset object
    if pargs.mode == 'train':
        tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
        tr_dataset = tr_dataset.map(
            lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation=True))
        tr_dataset = tr_dataset.shuffle(configuration.get_shuffle_size())
        tr_dataset = tr_dataset.batch(batch_size=configuration.get_batch_size())
        # tr_dataset = tr_dataset.repeat()
    
    if pargs.mode == 'train' or pargs.mode == 'test':
        val_dataset = tf.data.TFRecordDataset(tfr_test_file)
        val_dataset = val_dataset.map(
            lambda x: data.parser_tfrecord(x, input_shape, mean_image, number_of_classes, with_augmentation=False))
        val_dataset = val_dataset.batch(batch_size=configuration.get_batch_size())
 
    # Defining callback for saving checkpoints
    # save_freq: frecuency in terms of number steps each time checkpoint is saved
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath= checkpoints_path + '/{epoch:03d}.h5',
        save_weights_only=True,
        mode='max',
        monitor='val_acc',
        save_freq='epoch',
    )

    #Applying network architecture and optimization methods
    if pargs.arch == 'resnet':
        model = resnet.ResNet([3,4,6,3],[64,128,256,512], configuration.get_number_of_classes(), use_bottleneck = True)
        process_fun = imgproc.process_image
        input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name = 'input_image')     
        model(input_image)    
        model.summary()

    if pargs.arch == 'alexnet':
        model = alexnet.AlexNetModel(configuration.get_number_of_classes())
        process_fun = imgproc.process_image
        input_image = tf.keras.Input((input_shape[0], input_shape[1], input_shape[2]), name='input_image')
        model(input_image)
        model.summary()

    #aplicando pesos
    if configuration.use_checkpoint():
        model.load_weights(configuration.get_checkpoint_file(), by_name=True, skip_mismatch=True)

    #configurando optimizador
    if (pargs.method == 'sgd') :
        lr_schedule = tf.keras.experimental.CosineDecay(initial_learning_rate = configuration.get_learning_rate(),
                                                decay_steps = configuration.get_decay_steps(),
                                                alpha = 0.0001)                                       
        opt = tf.keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9, nesterov = True)
  
    if (pargs.method == 'adam'):    
        opt = tf.keras.optimizers.Adam(lr=configuration.get_learning_rate(), epsilon=1e-08)

    #Compile model
    model.compile(optimizer=opt, loss= losses.crossentropy_loss, metrics=['accuracy', metrics.simple_accuracy])
    
    if pargs.mode == 'train':                             
        trainning = model.fit(tr_dataset, 
            epochs = configuration.get_number_of_epochs(),                        
            validation_data=val_dataset,
            validation_steps = configuration.get_validation_steps(),
            callbacks=[model_checkpoint_callback])

        plt.figure(figsize=(20,5))
        plt.suptitle(pargs.arch + "-" + pargs.method + "-" + now)

        print ("Plotting Acurracy")
        plt.subplot(1,2,2)
        plt.xlabel('# Epocas')
        plt.legend(loc="upper left", title="Accuracy", frameon=False)
        plt.plot(trainning.history['accuracy'], label ='train_accuracy')
        plt.plot(trainning.history['val_accuracy'], label ='val_accuracy')

        print ("Plotting Loss")
        plt.subplot(1,2,1)
        plt.xlabel('# Epocas')
        plt.legend(loc="upper right", title="Loss", frameon=False)
        plt.plot(trainning.history['loss'], label ='train_loss')
        plt.plot(trainning.history['val_loss'], label ='val_loss')
        plt.show()

        filename = saved_to + "/training.txt"
        if not os.path.exists(saved_to):
            os.makedirs(saved_to)
        with open(filename, 'wb') as pyfile:  
            pickle.dump(trainning.history, pyfile)
        print("trainning historial saved in {}".format(filename))  

    elif pargs.mode == 'test' :
        model.evaluate(val_dataset, steps = configuration.get_validation_steps())
    
    elif pargs.mode == 'predict':
        filename = input('file :')
        while(filename != 'end') :
                target_size = (configuration.get_image_height(), configuration.get_image_width())
                image = process_fun(data.read_image(filename, configuration.get_number_of_channels()), target_size )
                image = image - mean_image
                image = tf.expand_dims(image, 0)        
                pred = model.predict(image)
                pred = pred[0]
                pred = np.exp(pred - max(pred))
                pred = pred / np.sum(pred)            
                cla = np.argmax(pred)
                print(pred)                                               
                print(cla)
                filename = input('file :')
    #save the model   
    if pargs.save:
        model.save(saved_to)
        print("model saved to {}".format(saved_to))