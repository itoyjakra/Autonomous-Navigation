import os
import csv
import cv2
import _pickle as pickle
from keras import initializers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Cropping2D, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
#from keras.preprocessing.image import flip_axis
from tensorflow.keras.preprocessing.image import flip_axis
from keras.optimizers import adam
import pandas as pd


def generate_training_batch(images, angles, batch_size, image_augment=True):
    """
    yield a batch of data for training
    """
    while 1:
        batch_images = []
        batch_angles = []
        for i in range(batch_size):
            index = np.random.randint(len(angles))
            img = cv2.imread(images[index])
            # add random brightness
            if image_augment:
                img = augment_brightness_camera_images(img)
            angle = angles[index]
            # randomly flip the image horizontally
            if np.random.randint(2) == 1:
                img = flip_axis(img, 1)
                angle = -angle
            batch_angles.append(angle)
            batch_images.append(img)

        yield (np.array(batch_images), np.array(batch_angles))

def get_log_data(steering_offset=0.3, include_center=True, dir_name='/ARTIFACTS/Autonomous-Navigation/DATA/UDACITY', log_file='driving_log.csv'):
    """
    collect image file names and corresponding steering angles
    """
    images = []
    angles = []
    with open(dir_name+'/'+log_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip the header
        for center_img, left_img, right_img, steering_angle, _, _, speed in csv.reader(csvfile):
            center_img = dir_name + '/' + center_img.strip()
            left_img = dir_name + '/' + left_img.strip()
            right_img = dir_name + '/' + right_img.strip()
            steering_angle = float(steering_angle)
            if include_center:
                images.extend([center_img, left_img, right_img])
                angles.extend([steering_angle, steering_angle+steering_offset, steering_angle-steering_offset])
            else:
                images.extend([left_img, right_img])
                angles.extend([steering_angle+steering_offset, steering_angle-steering_offset])

    return (images, angles)

def collect_carla_log_data(data_dir, log_file):
    """
    CARLA: collect image file names and corresponding steering angles
    """
    index = 0
    drive_hist = []
    log = data_dir + '/' + log_file
    with open(log) as f:
        for line in f:
            if "Vehicle" in line:
                s1 = line.split('),')[0]
                co = s1.split('at (')[-1]
                x = float(co.split(',')[0])
                y = float(co.split(',')[-1])
                
                n = int(line.split('/')[-1])
                params = {'x': x, 'y': y, 'n': n}
                
            if "steer" in line:
                steer = float(line.split('steer:')[-1])
                params['steer'] = steer
                params['RGB_file'] = '{0}/episode_0000/CameraRGB/{1:06d}.png'.format(data_dir, n)
                drive_hist.append(params)
                index +=1 

    df_dhist = pd.DataFrame(drive_hist)
    df_dhist.to_csv("test.csv", index=False)

    return df_dhist

def augment_brightness_camera_images(image):
    """
    randomly change brightness of the image
    using code from: 
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.aq3jet38c
    """
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def model_comma_ai(camera_format, crop=None):
    """
    model developed by comma.ai: https://github.com/commaai/research
    """
    row, col, ch = camera_format
    if crop is None:
        crop_top, crop_bot, crop_left, crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping = ((crop_top, crop_bot), (crop_left, crop_right)), input_shape = (row, col, ch), data_format = "channels_last"))
    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(row_cropped, col_cropped, ch),
                        output_shape=(row_cropped, col_cropped, ch)))

    model.add(Convolution2D(16, (8, 8), strides=(4, 4), padding="same"))
    model.add(ELU())

    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(ELU())

    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())

    model.add(Dropout(0.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model

def model_nvidia(camera_format, crop=None, gpu_count=1):
    """
    model developed by nvidia:
    https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    """
    row, col, ch = camera_format
    if crop is None:
        crop_top = crop_bot = crop_left = crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping = ((crop_top, crop_bot), (crop_left, crop_right)), input_shape = (row, col, ch), data_format = "channels_last"))

    model.add(Lambda(lambda x: x/127.5 - 1.,
                        input_shape=(row_cropped, col_cropped, ch),
                        output_shape=(row_cropped, col_cropped, ch)))

    model.add(Convolution2D(24, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())

    model.add(Convolution2D(64, (3, 3), strides=(1, 1), padding="valid"))
    model.add(ELU())
    model.add(Flatten())

    model.add(Dense(100, kernel_initializer=initializers.glorot_normal()))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(50, kernel_initializer=initializers.glorot_normal()))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(10, kernel_initializer=initializers.glorot_normal()))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(1))

    adam_opt = adam(lr=0.001, decay=1.0e-6)

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    model.compile(optimizer=adam_opt, loss="mse")

    return model

def train_model(model, data, epochs=3, n_batch=32, validate=False, num_samples=10000, check_path='dump.hd5'):
    """
    train model on supplied data
    """
    X, y = data
    num_train = 0.8*num_samples
    num_valid = 0.2*num_samples

    if validate:
        train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2)
        train_generator = generate_training_batch(train_X, train_y, batch_size=n_batch)
        validation_generator = generate_training_batch(val_X, val_y, batch_size=n_batch)
        checkpointer = ModelCheckpoint(filepath=check_path, verbose=1, save_best_only=True)

        history = model.fit_generator(train_generator,
                        steps_per_epoch=num_train,
                        validation_data=validation_generator,
                        validation_steps=num_valid,
                        callbacks=[checkpointer],
                        epochs=epochs)
    else:
        train_generator = generate_training_batch(X, y, batch_size=n_batch)
        history = model.fit_generator(train_generator, 
                        steps_per_epoch=num_samples,
                        epochs=epochs)

    return model, history

def tune_model(args):
    """
    tune model to pick the best steering offset
    """
    #dir_name='Udacity_Data/data'
    dir_name = args['data_dir']
    log_file='driving_log.csv'
    n_sample = 10000
    n_epochs = 5
    batch_size = 32
    offset_range = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    for steering_offset in offset_range:
        data = get_log_data(steering_offset=steering_offset, dir_name=dir_name, log_file=log_file, include_center=True)
        print ('steering offset = ', steering_offset)
        gpu_count = int(args['gpu_count'])
        if gpu_count > 1:
            model = model_nvidia((160, 320, 3), crop=(50, 20, 0, 0), gpu_count=gpu_count)
        #model_name = 'nvidia_model' + '_drop_0.5_steer_' + str(steering_offset) + '.h5'
        model_name = 'nvidia_model' + '_steer_' + str(steering_offset) + '.h5'
        model, history = train_model(model, data, epochs=n_epochs, n_batch=batch_size, 
                num_samples=n_sample, validate=True, check_path='tune_history/'+model_name)
        training_history = 'tune_history/nvidia_model' + '_steer_' + str(steering_offset) + '.pkl'
        print ("saving the history in %s" % training_history)
        with open(training_history, 'wb') as fid:
            pickle.dump((history.history['loss'], history.history['val_loss']), fid)

def train_single_model(args):
    """
    train a model with supplied parameters
    """
    #dir_name='Udacity_Data/data'
    dir_name = args['data_dir']
    log_file='driving_log.csv'
    n_sample = 1000
    n_epochs = 60
    batch_size = 32
    best_steering_offset = 0.3 # 0.4

    data = get_log_data(steering_offset=best_steering_offset, dir_name=dir_name, log_file=log_file, include_center=True)
    if args['preload_model'] is not None:
        try:
            model = load_model(args['preload_model'])
        except:
            print ("cannot find model to load")
    else:
        model = model_nvidia((160, 320, 3), crop=(50, 20, 0, 0))

    model, history = train_model(model, data, epochs=n_epochs, n_batch=batch_size, num_samples=n_sample, validate=True)
    print ("saving the model in %s" % args['save_model'])
    model.save(args['save_model'])
    print ("saving the history in %s" % args['save_history'])
    print (history.history['loss'])
    print (history.history['val_loss'])
    with open(args['save_history'], 'wb') as fid:
        pickle.dump((history.history['loss'], history.history['val_loss']), fid)
