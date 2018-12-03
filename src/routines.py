import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
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
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import flip_axis
from keras.optimizers import adam
import pandas as pd


def generate_training_batch(images, angles, batch_size, is_training):
    """
    yield a batch of data for training
    """
    shadow_params = {
        'alter_prob': 0.5,
        'y_pos': 0.6,
        'delta_y_pos': 0.3,
        's_low': 0.2,
        's_high': 0.2
        }
    while 1:
        batch_images = []
        batch_angles = []
        for i in range(batch_size):
            index = np.random.randint(len(angles))
            img = cv2.imread(images[index])
            angle = angles[index]

            # add random brightness and shadow
            if is_training:
                img = augment_brightness(img)
                img = add_shadow(img)
                #img = add_random_shadow(img, shadow_params)

            # randomly flip the image horizontally
            if is_training and (np.random.randint(2) == 1):
                img = flip_axis(img, 1)
                angle = -angle

            batch_angles.append(angle)
            batch_images.append(img)

        yield (np.array(batch_images), np.array(batch_angles))

def get_log_data(steering_offset, include_camera, dir_name, log_file='driving_log.csv'):
    """
    collect image file names and corresponding steering angles
    """
    images = []
    angles = []
    with open(log_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # skip the header
        for center_img, left_img, right_img, steering_angle, _, _, speed in csv.reader(csvfile):
            center_img = dir_name + '/' + center_img.strip()
            left_img = dir_name + '/' + left_img.strip()
            right_img = dir_name + '/' + right_img.strip()
            steering_angle = float(steering_angle)
            speed = float(speed)

            # check which camera images to include
            camera_images = []
            camera_angles = []
            if include_camera['center']:
                camera_images.append(center_img)
                camera_angles.append(steering_angle)
            if include_camera['left']:
                camera_images.append(left_img)
                camera_angles.append(steering_angle + steering_offset)
            if include_camera['right']:
                camera_images.append(right_img)
                camera_angles.append(steering_angle - steering_offset)

            images.extend(camera_images)
            angles.extend(camera_angles)

    return (images, angles)

def build_carla_training_data(df, steering_offset):
    """
    create a training data after correcting for steering offset
    """
    return (list(df.RGB_file), list(df.steer + steering_offset))

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
                index += 1

    df_dhist = pd.DataFrame(drive_hist)
    df_dhist.to_csv("test.csv", index=False)

    return df_dhist

def add_shadow(image):
    """
    randomly add shadow to the image
    using code from:
    https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py
    """
    image_height, image_width, _ = image.shape
    x1, y1 = image_width * np.random.rand(), 0
    x2, y2 = image_width * np.random.rand(), image_height

    ym, xm = np.mgrid[0:image_height, 0:image_width]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def add_random_shadow(im, params):
    """
    randomly add shadow to the image
    the shadow is cast above or below a line,
    where the line is a deviation from horizontal located
    around y=y_pos and the deviation is delta_y_pos
    using code snippet from:
    https://github.com/naokishibuya/car-behavioral-cloning/blob/master/utils.py
    """
    # keep image unchanged with probability of altering the image
    if np.random.randint(2) > params['alter_prob']:
        return im

    image_height, image_width, _ = im.shape
    f = params['y_pos'] #0.6
    df = params['delta_y_pos'] #0.3 
    r1 = (np.random.randint(2) * 2 - 1)
    r2 = (np.random.randint(2) * 2 - 1)

    x1, y1 = 0, image_height * f + image_height * np.random.rand() * df * r1
    x2, y2 = image_width, image_height * f + image_height * np.random.rand() * df * r2

    s_low = params['s_low'] #0.2 
    s_high = params['s_high'] #0.2 

    ym, xm = np.mgrid[0:image_height, 0:image_width]
    mask = np.zeros_like(im[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=s_low, high=s_high)
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def augment_brightness(image):
    """
    randomly change brightness of the image
    using code from:
    https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.aq3jet38c
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def model_comma_ai(camera_format, crop=None):
    """
    model developed by comma.ai: https://github.com/commaai/research
    """
    row, col, col_chan = camera_format
    if crop is None:
        crop_top, crop_bot, crop_left, crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bot), (crop_left, crop_right)),
                         input_shape=(row, col, col_chan),
                         data_format="channels_last"
                        ))
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row_cropped, col_cropped, col_chan),
                     output_shape=(row_cropped, col_cropped, col_chan)))

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
    row, col, col_chan = camera_format
    if crop is None:
        crop_top = crop_bot = crop_left = crop_right = 0
    else:
        crop_top, crop_bot, crop_left, crop_right = crop
    row_cropped = row - crop_top - crop_bot
    col_cropped = col - crop_left - crop_right

    model = Sequential()
    model.add(Cropping2D(cropping=((crop_top, crop_bot), (crop_left, crop_right)),
                         input_shape=(row, col, col_chan),
                         data_format="channels_last"
                        ))

    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=(row_cropped, col_cropped, col_chan),
                     output_shape=(row_cropped, col_cropped, col_chan))
             )

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

    #adam_opt = adam(lr=0.001, decay=1.0e-6)
    adam_opt = adam(lr=2e-4, decay=1.0e-6)

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
        train_generator = generate_training_batch(train_X, train_y, n_batch, is_training=True)
        validation_generator = generate_training_batch(val_X, val_y, n_batch, is_training=False)
        checkpointer = ModelCheckpoint(filepath=check_path, verbose=1, save_best_only=True)

        history = model.fit_generator(train_generator,
                                      steps_per_epoch=num_train,
                                      validation_data=validation_generator,
                                      validation_steps=num_valid,
                                      callbacks=[checkpointer],
                                      epochs=epochs
                                     )
    else:
        train_generator = generate_training_batch(X, y, batch_size=n_batch, is_training=True)
        history = model.fit_generator(train_generator,
                                      steps_per_epoch=num_samples,
                                      epochs=epochs
                                     )

    return model, history

def tune_model(args):
    """
    tune model to pick the best steering offset
    """
    dir_name = args['data_dir']
    log_file = args['log_file']
    model_desc = 'nvidia_model_for_mountain'
    history_path = 'tune_history/UDACITY_MOUNTAIN/'

    image_dim = (160, 320, 3)
    image_crop = (60, 20, 0, 0)
    n_sample = 20000
    n_epochs = 40
    batch_size = 32
    offset_range = [0.199, 0.211, 0.221]
    include_camera = {'center': True, 'left': True, 'right': True}

    for steering_offset in offset_range:
        data = get_log_data(steering_offset, include_camera, dir_name, log_file)
        print('steering offset = ', steering_offset)
        gpu_count = int(args['gpu_count'])
        if gpu_count > 1:
            model = model_nvidia(image_dim, crop=image_crop, gpu_count=gpu_count)
        model_name = model_desc + '_steer_' + str(steering_offset) + '.h5'
        model, history = train_model(model, data, epochs=n_epochs, n_batch=batch_size,
                                     num_samples=n_sample, validate=True,
                                     check_path=history_path+model_name)
        training_history = history_path + model_desc + '_steer_' + str(steering_offset) + '.pkl'
        print("saving the history in %s" % training_history)
        with open(training_history, 'wb') as fid:
            pickle.dump((history.history['loss'], history.history['val_loss']), fid)

def tune_model_carla(args):
    """
    tune model to pick the best steering offset
    """
    dir_name = args['data_dir']
    log_file = args['log_file']
    model_desc = 'nvidia_model_for_carla'
    history_path = 'tune_history/CARLA/'

    carla_image_dim = (600, 800, 3)
    carla_image_crop = (200, 100, 0, 0)
    n_sample = 1000
    n_epochs = 5
    batch_size = 32
    offset_range = [0.0, -0.01, 0.01]

    for steering_offset in offset_range:
        data_df = collect_carla_log_data(dir_name, log_file)
        data = build_carla_training_data(data_df, steering_offset)
        print('steering offset = ', steering_offset)
        gpu_count = int(args['gpu_count'])
        if gpu_count > 1:
            model = model_nvidia(carla_image_dim, crop=carla_image_crop, gpu_count=gpu_count)
        model_name = model_desc + '_steer_' + str(steering_offset) + '.h5'
        model, history = train_model(model, data, epochs=n_epochs, n_batch=batch_size,
                                     num_samples=n_sample, validate=True,
                                     check_path=history_path+model_name)
        training_history = history_path + model_desc + '_steer_' + str(steering_offset) + '.pkl'
        print("saving the history in %s" % training_history)
        with open(training_history, 'wb') as fid:
            pickle.dump((history.history['loss'], history.history['val_loss']), fid)

def train_single_model(args):
    """
    train a model with supplied parameters
    """
    dir_name = args['data_dir']
    log_file = 'driving_log.csv'
    n_sample = 1000
    n_epochs = 60
    batch_size = 32
    best_steering_offset = 0.3 # 0.4
    include_camera = {'center': True, 'left': False, 'right': False}

    data = get_log_data(best_steering_offset, include_camera, dir_name, log_file)
    if args['preload_model'] is not None:
        try:
            model = load_model(args['preload_model'])
        except:
            print("cannot find model to load")
    else:
        model = model_nvidia((160, 320, 3), crop=(50, 20, 0, 0))

    model, history = train_model(model, data, epochs=n_epochs, n_batch=batch_size, num_samples=n_sample, validate=True)
    print("saving the model in %s" % args['save_model'])
    model.save(args['save_model'])
    print("saving the history in %s" % args['save_history'])
    print(history.history['loss'])
    print(history.history['val_loss'])
    with open(args['save_history'], 'wb') as fid:
        pickle.dump((history.history['loss'], history.history['val_loss']), fid)
