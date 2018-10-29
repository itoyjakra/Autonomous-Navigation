import argparse
import routines as ro

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drive car autonomously')
    parser.add_argument('-s','--save_model', help='Save the trained model', default='model_dump.h5')
    parser.add_argument('-y','--save_history', help='Save the training history', default='history_dump.pkl')
    parser.add_argument('-t', '--tune_model', action='store_true')
    parser.add_argument('-p', '--preload_model', default=None)
    parser.add_argument('-d', '--data_dir', help='directory for training data', default=None)
    parser.add_argument('-g', '--gpu_count', help='number of gpus', default=1)
    parser.add_argument('-r', '--simulator', help='choose between udacity and carla', default='carla')
    parser.add_argument('-l', '--log_file', help='log file for training data images', default=None)
    args = vars(parser.parse_args())


    data_dir = args['data_dir']
    log_file = args['log_file']
    if  args['simulator']=='carla':
        ro.collect_carla_log_data(data_dir, log_file)
    assert 1==2

    if  args['tune_model']:
        ro.tune_model(args)
    else:
        ro.train_single_model(args)
