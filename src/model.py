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
    args = vars(parser.parse_args())

    if  args['tune_model']:
        ro.tune_model(args)
    else:
        ro.train_single_model(args)
