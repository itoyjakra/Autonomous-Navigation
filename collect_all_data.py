import os
from shutil import copyfile
import sys
import glob
import pandas as pd

def get_dirnames(pat):
    names = glob.glob(pat + '*')
    return names

def create_logfile(dirs, log_name='driving_log.csv', img_dir='IMG'):
    col_names = ['center', 'left', 'right', 'steer', 'junk1', 'junk2', 'speed']
    all_logs = []
    file_names = []
    for dir_name in dirs:
        file_name = dir_name + '/' + log_name
        df_log = pd.read_csv(file_name, header=None, names=col_names)

        # collect all the paths
        file_names.append(df_log[['center', 'left', 'right']])

        # remove the path from the file names
        df_log.center = df_log.center.apply(lambda x: img_dir + '/' + x.split('/IMG/')[-1])
        df_log.left = df_log.left.apply(lambda x: img_dir + '/' + x.split('/IMG/')[-1])
        df_log.right = df_log.right.apply(lambda x: img_dir + '/' + x.split('/IMG/')[-1])

        print('shape of {} = {}'.format(dir_name, df_log.shape))
        all_logs.append(df_log)

    # create the combined log file
    df_alllogs = pd.concat(all_logs, axis=0, ignore_index=True)
    print('size of training data = {}'.format(len(df_alllogs)))
    df_alllogs.to_csv(log_name, header=False, index=False)

    # collect the image file names
    df_imfiles = pd.concat(file_names, axis=0, ignore_index=True)

    return df_imfiles

def move_images_new(df, pat, img_dir='IMG'):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    data_dirs = get_dirnames(pat)
    colnames = df.columns
    df['subdir'] = df[colnames[0]].apply(lambda x: get_subdir(x, pat))

    pass

def get_subdir(name, pat):
    data_dirs = get_dirnames(pat)
    for data_dir in data_dirs:
        if data_dir in name:
            return data_dir
    return None

def move_images(df, pat, img_dir='IMG'):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    data_dirs = get_dirnames(pat)
    colnames = df.columns
    df['subdir'] = df[colnames[0]].apply(lambda x: get_subdir(x, pat))

    for index in df.index:
        for col in colnames:
            src_file = df.loc[index][col]
            subdir = df.loc[index].subdir
            print(index, src_file, subdir)
            print('---')
            splits = src_file.split(subdir)
            src_file = subdir + splits[-1]
            tgt_file = splits[-1][1:]
            print('src = ', src_file)
            print('tgt = ', tgt_file)
            copyfile(src_file, tgt_file)

def main():
    pattern = 'AV_'
    all_dirs = get_dirnames(pattern)
    print(all_dirs)
    df_ims = create_logfile(all_dirs)
    move_images(df_ims, pattern)

if __name__ == "__main__":
    main()
