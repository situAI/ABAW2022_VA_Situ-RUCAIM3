import h5py
import os, glob

def get_dim(feat_name):
    dim_dict = {
        'denseface': 342,
        'vggface2': 2048,
        'vggish': 128,
        'compare':130,
        'wav2vec': 768,
        'egemaps': 23,
        'affectnet': 342,
        'FAU': 35,
        'head_pose':3,
        'eye_gaze': 120,
        'FAU_situ': 512
    }
    if dim_dict.get(feat_name) is not None:
        return dim_dict[feat_name]
    else:
        return dim_dict[feat_name.split('_')[0]]

def calc_total_dim(feature_set):
    return sum(map(lambda x: get_dim(x), feature_set))

def get_each_dim(feature_set):
    return list(map(lambda x: get_dim(x), feature_set))

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)