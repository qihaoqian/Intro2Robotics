import sys
import time
import pickle

def tic():
    return time.time()

def toc(tstart, nm=""):
    print(f'{nm} took: {time.time() - tstart} sec.\n')

def read_data(fname):
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            data = pickle.load(f)
        else:
            data = pickle.load(f, encoding='latin1')
    return data

def load_imu_dataset(dataset_num):
    ifile = f"data/imu/imuRaw{dataset_num}.p"
    ts = tic()
    imu_data = read_data(ifile)
    toc(ts, "Data import")    
    return imu_data
def load_vicon_dataset(dataset_num):
    vfile = f"data/vicon/viconRot{dataset_num}.p"
    ts = tic()
    vicon_data = read_data(vfile)
    toc(ts, "Data import")
    return vicon_data
def load_cam_dataset(dataset_num):
    cfile = f"data/cam/cam{dataset_num}.p"
    ts = tic()
    cam_data = read_data(cfile)
    toc(ts, "Data import")
    return cam_data