import pickle


path = '/mnt/smb/locker/issa-locker/users/hc3190/data/results/curve/01/barlow_CD/0049/layer4.1.pkl' 
with open(path, 'rb') as f:
    data = pickle.load(f)
    
print(data)

path_ = '/mnt/smb/locker/issa-locker/users/Eugénie/data/hk2_cam_color_no_pca/00/barlow_v2/0014/layer4.1/y.pkl' 
with open(path_, 'rb') as fp:
    datap = pickle.load(fp)
    
print(datap)
