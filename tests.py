import pickle


path = '/mnt/smb/locker/issa-locker/users/hc3190/data/results/curve/OO/barlow_CD/0049/layer4.1.pkl' 
with open(path, 'rb') as f:
    data = pickle.load(f)
    
print(data)

path = '/mnt/smb/locker/issa-locker/users/Eugénie/data/hk2_cam_color_no_pca/OO/barlow_v2/0014/layer4.1/y.pkl' 
with open(path, 'rb') as f:
    data = pickle.load(f)
    
print(data)
