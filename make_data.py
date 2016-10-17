import PIL.Image as Image
import numpy as np
import six.moves.cPickle as pickle
import random
import sys
l_image_path = '/home/xuehy/Workspace/lstm-cnn/training/image_2/'
r_image_path = '/home/xuehy/Workspace/lstm-cnn/training/image_3/'
disp_image_path = '/home/xuehy/Workspace/lstm-cnn/training/disp_noc_0/'
train_data = '/home/xuehy/Workspace/lstm-cnn/train/'
L = []
R = []
D = []
mask = []
rows = []
W = 1242
H = 376
sample_number = 300
for i in range(200):
    filename = "%06d" % i
    im0 = Image.open(l_image_path + filename + '_10.png')
    im1 = Image.open(r_image_path + filename + '_10.png')
    img0 = np.asarray(im0, dtype=np.float32)
    img1 = np.asarray(im1, dtype=np.float32)

    depth = Image.open(disp_image_path + filename + '_10.png')

    disp0 = np.asarray(depth, dtype=np.uint16)

    disp0 = disp0 / 256.0
    width = img0.shape[1]
    height = img0.shape[0]

    # padding images
    img0 = np.pad(img0, ((0, H - height), (0, W - width), (0, 0)),
                  mode='constant', constant_values=0)
    img1 = np.pad(img1, ((0, H - height), (0, W - width), (0, 0)),
                  mode='constant', constant_values=0)
    disp0 = np.pad(disp0, ((0, H - height), (0, W - width)),
                   mode='constant', constant_values=0)
    row_sample = random.sample(range(4, height - 4), sample_number)
    # patch size
    patch_size = 9
    half_size = patch_size // 2
    sample_d = disp0[row_sample, half_size: W - half_size].astype(np.int32)
    maski = np.zeros((W - 2 * half_size))
    maski[0: width - half_size] = 1
    L.append(img0)
    R.append(img1)
    rows.append(row_sample)
    D.append(sample_d)
    mask.append(maski)
    sys.stdout.write('\r %d/200' % i)

train_set = pickle.dump([L, R, D, rows, mask], open('kitti2015', 'wb'))
print('\n done')
