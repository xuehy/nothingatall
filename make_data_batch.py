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
row_sample = []
W = 1242
H = 376

for i in range(200):
    filename = "%06d" % i
    im0 = Image.open(l_image_path + filename + '_10.png')
    im1 = Image.open(r_image_path + filename + '_10.png')
    img0 = np.asarray(im0, dtype=np.float32)
    img1 = np.asarray(im1, dtype=np.float32)

    img0 = img0 / 255
    img1 = img1 / 255

    # normalize the data
    for k in range(3):
        meanV = img0[:, :, k].mean()
        std = img0[:, :, k].std()
        img0[:, :, k] = (img0[:, :, k] - meanV) / std
        meanV = img1[:, :, k].mean()
        std = img1[:, :, k].std()
        img1[:, :, k] = (img1[:, :, k] - meanV) / std

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
    patch_size = 9
    half_size = patch_size // 2

    # find valid rows (imageId, rowId)
    rows = [(i, j)
            for j in range(4, height - 4)
            if (disp0[j, half_size: W - half_size]).sum() > 1]
    row_sample = row_sample + rows
    # patch size
    # sample_d = disp0[row_sample, half_size: W - half_size].astype(np.int32)
    maski = np.zeros((W - 2 * half_size))
    maski[0: width - half_size] = 1
    L.append(img0)
    R.append(img1)
    D.append(disp0)
    mask.append(maski)
    sys.stdout.write('\r processing image %d/200' % i)
print('\n')
print('total sentences: %i' % len(row_sample))
random.shuffle(row_sample)
row_sample = row_sample[0:50000]
train_set = pickle.dump([L, R, D, row_sample, mask], open('kitti2015b', 'wb'))
print('\n done')
