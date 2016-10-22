import six.moves.cPickle as pickle
import numpy as np
import PIL.Image as Image

L, R, D, rows, masks = pickle.load(open('kitti2015b', 'rb'))

half_size = 4
W = 1242
H = 376

img = np.asarray(L[150])

im = Image.fromarray(np.asarray(img * 255, dtype=np.uint8))
im.save('test.png')
