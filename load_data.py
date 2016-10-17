import six.moves.cPickle as pickle
import numpy as np
import PIL.Image as Image

L, R, D, rows, masks = pickle.load(open('kitti2015', 'rb'))

half_size = 4
W = 1242
H = 376

img = np.asarray(L[10])
print(sum(masks[10]))
im = Image.fromarray(np.asarray(img, dtype=np.uint8))
im.save('test.png')
