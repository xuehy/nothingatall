import six.moves.cPickle as pickle
import numpy as np
import PIL.Image as Image

L, R, D = pickle.load(open('train_set', 'rb'))

sentence = np.asarray(L[1500])
width = sentence.shape[0] - 1 + 9
print(width)

image = np.empty((9, width, 3))

for i in range(4, width - 5, 5):
    for k in range(3):
        image[:, i - 4: i + 5, k] = sentence[i - 4, k, :, :]

im = Image.fromarray(np.asarray(image, dtype=np.uint8))
im.save('test.png')
