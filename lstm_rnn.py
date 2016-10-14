import theano
import theano.tensor as T
import numpy as np
import six.moves.cPickle as pickle


class ConvReLULayer(object):
    def __init__(self,
                 rng,
                 patch_shape,
                 filter_shape):
        """

        one patch at a time

        input: 4d tensor [patch_number, num feature maps,
        patch_width, patch_height]
        :type patch_shape: tuple of length 4
        :param patch_shape: (batch_size, num feature maps,
        patch_width, patch_height)

        :type filter_shape: tuple of length 4
        :param filter_shape: (number_of_filters, num input feature maps,
        filter height, filter width)

        """

        assert filter_shape[1] == patch_shape[1]

        self.patch_shape = patch_shape
        self.filter_shape = filter_shape
        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = np.zeros((filter_shape[0], ), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        self.params = [self.W, self.b]

    # def _step(self, _patch):
    #     patch = _patch.dimshuffle('x', 0, 1, 2)
    #     conv_result = T.nnet.conv2d(
    #         input=patch,
    #         filters=self.W,
    #         input_shape=self.patch_shape,
    #         filter_shape=self.filter_shape,
    #         border_mode='valid'
    #     ) + self.b.dimshuffle('x', 0, 'x', 'x')
    #     return conv_result[0, :, :, :]

    def makeLayer(self, inputL, inputR, ReLU):
        # _patch : num of feature maps x patch width x height
        # conv_result: 1 x number of features x width x height
        # conv_results : num of patches x num of features x width x height

        # conv_results, updates = theano.scan(
        #     fn=self._step,
        #     sequences=[input]
        # )
        conv_results = T.nnet.conv2d(
            input=T.concatenate([inputL, inputR]),
            filters=self.W,
            input_shape=(1234 * 2, self.patch_shape[1], self.patch_shape[2],
                         self.patch_shape[3]),
            filter_shape=self.filter_shape,
            border_mode='valid'
        ) + self.b.dimshuffle('x', 0, 'x', 'x')
        if (ReLU):
            conv_results = T.nnet.relu(
                conv_results
            )
        return conv_results[0:1234], conv_results[1234:2468]


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


class lstm_layer(object):
    def __init__(self, hidden_dim):
        """
        :type state_below: tensor of [nsteps, hidden_dim]
        :param state_below: every row is the input of a time

        :type W: tensor of [4, hidden_dim, hidden_dim]
        :param W: [W_i, W_f, W_o, W_c]

        """

        W_values = np.concatenate([ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim)],
                                  axis=1
        )

        U_values = np.concatenate([ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim),
                                   ortho_weight(hidden_dim)],
                                  axis=1
        )

        b_value = np.zeros((4 * hidden_dim, ))

        W = theano.shared(
            value=W_values,
            borrow=True,
            name='W'
        )

        U = theano.shared(
            value=U_values,
            borrow=True,
            name='U'
        )

        b = theano.shared(
            value=np.asarray(
                b_value,
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
        self.W = W
        self.U = U
        self.b = b
        self.hidden_dim = hidden_dim
        self.params = [self.W, self.U, self.b]

    def _slice(self, _x, n, dim):
        """
        to get the stacked Wi, Wf, Ui, Uf, etc.
        """
        if _x.ndim == 3:
            return _x[:, :, n * dim: (n + 1) * dim]
        return _x[:, n * dim: (n + 1) * dim]

    def _step(self, x_, h_, c_):
        """
        h_: hidden_dim row vector
        """
        preact = T.dot(h_, self.W)
        preact += x_

        i = T.nnet.sigmoid(self._slice(preact, 0, self.hidden_dim))
        f = T.nnet.sigmoid(self._slice(preact, 1, self.hidden_dim))
        o = T.nnet.sigmoid(self._slice(preact, 2, self.hidden_dim))
        c = T.tanh(self._slice(preact, 3, self.hidden_dim))

        c = f * c_ + i * c
        h = o * T.tanh(c)

        return h, c

    def makeLayer(self, state_below):
        # this is W*x_t for t = 0, 1, 2, ...
        # and arranged in rows
        nsteps = state_below.shape[0]
        state_below = (T.dot(state_below, self.W) + self.b)

        def np_floatX(data):
            return np.asarray(data, dtype=theano.config.floatX)

        rval, updates = theano.scan(self._step,
                                    sequences=[state_below],
                                    outputs_info=[T.alloc(np_floatX(0.),
                                                          1,
                                                          self.hidden_dim),
                                                  T.alloc(np_floatX(0.),
                                                          1,
                                                          self.hidden_dim)
                                                  ],
                                    name='LSTM_layers',
                                    n_steps=nsteps)

        return rval[0][:, 0, :]


class bi_lstm_cnn(object):
    def __init__(self, input_patches_left, input_patches_right,
                 input_shape, patch_shape,
                 input_filter_shape, filter_shape):

        """
        input_patches: patch num x num of features x width x height
        """
        # make patches from ImageL and ImageR with row id and width
        rng = np.random.RandomState(1234)
        conv1 = ConvReLULayer(
            rng=rng,
            patch_shape=input_shape,
            filter_shape=input_filter_shape
        )

        patch_size2 = patch_shape[3] - filter_shape[3] + 1
        patch_shape2 = (1,
                        filter_shape[0], patch_size2, patch_size2)

        conv2 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape2,
            filter_shape=filter_shape
        )
        patch_size3 = patch_size2 - filter_shape[3] + 1
        patch_shape3 = (1,
                        filter_shape[0], patch_size3, patch_size3)

        conv3 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape3,
            filter_shape=filter_shape
        )

        patch_size4 = patch_size3 - filter_shape[3] + 1
        patch_shape4 = (1,
                        filter_shape[0], patch_size4, patch_size4)

        conv4 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape4,
            filter_shape=filter_shape
        )

        conv1_l, conv1_r = conv1.makeLayer(input_patches_left,
                                           input_patches_right,
                                           True)
        conv2_l, conv2_r = conv2.makeLayer(conv1_l, conv1_r, True)
        conv3_l, conv3_r = conv3.makeLayer(conv2_l, conv2_r, True)
        conv4_l, conv4_r = conv4.makeLayer(conv3_l, conv3_r, True)

        # conv4.output : num of patches x num features x width x height
        left_trans, updates = theano.scan(
            # flatten makes a tensor as a vector
            fn=lambda _patch: _patch.flatten(1),
            sequences=[conv4_l]
        )

        right_trans, updates = theano.scan(
            # flatten makes a tensor as a vector
            fn=lambda _patch: _patch.flatten(1),
            sequences=[conv4_r]
        )

        # trans: num of patches x full feature dim
        forward_lstm_input_left = left_trans
        backward_lstm_input_left = left_trans[::-1]

        forward_lstm_input_right = right_trans
        backward_lstm_input_right = right_trans[::-1]

        forward_lstm = lstm_layer(
            hidden_dim=filter_shape[0]
        )

        backward_lstm = lstm_layer(
            hidden_dim=filter_shape[0]
        )

        self.params = [conv1.params +
                       conv2.params +
                       conv3.params +
                       conv4.params +
                       forward_lstm.params +
                       backward_lstm.params]

        left_1 = forward_lstm.makeLayer(forward_lstm_input_left)
        left_2 = backward_lstm.makeLayer(backward_lstm_input_left)

        left_feature = T.concatenate([left_1,
                                      left_2[::-1]],
                                     axis=1)

        right_1 = forward_lstm.makeLayer(forward_lstm_input_right)
        right_2 = backward_lstm.makeLayer(backward_lstm_input_right)
        right_feature = T.concatenate([right_1,
                                       right_2[::-1]],
                                      axis=1)
        product = T.dot(left_feature, right_feature.T)
        self.softmax_result = T.nnet.nnet.softmax(product)
        self.match_pred = T.argmax(self.softmax_result, axis=1)

    # y is directly disparity
    # have not dealt with problems:
    # the pixels without matches
    def negativeLogLikelihood(self, y):
        match_idx = T.arange(y.shape[0]) - y
        return -T.mean(T.log(self.softmax_result)[T.arange(y.shape[0]),
                                                  match_idx])

    def bad30(self, y):
        match_idx = T.arange(y.shape[0]) - y
        return T.sum(T.ge(abs(match_idx -
                              self.match_pred[0: y.shape[0]]),
                          3.0)) / y.shape[0]


def makeShare(x):
    return theano.shared(np.asarray(x,
                                    dtype=theano.config.floatX),
                         borrow=True)


def generateData(ImageL, ImageR, ImgId, RowSamples):
    """
    ImageL: 200 x [width x height x 3]
    """
    img0 = ImageL[ImgId]
    img1 = ImageR[ImgId]
    patch_size = 9
    half_size = patch_size // 2
    width = img0.shape[1]

    height = img0.shape[0]
    pad_num_x = 1242 - width
    pad_num_y = 376 - height
    img0 = np.pad(img0, ((0, pad_num_y), (0, pad_num_x), (0, 0)),
                  mode='constant', constant_values=0)
    img1 = np.pad(img0, ((0, pad_num_y), (0, pad_num_y), (0, 0)),
                  mode='constant', constant_values=0)
    # sentence_L = np.empty((width - patch_size + 1, 3, patch_size,patch_size))
    # sentence_R = np.empty((width - patch_size + 1, 3, patch_size,patch_size))
    sentence_L = np.empty((1242 - patch_size + 1, 3, patch_size, patch_size))
    sentence_R = np.empty((1242 - patch_size + 1, 3, patch_size, patch_size))

    L = []
    R = []
    for j in RowSamples:
        patch_id = 0
        for x in range(half_size, 1242 - half_size):
            sample_l = img0[j - half_size: j + half_size + 1,
                            x - half_size: x + half_size + 1,
                            :]
            sample_r = img1[j - half_size: j + half_size + 1,
                            x - half_size: x + half_size + 1,
                            :]
            for k in range(3):
                sentence_L[patch_id, k, :, :] = sample_l[:, :, k]
                sentence_R[patch_id, k, :, :] = sample_r[:, :, k]
            patch_id = patch_id + 1
        # sentence_L = np.asarray(sentence_L, dtype=theano.config.floatX)
        # sentence_R = np.asarray(sentence_R, dtype=theano.config.floatX)
        L.append(sentence_L)
        R.append(sentence_R)
    L = np.asarray(L, dtype=theano.config.floatX)
    R = np.asarray(R, dtype=theano.config.floatX)
    return L, R

if __name__ == '__main__':
    index = T.lscalar('index')
    xl = T.tensor4('xl')
    xr = T.tensor4('xr')
    y = T.ivector('y')

    trainL = theano.shared(np.asarray(np.empty((50, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))

    trainR = theano.shared(np.asarray(np.empty((50, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))

    batchL = theano.shared(np.asarray(np.empty((50, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))
    batchR = theano.shared(np.asarray(np.empty((50, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))
    # let's just test
    tensor5 = T.TensorType(theano.config.floatX, (False,)*5)
    ImageL, ImageR, disp, RowSample = pickle.load(open('train_set', 'rb'))
    print('...building the model')
    cnn_lstm = bi_lstm_cnn(
        input_patches_left=xl,
        input_patches_right=xr,
        input_shape=(1, 3, 9, 9),
        patch_shape=(1, 112, 9, 9),
        input_filter_shape=(112, 3, 3, 3),
        filter_shape=(112, 112, 3, 3)
    )

    cost = cnn_lstm.negativeLogLikelihood(y)

    validate_model = theano.function(
        inputs=[index, y],
        outputs=cnn_lstm.bad30(y),
        givens={
            xl: batchL[index],
            xr: batchR[index]
        }
    )

    print('...model built')
    disp = [x.astype('int32') for x in disp]
    val_loss = 0
    for i in range(200):
        valSample = RowSample[i][50:100]
        disp_I = disp[i][50:100]
        print(i)
        L, R = generateData(ImageL, ImageR, i, valSample)
        batchL.set_value(L, borrow=True)
        batchR.set_value(R, borrow=True)
        print('sent new data to gpu')
        val_loss_perI = [validate_model(j, disp_I[j])
                         for j in range(50)]
        val_loss += sum(val_loss_perI)

    print(val_loss / 10000.0)

    learning_rate = 0.003

    params = cnn_lstm.params
    g_params = T.grad(cost, params)
    updates = [(parami, parami - learning_rate * gradi)
               for parami, gradi in zip(params, g_params)]

    train_model = theano.function(
        inputs=[index, y],
        outputs=cost,
        updates=updates,
        givens={
            xl: trainL[index],
            xr: trainR[index]
        }
    )
    
