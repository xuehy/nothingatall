import theano
import theano.tensor as T
import numpy as np
import six.moves.cPickle as pickle
import sys
import time


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
        preact = T.dot(h_, self.U)
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
        self.conv1 = ConvReLULayer(
            rng=rng,
            patch_shape=input_shape,
            filter_shape=input_filter_shape
        )

        patch_size2 = patch_shape[3] - filter_shape[3] + 1
        patch_shape2 = (1,
                        filter_shape[0], patch_size2, patch_size2)

        self.conv2 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape2,
            filter_shape=filter_shape
        )
        patch_size3 = patch_size2 - filter_shape[3] + 1
        patch_shape3 = (1,
                        filter_shape[0], patch_size3, patch_size3)

        self.conv3 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape3,
            filter_shape=filter_shape
        )

        patch_size4 = patch_size3 - filter_shape[3] + 1
        patch_shape4 = (1,
                        filter_shape[0], patch_size4, patch_size4)

        self.conv4 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape4,
            filter_shape=filter_shape
        )

        conv1_l, conv1_r = self.conv1.makeLayer(input_patches_left,
                                                input_patches_right,
                                                True)
        conv2_l, conv2_r = self.conv2.makeLayer(conv1_l, conv1_r, True)
        conv3_l, conv3_r = self.conv3.makeLayer(conv2_l, conv2_r, True)
        conv4_l, conv4_r = self.conv4.makeLayer(conv3_l, conv3_r, True)

        # conv4.output : num of patches x num features x width x height
        # left_trans, updates = theano.scan(
        #     # flatten makes a tensor as a vector
        #     fn=lambda _patch: _patch.flatten(1),
        #     sequences=[conv4_l]
        # )

        # right_trans, updates = theano.scan(
        #     # flatten makes a tensor as a vector
        #     fn=lambda _patch: _patch.flatten(1),
        #     sequences=[conv4_r]
        # )
        left_trans = conv4_l.flatten(2)
        right_trans = conv4_r.flatten(2)
        # trans: num of patches x full feature dim
        forward_lstm_input_left = left_trans
        backward_lstm_input_left = left_trans[::-1]

        forward_lstm_input_right = right_trans
        backward_lstm_input_right = right_trans[::-1]

        self.forward_lstm = lstm_layer(
            hidden_dim=filter_shape[0]
        )

        self.backward_lstm = lstm_layer(
            hidden_dim=filter_shape[0]
        )

        self.params = self.conv1.params + self.conv2.params + self.conv3.params
        self.params = self.params + self.conv4.params
        self.params = self.params + self.forward_lstm.params
        self.params = self.params + self.backward_lstm.params

        left_1 = self.forward_lstm.makeLayer(forward_lstm_input_left)
        left_2 = self.backward_lstm.makeLayer(backward_lstm_input_left)

        left_feature = T.concatenate([left_1,
                                      left_2[::-1]],
                                     axis=1)

        right_1 = self.forward_lstm.makeLayer(forward_lstm_input_right)
        right_2 = self.backward_lstm.makeLayer(backward_lstm_input_right)
        right_feature = T.concatenate([right_1,
                                       right_2[::-1]],
                                      axis=1)
        product = T.dot(left_feature, right_feature.T)
        self.softmax_result = T.nnet.nnet.softmax(product)
        self.match_pred = T.argmax(self.softmax_result, axis=1)

    # y is directly disparity
    # have not dealt with problems:
    # the pixels without matches
    def negativeLogLikelihood(self, y, m):
        match_idx = T.arange(y.shape[0]) - y
        mask = (match_idx > 0) * m * (y > 0)
        # in GT:
        # 0 indicates invalid pixel
        N = T.sum(mask)
        nll = T.log(self.softmax_result)[T.arange(y.shape[0]),
                                         match_idx]
        return -T.sum(nll * mask) / (N + 1)

    def bad30(self, y, m):
        match_idx = T.arange(y.shape[0]) - y
        mask = (match_idx > 0) * m * (y > 0)
        N = T.sum(mask)
        return T.sum(T.ge(abs(match_idx -
                              self.match_pred[0: y.shape[0]]),
                          3.0) * mask) / (N + 1)


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


def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)


# rmsprop optimizer
def rmsprop(lr, params, grads, trainL, trainR,
            xl, xr, m, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.))
                    for p in params]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.))
                     for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.))
                      for p in params]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g)
            for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([index, y, m], cost,
                                    updates=zgup + rgup + rg2up,
                                    givens={
                                        xl: trainL[index],
                                        xr: trainR[index]
                                    },
                                    name="rmsprop_f_grad_shared")
    updir = [theano.shared(p.get_value() * numpy_floatX(0.))
             for p in params]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / T.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads,
                                            running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(params, updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               givens={
                                   xl: trainL[index],
                                   xr: trainR[index]
                               },
                               name='rmsprop_f_update')
    return f_grad_shared, f_update

if __name__ == '__main__':
    index = T.lscalar('index')
    xl = T.tensor4('xl')
    xr = T.tensor4('xr')
    y = T.ivector('y')
    m = T.vector('m')
    trainL = theano.shared(np.asarray(np.empty((200, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))

    trainR = theano.shared(np.asarray(np.empty((200, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))

    batchL = theano.shared(np.asarray(np.empty((100, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))
    batchR = theano.shared(np.asarray(np.empty((100, 1242-2*4, 3, 9, 9)),
                                      dtype=theano.config.floatX))
    # let's just test
    tensor5 = T.TensorType(theano.config.floatX, (False,)*5)
    ImageL, ImageR, disp, RowSample, mask = pickle.load(open('kitti2015',
                                                             'rb'))

    print('...building the model')
    cnn_lstm = bi_lstm_cnn(
        input_patches_left=xl,
        input_patches_right=xr,
        input_shape=(1, 3, 9, 9),
        patch_shape=(1, 112, 9, 9),
        input_filter_shape=(112, 3, 3, 3),
        filter_shape=(112, 112, 3, 3)
    )

    cost = cnn_lstm.negativeLogLikelihood(y, m)
# theano.printing.pydotprint(cost, outfile="cost.png", var_with_name_simple
# =True)
# sys.exit(0)

    validate_model = theano.function(
        inputs=[index, y, m],
        outputs=cnn_lstm.bad30(y, m),
        givens={
            xl: batchL[index],
            xr: batchR[index]
        }
    )

    disp = [x.astype('int32') for x in disp]
    learning_rate = 0.00001
    lr = T.scalar('lr')
    validate_frequency = 5000
    n_epochs = 200
    # params = cnn_lstm.conv1.params + cnn_lstm.conv2.params
    # params = params + cnn_lstm.conv3.params + cnn_lstm.conv4.params
    # params = params + cnn_lstm.forward_lstm.params
    # params = params + cnn_lstm.backward_lstm.params
    params = cnn_lstm.params

    # sgd optimizer
    g_params = T.grad(cost, params)
    updates = [(parami, parami - learning_rate * gradi)
               for parami, gradi in zip(params, g_params)]

    train_model = theano.function(
        inputs=[index, y, m],
        outputs=cost,
        updates=updates,
        givens={
            xl: trainL[index],
            xr: trainR[index]
        }
    )

    # use rmsprop here
    f_grad_shared, f_update = rmsprop(lr, params, g_params,
                                      trainL, trainR,
                                      xl, xr, m, y, cost)
    print('...model built')
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
    print('...training')
    epoch = 0
    done_looping = False
    save_interval = 2
    iter = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for image_index in range(200):
            # iter = (epoch - 1) * 200 + image_index
            trainSample = RowSample[image_index][0:200]
            L, R = generateData(ImageL, ImageR, image_index, trainSample)
            trainL.set_value(L, borrow=True)
            trainR.set_value(R, borrow=True)
            # for each row
            disp_I = disp[image_index][0:200]
            mask_I = mask[image_index]
            mask_I = mask_I.astype(np.float32)

            for sample_index in range(200):
                # cost_ij = train_model(sample_index, disp_I[sample_index],
                #                       mask_I)
                cost_ij = f_grad_shared(sample_index, disp_I[sample_index],
                                        mask_I)
                f_update(learning_rate)
                # skip rows without groundtruth
                if (disp_I[sample_index].sum() == 0):
                    # print('skip')
                    continue
                if iter % 500 == 0:
                    # print('image %i' % image_index)
                    # print(disp_I[sample_index].sum())
                    print(time.strftime(ISOTIMEFORMAT, time.localtime()))
                    print('training iter = %i, loss = %f' % (iter, cost_ij))

                if iter % validate_frequency == 0:
                    val_loss = 0
                    for i in range(200):
                        valSample = RowSample[i][200:300]
                        disp_I_val = disp[i][200:300]
                        mask_I_val = mask[i]
                        mask_I_val = mask_I.astype(np.float32)
                        L_val, R_val = generateData(ImageL, ImageR, i,
                                                    valSample)
                        batchL.set_value(L_val, borrow=True)
                        batchR.set_value(R_val, borrow=True)
                        val_loss_perI = [
                            validate_model(j, disp_I_val[j], mask_I_val)
                            for j in range(100)]
                        val_cnt = sum([disp_I_val[j].sum() != 0
                                       for j in range(100)])
                        val_loss += sum(val_loss_perI) / val_cnt
                    print('epoch %i, image %i/%i, validation_error %f %%' %
                          (epoch, sample_index, 200, val_loss / 2))
                iter += 1

        if epoch % save_interval == 0:
            pickle.dump([cnn_lstm.conv1.W, cnn_lstm.conv1.b,
                         cnn_lstm.conv2.W, cnn_lstm.conv2.b,
                         cnn_lstm.conv3.W, cnn_lstm.conv3.b,
                         cnn_lstm.conv4.W, cnn_lstm.conv4.b,
                         cnn_lstm.forward_lstm.W,
                         cnn_lstm.forward_lstm.U,
                         cnn_lstm.forward_lstm.b,
                         cnn_lstm.backward_lstm.W,
                         cnn_lstm.backward_lstm.U,
                         cnn_lstm.backward_lstm.b],
                        open('trained_%i' % epoch, 'wb'))
