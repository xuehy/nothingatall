import theano
import theano.tensor as T
from theano.gradient import grad_clip
import numpy as np
import six.moves.cPickle as pickle
import sys
import time
import socket
import theano.misc.pkl_utils


class ConvReLULayer(object):
    def __init__(self,
                 rng,
                 patch_shape,
                 filter_shape):
        """

        one patch at a time

        input: 4d tensor [patch_number * 2 * batch_size
, num feature maps,
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

    def makeLayer(self, inputL, inputR, ReLU):
        """
        inputX : (batch_size x number of patches) x
        num of feature maps x patch width x patch height
        conv_results : (bs x num of patches) x num of features x width x height

        """

        conv_results = T.nnet.conv2d(
            # inputL, inputR: (batch_size x n_patch) x n_fea x w x h
            input=T.concatenate([inputL, inputR]),
            filters=self.W,
            input_shape=(1234 * 2 * self.patch_shape[0],
                         self.patch_shape[1], self.patch_shape[2],
                         self.patch_shape[3]),
            filter_shape=self.filter_shape,
            border_mode='valid'
        ) + self.b.dimshuffle('x', 0, 'x', 'x')
        if (ReLU):
            conv_results = T.nnet.relu(
                conv_results
            )
        batch_size = self.patch_shape[0]
        left = conv_results[0: 1234 * batch_size]
        right = conv_results[1234 * batch_size: 2468 * batch_size]
        return left, right


def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


class lstm_layer(object):
    def __init__(self, hidden_dim):
        """
        :type state_below: tensor of [nsteps, batch_size, hidden_dim]

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
            name='lstm_W'
        )

        U = theano.shared(
            value=U_values,
            borrow=True,
            name='lstm_U'
        )

        b = theano.shared(
            value=np.asarray(
                b_value,
                dtype=theano.config.floatX
            ),
            name='lstm_b',
            borrow=True
        )
        self.W = W
        self.U = U
        self.b = b
        # for gradient clipping
        self.W_ = grad_clip(W, -10, 10)
        self.U_ = grad_clip(U, -10, 10)

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
        preact = T.dot(h_, self.U_)
        preact += x_

        i = T.nnet.sigmoid(self._slice(preact, 0, self.hidden_dim))
        f = T.nnet.sigmoid(self._slice(preact, 1, self.hidden_dim))
        o = T.nnet.sigmoid(self._slice(preact, 2, self.hidden_dim))
        c = T.tanh(self._slice(preact, 3, self.hidden_dim))

        c = f * c_ + i * c
        h = o * T.tanh(c)

        return h, c

    def makeLayer(self, state_below, batch_size):
        # this is W*x_t for t = 0, 1, 2, ...
        # and arranged in rows
        nsteps = state_below.shape[0]
        state_below = (T.dot(state_below, self.W_) + self.b)

        def np_floatX(data):
            return np.asarray(data, dtype=theano.config.floatX)

        rval, updates = theano.scan(self._step,
                                    sequences=[state_below],
                                    outputs_info=[T.alloc(np_floatX(0.),
                                                          batch_size,
                                                          self.hidden_dim),
                                                  T.alloc(np_floatX(0.),
                                                          batch_size,
                                                          self.hidden_dim)
                                                  ],
                                    name='LSTM_layers',
                                    n_steps=nsteps)
        # rval[0] : nSteps x batch size x dim
        return rval[0]


class bi_lstm_cnn(object):
    def __init__(self, input_patches_left, input_patches_right,
                 input_shape, patch_shape,
                 input_filter_shape, filter_shape):

        """
        input_patches: (patch num x batch size) x num of features x w x h
        """
        # make patches from ImageL and ImageR with row id and width

        batch_size = input_shape[0]
        rng = np.random.RandomState(1234)
        self.conv1 = ConvReLULayer(
            rng=rng,
            patch_shape=input_shape,
            filter_shape=input_filter_shape
        )

        patch_size2 = patch_shape[3] - filter_shape[3] + 1
        patch_shape2 = (input_shape[0],
                        filter_shape[0], patch_size2, patch_size2)

        self.conv2 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape2,
            filter_shape=filter_shape
        )
        patch_size3 = patch_size2 - filter_shape[3] + 1
        patch_shape3 = (input_shape[0],
                        filter_shape[0], patch_size3, patch_size3)

        self.conv3 = ConvReLULayer(
            rng=rng,
            patch_shape=patch_shape3,
            filter_shape=filter_shape
        )

        patch_size4 = patch_size3 - filter_shape[3] + 1
        patch_shape4 = (input_shape[0],
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

        # conv4.output : (batch_size x num of patches) x num features x w x h
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
        # trans: (batch_size x num of patches) x full feature dim
        # -> batch_size x 1234 x 112
        left_trans = T.reshape(left_trans, (input_shape[0], 1234, 112))
        right_trans = T.reshape(right_trans,
                                (input_shape[0], 1234, 112))

        # -> 1234(nSteps) x batch_size x 112(Emb_Dim)
        left_trans = left_trans.swapaxes(0, 1)
        right_trans = right_trans.swapaxes(0, 1)
        forward_lstm_input_left = left_trans
        backward_lstm_input_left = left_trans[:, ::-1, :]

        forward_lstm_input_right = right_trans
        backward_lstm_input_right = right_trans[:, ::-1, :]

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

        # left: nSteps x batch size x dim
        left_1 = self.forward_lstm.makeLayer(forward_lstm_input_left,
                                             batch_size)
        left_2 = self.backward_lstm.makeLayer(backward_lstm_input_left,
                                              batch_size)

        left_feature = T.concatenate([left_1,
                                      left_2[::-1]],
                                     axis=2)
        # -< batch_size x nSteps x dim
        left_feature = left_feature.swapaxes(0, 1)
        right_1 = self.forward_lstm.makeLayer(forward_lstm_input_right,
                                              batch_size)
        right_2 = self.backward_lstm.makeLayer(backward_lstm_input_right,
                                               batch_size)
        right_feature = T.concatenate([right_1,
                                       right_2[::-1]],
                                      axis=2)
        # -> batch_size x nSteps x dim -> bs x dim x nSteps
        right_feature = right_feature.swapaxes(0, 1)
        right_feature = right_feature.swapaxes(1, 2)
        # product: batch_size x nSteps x nSteps
        product = T.batched_dot(left_feature, right_feature)
        product = T.flatten(product)
        product = T.reshape(product, (1234 * batch_size, 1234))
        softmax_result = T.nnet.nnet.softmax(product)
        self.batch_size = batch_size
        # (batch size x nSteps) x nSteps
        self.softmax_result = softmax_result
        # match_pred: (batch_size x nSteps) x 1
        self.match_pred = T.argmax(self.softmax_result, axis=1)

    # y is directly disparity: batch size x nSteps
    # m: batch size x nSteps
    # have not dealt with problems:
    # the pixels without matches
    def negativeLogLikelihood(self, y, m):
        match_idx = T.arange(y.shape[1]) - y

        mask = (match_idx > 0) * m * (y > 0)
        mask = mask.reshape((self.batch_size * 1234,))
        match_idx = match_idx.reshape((self.batch_size * 1234,))
        # in GT:
        # 0 indicates invalid pixel
        N = T.sum(mask)
        nll = T.log(self.softmax_result + 1e-10)[
            T.arange(self.batch_size * 1234),
            match_idx]
        return -T.sum(nll * mask) / (N + 1)

    def bad30(self, y, m):
        match_idx = T.arange(y.shape[1]) - y
        mask = (match_idx > 0) * m * (y > 0)
        mask = mask.reshape((self.batch_size * 1234,))
        match_idx = match_idx.reshape((self.batch_size * 1234,))

        N = T.sum(mask)
        return T.sum(T.ge(abs(match_idx -
                              self.match_pred[0: self.batch_size *
                                              y.shape[1]]),
                          3.0) * mask) / (N + 1)

    def load_model(self, snapshot):
        load_file = open(snapshot, 'rb')
        for p in self.params:
            p.set_value(pickle.load(load_file), borrow=True)
        print('loaded snapshot from ' + snapshot)
        load_file.close()
        # self.conv1.W,
        # self.conv1.b,
        # self.conv2.W,
        # self.conv2.b,
        # self.conv3.W,
        # self.conv3.b,
        # self.conv4.W,
        # self.conv4.b,
        # self.forward_lstm.W,
        # self.forward_lstm.U,
        # self.forward_lstm.b,
        # self.backward_lstm.W,
        # self.backward_lstm.U,
        # self.backward_lstm.b = theano.misc.pkl_utils.load(
        #     open(snapshot, 'rb')
        # )


def makeShare(x):
    return theano.shared(np.asarray(x,
                                    dtype=theano.config.floatX),
                         borrow=True)


def generateData(ImageL, ImageR, ImageD, ImageM,
                 RowSamples,
                 batch_number, batch_size):
    """
    batch_number: current batch id
    batch_size: number of sentences in each batch
    ImageL: 200 x [width x height x 3]
    """
    samples = RowSamples[batch_number * batch_size:
                         (batch_number + 1) * batch_size]
    L = []
    R = []
    D = []
    M = []
    for sample in samples:
        ImgId = sample[0]
        rowId = sample[1]
        m = ImageM[ImgId]
        img0 = ImageL[ImgId]
        img1 = ImageR[ImgId]
        patch_size = 9
        half_size = patch_size // 2

        sentence_L = np.empty((1242 - patch_size + 1,
                               3, patch_size, patch_size))
        sentence_R = np.empty((1242 - patch_size + 1,
                               3, patch_size, patch_size))
        sentence_D = ImageD[ImgId][rowId, half_size: 1242 - half_size]
        patch_id = 0
        for x in range(half_size, 1242 - half_size):
            sample_l = img0[rowId - half_size: rowId + half_size + 1,
                            x - half_size: x + half_size + 1,
                            :]
            sample_r = img1[rowId - half_size: rowId + half_size + 1,
                            x - half_size: x + half_size + 1,
                            :]
            for k in range(3):
                sentence_L[patch_id, k, :, :] = sample_l[:, :, k]
                sentence_R[patch_id, k, :, :] = sample_r[:, :, k]
            patch_id = patch_id + 1

        L.append(sentence_L)
        R.append(sentence_R)
        D.append(sentence_D)
        M.append(m)
    L = np.asarray(L, dtype=theano.config.floatX)
    L = L.reshape((batch_size * 1234, 3, 9, 9))
    R = np.asarray(R, dtype=theano.config.floatX)
    R = L.reshape((batch_size * 1234, 3, 9, 9))
    D = np.asarray(D, dtype=np.int32)
    M = np.asarray(M, dtype=theano.config.floatX)
    return L, R, D, M


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
    f_grad_shared = theano.function([y, m], cost,
                                    updates=zgup + rgup + rg2up,
                                    givens={
                                        xl: trainL,
                                        xr: trainR
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
                                   xl: trainL,
                                   xr: trainR
                               },
                               name='rmsprop_f_update')
    return f_grad_shared, f_update


# adadelta
def adadelta(lr, params, grads, trainL, trainR,
             xl, xr, m, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.))
                    for p in params]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.))
                   for p in params]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.))
                      for p in params]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([y, m], cost,
                                    updates=zgup + rg2up,
                                    givens={
                                        xl: trainL,
                                        xr: trainR
                                    },
                                    name='adadelta_f_grad_shared')
    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]

    param_up = [(p, p + ud) for p, ud in zip(params, updir)]
    f_update = theano.function([lr], [],
                               updates=ru2up + param_up,
                               givens={
                                   xl: trainL,
                                   xr: trainR
                               },
                               on_unused_input='ignore',
                               name='adadelta_f_update')
    return f_grad_shared, f_update


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    address = ('10.76.1.30', 5274)
    xl = T.tensor4('xl')
    xr = T.tensor4('xr')
    y = T.imatrix('y')
    m = T.matrix('m')
    batch_size = 16
    val_batch_size = 16

    trainL = theano.shared(
        np.asarray(np.empty((batch_size * (1242-2*4), 3, 9, 9)),
                   dtype=theano.config.floatX))

    trainR = theano.shared(
        np.asarray(np.empty((batch_size * (1242-2*4), 3, 9, 9)),
                   dtype=theano.config.floatX))

    batchL = theano.shared(
        np.asarray(np.empty((val_batch_size * 1234, 3, 9, 9)),
                   dtype=theano.config.floatX))
    batchR = theano.shared(
        np.asarray(np.empty((val_batch_size * 1234, 3, 9, 9)),
                   dtype=theano.config.floatX))
    # let's just test
    tensor5 = T.TensorType(theano.config.floatX, (False,)*5)
    ImageL, ImageR, disp, RowSample, mask = pickle.load(open('kitti2015b',
                                                             'rb'))
    print('...building the model')
    cnn_lstm = bi_lstm_cnn(
        input_patches_left=xl,
        input_patches_right=xr,
        input_shape=(batch_size, 3, 9, 9),
        patch_shape=(1, 112, 9, 9),
        input_filter_shape=(112, 3, 3, 3),
        filter_shape=(112, 112, 3, 3)
    )

    cost = cnn_lstm.negativeLogLikelihood(y, m)
# theano.printing.pydotprint(cost, outfile="cost.png", var_with_name_simple
# =True)
# sys.exit(0)

    validate_model = theano.function(
        inputs=[y, m],
        outputs=cnn_lstm.bad30(y, m),
        givens={
            xl: batchL,
            xr: batchR
        }
    )

    disp = [x.astype('int32') for x in disp]
    weight_decay = 0.00001
    learning_rate = 0.00001
    lr = T.scalar('lr')
    validate_frequency = 2500
    n_epochs = 200
    n_batches = 40000 // batch_size
    n_val_batches = 10000 // val_batch_size
    # params = cnn_lstm.conv1.params + cnn_lstm.conv2.params
    # params = params + cnn_lstm.conv3.params + cnn_lstm.conv4.params
    # params = params + cnn_lstm.forward_lstm.params
    # params = params + cnn_lstm.backward_lstm.params
    params = cnn_lstm.params

    # weight decay term
    W1 = cnn_lstm.conv1.params[0]
    W2 = cnn_lstm.conv2.params[0]
    W3 = cnn_lstm.conv3.params[0]
    W4 = cnn_lstm.conv4.params[0]
    W5 = cnn_lstm.forward_lstm.params[0]
    U5 = cnn_lstm.forward_lstm.params[1]
    W6 = cnn_lstm.backward_lstm.params[0]
    U6 = cnn_lstm.backward_lstm.params[1]
    weight_decay_ = 0.
    weight_decay_ += (W1 ** 2).sum() + (W2 ** 2).sum()
    weight_decay_ += (W3 ** 2).sum() + (W4 ** 2).sum() + (W5 ** 2).sum()
    weight_decay_ += (W6 ** 2).sum() + (U5 ** 2).sum() + (U6 ** 2).sum()
    weight_decay_ *= weight_decay
    cost += weight_decay_
    # sgd optimizer
    g_params = T.grad(cost, params)
    # updates = [(parami, parami - learning_rate * gradi)
    #            for parami, gradi in zip(params, g_params)]

    # train_model = theano.function(
    #     inputs=[y, m],
    #     outputs=cost,
    #     updates=updates,
    #     givens={
    #         xl: trainL,
    #         xr: trainR
    #     }
    # )

    # use rmsprop here
    f_grad_shared, f_update = rmsprop(lr, params, g_params,
                                      trainL, trainR,
                                      xl, xr, m, y, cost)
    print('...model built')
    ISOTIMEFORMAT = '%Y-%m-%d %H:%M:%S'
    print('...training with batch size %i, weight decay %f'
          % (batch_size, weight_decay))
    print('total number of batches %i' % n_batches)
    print('All Conv Layer are with ReLU')
    print('With gradient clipping')
    print('lr policy: RMSProp')
    epoch = 0
    done_looping = False
    save_interval = 2
    iter = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for batch_id in range(n_batches):
            iter = (epoch - 1) * n_batches + batch_id
            L, R, D, m = generateData(ImageL, ImageR, disp, mask,
                                      RowSample[0:40000], batch_id, batch_size)
            trainL.set_value(L, borrow=True)
            trainR.set_value(R, borrow=True)
            # for each row
            # cost_ij = train_model(sample_index, disp_I[sample_index],
            #                       mask_I)
            cost_ij = f_grad_shared(D, m)
            f_update(learning_rate)
            if iter % 500 == 0:
                # print('image %i' % image_index)
                # print(disp_I[sample_index].sum())
                print(time.strftime(ISOTIMEFORMAT, time.localtime()))
                print('training iter = %i, loss = %f' % (iter, cost_ij))
                msg = "Iter = %d, Loss = %f" % (iter, cost_ij)
                s.sendto(msg.encode(), address)

            if (iter + 1) % validate_frequency == 0:
                print('validating...')
                val_loss = 0
                valSample = RowSample[40000:50000]
                for i in range(n_val_batches):
                    L_val, R_val, D_val, m_val = generateData(ImageL,
                                                              ImageR,
                                                              disp, mask,
                                                              valSample,
                                                              i,
                                                              val_batch_size)
                    batchL.set_value(L_val, borrow=True)
                    batchR.set_value(R_val, borrow=True)
                    val_loss += validate_model(D_val, m_val)
                print('epoch %i, validation_error %f %%' %
                      (epoch, val_loss / n_val_batches * 100))

        if epoch % save_interval == 0:
            print('...saving snapshot into trained_epoch%i' % epoch)
            save_file = open('trained_epoch%i' % epoch, 'wb')
            for p in cnn_lstm.params:
                pickle.dump(p.get_value(borrow=True), save_file)
            save_file.close()
            # theano.misc.pkl_utils.dump((cnn_lstm.conv1.W, cnn_lstm.conv1.b,
            #                             cnn_lstm.conv2.W, cnn_lstm.conv2.b,
            #                             cnn_lstm.conv3.W, cnn_lstm.conv3.b,
            #                             cnn_lstm.conv4.W, cnn_lstm.conv4.b,
            #                             cnn_lstm.forward_lstm.W,
            #                             cnn_lstm.forward_lstm.U,
            #                             cnn_lstm.forward_lstm.b,
            #                             cnn_lstm.backward_lstm.W,
            #                             cnn_lstm.backward_lstm.U,
            #                             cnn_lstm.backward_lstm.b),
            #                            open('trained_e%i.zip' % epoch, 'wb'))

    s.close()
