from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from model_utils import frontend_3D, backend_resnet, conv_backend, stacked_lstm, MultiLayerOutput  # blstm_encoder, lstm_encoder
from metrics import char_accuracy, flatten_list
from data_utils import get_data_paths, get_number_of_steps, get_training_data_batch, get_inference_data_batch
from tqdm import tqdm
from video2tfrecord3 import decrypt
from tensorflow.contrib.rnn import LSTMStateTuple


class BasicModel:
    """
    Model class with basic functionality

    options: (dict) all  model and training options/parameters
    """

    def __init__(self, options):

        self.options = options
        self.data_paths = get_data_paths(self.options)

        self.number_of_steps_per_epoch, self.number_of_steps = \
            get_number_of_steps(self.data_paths, self.options)

        self.init_global_step()

        if self.options['mode'] not in ['train', 'test']:
            raise ValueError("options.mode must be either 'train' or 'test'")

        if self.options['save'] or self.options['restore']:
            self.saver = tf.train.Saver(var_list=tf.global_variables(),
                                        max_to_keep=self.options['num_models_saved'])

        if self.options['save_graph'] or self.options['save_summaries']:
            self.writer = tf.summary.FileWriter(self.options['save_dir'])

    def build_train_graph(self):
        pass

    def train(self, sess, number_of_steps=None, reset_global_step=False):
        pass

    def build_inference_graph(self):
        pass

    def predict(self, sess, num_steps=None):
        pass

    @property
    def learn_rate_decay_steps(self):
        if self.options['num_decay_steps'] is None:  # decay every epoch
            num_decay_steps = self.number_of_steps_per_epoch
        elif type(self.options['num_decay_steps']) is float:   # decay at a proportion to steps per epoch
            num_decay_steps = int(self.options['num_decay_steps'] * self.number_of_steps_per_epoch)
        else:  # explicitly specify decay steps
            num_decay_steps = self.options['num_decay_steps']
        return num_decay_steps

    def init_global_step(self, value=0):
        print("initializing global step at %d" % value)
        self.global_step = tf.Variable(value, trainable=False)
        self.increment_global_step = tf.assign(self.global_step, self.global_step + 1)

    def restore_model(self, sess):
        print("reading model %s ..." % self.options['restore_model'])
        self.saver.restore(sess, self.options['restore_model'])
        print("model restored.")

    def save_model(self, sess, save_path):
        print("saving model %s ..." % save_path)
        self.saver.save(sess=sess, save_path=save_path)
        print("model saved.")

    def save_graph(self, sess):
        # writer = tf.summary.FileWriter(self.options['graph_save_path'])
        self.writer.add_graph(sess.graph)
        # writer = tf.summary.FileWriter(logdir='logdir', graph=graph)
        self.writer.flush()
        # self.writer.close()

    def save_summaries(self, sess, summaries):
        s, gs = sess.run([summaries, self.global_step])
        self.writer.add_summary(s, gs)
        self.writer.flush()


class VisualFeaturePretrainModel(BasicModel):
    """
    03/07/18 : The 2nd seq2seq model trained on MVLRS
    Additions:
    ---------
    - mean encoder out as initial decoder hidden state. fc layer transforms concatenated
        blstm state to decoder size
    - added dropout after resnet
    """

    def __init__(self, options):
        super(VisualFeaturePretrainModel, self).__init__(options=options)

        if self.options['mode'] == 'train':
            self.train_era_step = self.options['train_era_step']
            self.encoder_inputs, self.target_labels = get_training_data_batch(self.data_paths, self.options)
            self.num_decay_steps = self.learn_rate_decay_steps
            # if self.options['save_summaries']:
            #     tf.summary.image('sample_image', self.encoder_inputs[0, :, :, :, -1:], max_outputs=50)

            self.build_train_graph()

        elif self.options['mode'] == 'test':
            self.encoder_inputs, self.target_labels = get_inference_data_batch(self.data_paths, self.options)
            # self.max_decoding_steps = tf.to_int32(
            #     tf.round(self.options['max_out_len_multiplier'] * tf.to_float(self.max_input_len)))
            self.build_inference_graph()

        else:
            raise ValueError("options.mode must be either 'train' or 'test'")

        if self.options['save_summaries']:  # does this merge zero summaries? CHECK
            self.merged_summaries = tf.summary.merge_all()

    def build_train_graph(self):

        final_size = 512  # must be 512 with current ResNet architecture
        if self.options['resnet_num_features'] != final_size:
            print("A dense layer is added to final ResNet layer to match desired number of classes")
            num_classes = self.options['resnet_num_features']
        else:
            num_classes = None

        if self.options['frontend_3d']:
            with tf.variable_scope('3dconv'):
                features_3d = frontend_3D(self.encoder_inputs)

            with tf.variable_scope('resnet'):
                self.features_res = backend_resnet(x_input=features_3d,
                                              resnet_size=self.options['resnet_size'],
                                              final_size=final_size,
                                              num_classes=num_classes,
                                              training=True,
                                              frontend_3d=True)
        else:
            with tf.variable_scope('resnet'):
                self.features_res = backend_resnet(x_input=self.encoder_inputs,
                                              resnet_size=self.options['resnet_size'],
                                              final_size=final_size,
                                              num_classes=num_classes,
                                              training=True,
                                              frontend_3d=False)
        if self.options['res_features_keep_prob'] != 1.0:
            self.features_res = tf.layers.dropout(self.features_res,
                                             rate=1. - self.options['res_features_keep_prob'],
                                             training=True,
                                             name='features_res_dropout')

        with tf.variable_scope('temporal_conv'):
            self.logits = conv_backend(self.features_res, self.options)

        with tf.variable_scope('train_metrics'):
            self.train_loss = tf.reduce_mean(
                tf.losses.softmax_cross_entropy(self.target_labels, self.logits))
            correct_prediction = tf.equal(tf.argmax(self.logits, 1),
                                          tf.argmax(self.target_labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            if self.options['save_summaries']:
                tf.summary.scalar('loss', self.train_loss)
                tf.summary.scalar('accuracy', self.accuracy)

        with tf.variable_scope('optimizer'):
            params = tf.trainable_variables()
            self.gradients = tf.gradients(self.train_loss, params)

            initial_learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            learn_rate = tf.train.exponential_decay(learning_rate=initial_learn_rate, global_step=self.global_step,
                                                    decay_steps=self.num_decay_steps, decay_rate=self.options['decay_rate'],
                                                    staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learn_rate)
            self.update_step = self.optimizer.apply_gradients(zip(self.gradients, params), global_step=self.global_step)

    def train(self, sess, number_of_steps=None, reset_global_step=False):

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = self.optimizer.minimize(self.train_loss)

        if number_of_steps is not None:
            assert type(number_of_steps) is int
            start_epoch = 0
            num_epochs = 1
        else:
            number_of_steps = self.number_of_steps_per_epoch
            start_epoch = self.options['start_epoch']
            num_epochs = self.options['num_epochs']

        if self.options['restore']:
            self.restore_model(sess)

        if reset_global_step:
            initial_global_step = self.global_step.assign(0)
            sess.run(initial_global_step)

        for epoch in range(start_epoch, start_epoch + num_epochs):

            for step in range(number_of_steps):

                _, _, loss, acc, lr, gs = sess.run(
                    [self.update_step,
                     self.increment_global_step,
                     self.train_loss,
                     self.accuracy,
                     self.optimizer._lr_t,  # _learning_rate,  # .optimizer._lr_t,
                     self.global_step])
                print("%d,%d,%d,%d,%.4f,%.4f,%.8f"
                      % (epoch,
                         self.options['num_epochs'],
                         step,
                         self.number_of_steps_per_epoch,
                         loss, acc, lr))

                if self.options['save_summaries']:
                    self.writer.add_summary(summ, gs)

                if (self.train_era_step % self.options['save_steps'] == 0) and self.options['save']:
                    # print("saving model at global step %d..." % global_step)
                    self.save_model(sess=sess, save_path=self.options['save_model'] + "_epoch%d_step%d" % (epoch, step))
                    # print("model saved.")

                self.train_era_step += 1

        # save before closing
        if self.options['save']:
            self.save_model(sess=sess, save_path=self.options['save_model'] + "_final")

    def build_inference_graph(self):
        pass

    def predict(self, sess, num_steps=None):
        pass



class Model3(BasicModel):
    """
    03/07/18 : The 2nd seq2seq model trained on MVLRS
    Additions:
    ---------
    - mean encoder out as initial decoder hidden state. fc layer transforms concatenated
        blstm state to decoder size
    - added dropout after resnet
    """

    def __init__(self, options):
        super(Model3, self).__init__(options=options)

        if self.options['mode'] == 'train':
            self.train_era_step = self.options['train_era_step']
            self.encoder_inputs, self.target_labels, self.decoder_inputs, self.encoder_inputs_lengths, \
            self.target_labels_lengths, self.decoder_inputs_lengths, self.max_input_len = \
                get_training_data_batch(self.data_paths, self.options)
            # if self.options['num_decay_steps'] is not None:
            #     self.num_decay_steps = self.options['num_decay_steps']
            # else:
            #     self.num_decay_steps = self.number_of_steps_per_epoch
            self.num_decay_steps = self.learn_rate_decay_steps
            # if self.options['save_summaries']:
            #     tf.summary.image('sample_image', self.encoder_inputs[0, :, :, :, -1:], max_outputs=50)

            self.build_train_graph()

        elif self.options['mode'] == 'test':
            self.encoder_inputs, self.target_labels, self.decoder_inputs, self.encoder_inputs_lengths, \
            self.target_labels_lengths, self.decoder_inputs_lengths, self.max_input_len = \
                get_inference_data_batch(self.data_paths, self.options)
            self.max_decoding_steps = tf.to_int32(
                tf.round(self.options['max_out_len_multiplier'] * tf.to_float(self.max_input_len)))
            self.build_inference_graph()

        if self.options['save_summaries']:
            self.merged_summaries = tf.summary.merge_all()

    def build_train_graph(self):
        ss_prob = self.options['ss_prob']

        final_size = 512  # must be 512 with current ResNet architecture
        if self.options['resnet_num_features'] != final_size:
            print("A dense layer is added to final ResNet layer to match desired number of classes")
            num_classes = self.options['resnet_num_features']
        else:
            num_classes = None

        if self.options['frontend_3d']:
            with tf.variable_scope('3dconv'):
                features_3d = frontend_3D(self.encoder_inputs)

            with tf.variable_scope('resnet'):
                self.features_res = backend_resnet(x_input=features_3d,
                                              resnet_size=self.options['resnet_size'],
                                              final_size=final_size,
                                              num_classes=num_classes,
                                              training=True,
                                              frontend_3d=True)
        else:
            with tf.variable_scope('resnet'):
                self.features_res = backend_resnet(x_input=self.encoder_inputs,
                                              resnet_size=self.options['resnet_size'],
                                              final_size=final_size,
                                              num_classes=num_classes,
                                              training=True,
                                              frontend_3d=False)
        if self.options['res_features_keep_prob'] != 1.0:
            self.features_res = tf.layers.dropout(self.features_res,
                                             rate=1. - self.options['res_features_keep_prob'],
                                             training=True,
                                             name='features_res_dropout')


        with tf.variable_scope('encoder'):
            encoder_out, encoder_hidden = stacked_lstm(
                input_forw=self.features_res, num_layers=self.options['encoder_num_layers'],
                num_hidden=self.options['encoder_num_hidden'],
                residual=self.options['residual_encoder'], use_peepholes=True)
            # if self.options['bidir_encoder']:
            #     encoder_out, encoder_hidden = blstm_encoder(features_res, self.options)
            # else:
                # encoder_out, encoder_hidden = lstm_encoder(features_res, self.options)


        with tf.variable_scope('decoder'):
            self.sampling_prob = tf.constant(ss_prob, dtype=tf.float32)
            helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                self.decoder_inputs,
                self.decoder_inputs_lengths,
                self.sampling_prob)

            # decoder_cell = [tf.contrib.rnn.LSTMCell(self.options['decoder_num_hidden'])
            #                 for _ in range(self.options['encoder_num_layers'])]
            # decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell, state_is_tuple=True)
            decoder_cell = stacked_lstm(
                input_forw=self.features_res, num_layers=self.options['decoder_num_layers'],
                num_hidden=self.options['decoder_num_hidden'],
                residual=self.options['residual_decoder'], use_peepholes=True,
                return_cell=True)

            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.options['attention_num_units'],  # The depth of the query mechanism.
                memory=encoder_out,  # The memory to query; usually the output of an RNN encoder
                memory_sequence_length=self.encoder_inputs_lengths,  # Sequence lengths for the batch
                # entries in memory. If provided, the memory tensor rows are masked with zeros for values
                # past the respective sequence lengths.
                normalize=self.options['norm_attention_layer'],  # boolean. Whether to normalize the energy term.
                name='BahdanauAttention')
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.options['attention_layer_size'],
                alignment_history=False,
                cell_input_fn=None,
                output_attention=True, # Luong: True, Bahdanau: False ?
                initial_cell_state=None,
                name=None)

            decoder_init_state = self.get_decoder_init_state(encoder_hidden, attn_cell)

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=attn_cell,
                helper=helper,
                initial_state=decoder_init_state,
                output_layer=MultiLayerOutput([self.options['decoder_output_num_units'],
                                               self.options['num_classes']]))
            outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=self.options['max_out_len'])
            decoder_outputs = outputs.rnn_output
            decoder_greedy_pred = tf.argmax(decoder_outputs, axis=2)

        with tf.variable_scope('loss_function'):
            target_weights = tf.sequence_mask(self.target_labels_lengths,  # +1 for <eos> token
                                              maxlen=None,  # data_options['max_out_len']+1,
                                              dtype=tf.float32)
            self.train_loss = tf.contrib.seq2seq.sequence_loss(
                decoder_outputs, self.target_labels, weights=target_weights)
            if self.options['save_summaries']:
                tf.summary.scalar('cross_entropy', self.train_loss)

        with tf.variable_scope('training_parameters'):
            params = tf.trainable_variables()
            # clip by gradients
            max_gradient_norm = tf.constant(10., dtype=tf.float32, name='max_gradient_norm')
            self.gradients = tf.gradients(self.train_loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, max_gradient_norm)

            # Optimization
            initial_learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            learn_rate = tf.train.exponential_decay(learning_rate=initial_learn_rate, global_step=self.global_step,
                                                    decay_steps=self.num_decay_steps, decay_rate=self.options['decay_rate'],
                                                    staircase=True)
            # self.optimizer = tf.train.MomentumOptimizer(learn_rate, momentum=0.9)#.minimize(cross_entropy)
            # learn_rate = tf.constant(self.options['learn_rate'], tf.float32)
            self.optimizer = tf.train.AdamOptimizer(learn_rate)

            self.update_step = self.optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
            self.accuracy, self.accuracy2 = char_accuracy(self.target_labels, decoder_greedy_pred, self.target_labels_lengths)

            if self.options['save_summaries']:
                tf.summary.scalar('accuracy', self.accuracy2)

    def train(self, sess, number_of_steps=None, reset_global_step=False):

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     train_op = self.optimizer.minimize(self.train_loss)

        if number_of_steps is not None:
            assert type(number_of_steps) is int
            start_epoch = 0
            num_epochs = 1
        else:
            number_of_steps = self.number_of_steps_per_epoch
            start_epoch = self.options['start_epoch']
            num_epochs = self.options['num_epochs']

        if self.options['restore']:
            self.restore_model(sess)

        if reset_global_step:
            initial_global_step = self.global_step.assign(0)
            sess.run(initial_global_step)

        for epoch in range(start_epoch, start_epoch + num_epochs):

            for step in range(number_of_steps):

                _, _, loss, acc, acc2, lr, sp, summ, gs = sess.run(
                    [self.update_step,
                     self.increment_global_step,
                     self.train_loss,
                     self.accuracy,
                     self.accuracy2,
                     self.optimizer._lr_t,  # _learning_rate,  # .optimizer._lr_t,
                     self.sampling_prob,
                     self.merged_summaries,
                     self.global_step])
                print("%d,%d,%d,%d,%.4f,%.4f,%.4f,%.8f,%.4f"
                      % (epoch,
                         self.options['num_epochs'],
                         step,
                         self.number_of_steps_per_epoch,
                         loss, acc, acc2, lr, sp))

                if self.options['save_summaries']:
                    self.writer.add_summary(summ, gs)

                if (self.train_era_step % self.options['save_steps'] == 0) and self.options['save']:
                    # print("saving model at global step %d..." % global_step)
                    self.save_model(sess=sess, save_path=self.options['save_model'] + "_epoch%d_step%d" % (epoch, step))
                    # print("model saved.")

                self.train_era_step += 1

        # save before closing
        if self.options['save']:
            self.save_model(sess=sess, save_path=self.options['save_model'] + "_final")

    def build_inference_graph(self):
        """
        with beam search decoder
        """
        with tf.variable_scope('resnet'):
            features_res = backend_resnet(x_input=self.encoder_inputs,
                                          resnet_size=self.options['resnet_size'],
                                          num_classes=self.options['resnet_num_features'],
                                          training=False)
            if self.options['res_features_keep_prob'] != 1.0:
                features_res = tf.layers.dropout(features_res,
                                                 rate=1.-self.options['res_features_keep_prob'],
                                                 training=False,
                                                 name='features_res_dropout')

        with tf.variable_scope('encoder_blstm'):
            encoder_out, encoder_hidden = blstm_encoder(features_res, self.options)

        with tf.variable_scope('decoder_lstm'):
            embedding_decoder = tf.diag(tf.ones(self.options['num_classes']))
            decoder_cell = [tf.contrib.rnn.LSTMCell(self.options['decoder_num_hidden'])
                            for _ in range(self.options['encoder_num_layers'])]
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(decoder_cell, state_is_tuple=True)
            encoder_out_beam = tf.contrib.seq2seq.tile_batch(
                encoder_out, multiplier=self.options['beam_width'])
            encoder_inputs_lengths_beam = tf.contrib.seq2seq.tile_batch(
                self.encoder_inputs_lengths, multiplier=self.options['beam_width'])
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.options['decoder_num_hidden'],
                memory=encoder_out_beam,
                memory_sequence_length=encoder_inputs_lengths_beam,
                normalize=self.options['norm_attention_layer'],
                name='BahdanauAttention')
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=self.options['attention_layer_size'],
                alignment_history=False,
                cell_input_fn=None,
                output_attention=False,
                initial_cell_state=None,
                name=None)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attn_cell, self.options['num_classes'])

            if self.options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
                init_state = self.get_decoder_init_state(encoder_hidden)
                decoder_init_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
                        cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))
            else:  # use zero state
                decoder_init_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width'])

            # decoder_init_state = out_cell.zero_state(
            #     dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width'])
            # init_state = self.get_decoder_init_state(encoder_hidden)
            # decoder_init_state = out_cell.zero_state(
            #     dtype=tf.float32, batch_size=self.options['batch_size'] * self.options['beam_width']).clone(
            #     cell_state=tf.contrib.seq2seq.tile_batch(init_state, self.options['beam_width']))

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=out_cell,
                embedding=embedding_decoder,
                start_tokens=tf.fill([self.options['batch_size']], 27),
                end_token=28,
                initial_state=decoder_init_state,
                beam_width=self.options['beam_width'],
                length_penalty_weight=0.0)
            outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False, maximum_iterations=self.max_decoding_steps,
                swap_memory=True)
            beam_search_outputs = outputs.predicted_ids
            self.best_output = beam_search_outputs[:, :, 0]   ### IS THIS THE BEST???
            # CHECK : https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/FinalBeamSearchDecoderOutput

    def predict(self, sess, num_steps=None):
        # sometimes decoder outputs -1s at the end of the sequence, replace those with 0s
        def replace_(s, vout=-1, vin=0):
            s[s == vout] = vin
            return s
        if self.options['restore']:
            self.restore_model(sess)
        if num_steps is None:
            num_steps = self.number_of_steps_per_epoch
        res = []
        for step in tqdm(range(num_steps)):
            tl, pred = sess.run([self.target_labels, self.best_output])
            res.append([tl, pred])
        labels_ = flatten_list([decrypt(res_[0]) for res_ in res])
        predictions_ = flatten_list([decrypt(replace_(res_[1])) for res_ in res])
        return labels_, predictions_

    def get_decoder_init_state(self, encoder_states, cell):
        """
        initial values for (unidirectional lstm) decoder network from (equal depth bidirectional lstm)
        encoder hidden states. initially, the states of the forward and backward networks are concatenated
        and a fully connected layer is defined for each lastm parameter (c, h) mapping from encoder to
        decoder hidden size state
        """
        if not self.options['bidir_encoder']:
            if self.options['encoder_state_as_decoder_init']:  # use encoder state for decoder init
                init_state = self.get_decoder_init_state(encoder_states)
                decoder_init_state = cell.zero_state(dtype=tf.float32, batch_size=self.options['batch_size']
                                                         ).clone(cell_state=init_state)
            else:  # use zero state
                decoder_init_state = cell.zero_state(dtype=tf.float32, batch_size=self.options['batch_size'])
            return decoder_init_state
        else:
            raise NotImplemented

        # self.encoder_state = []
        #
        # for i in range(self.num_layers):
        #     if isinstance(encoder_fw_state[i], LSTMStateTuple):
        #
        #         encoder_state_c = tf.concat((encoder_fw_state[i].c, encoder_bw_state[i].c), 1,
        #                                     name='bidirectional_concat_c')
        #         encoder_state_h = tf.concat((encoder_fw_state[i].h, encoder_bw_state[i].h), 1,
        #                                     name='bidirectional_concat_h')
        #         encoder_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
        #     elif isinstance(encoder_fw_state[i], tf.Tensor):
        #         encoder_state = tf.concat((encoder_fw_state[i], encoder_bw_state[i]), 1, name='bidirectional_concat')
        #     self.encoder_state.append(encoder_state)
        #
        # self.encoder_state = tuple(self.encoder_state)

        # def encoder2decoder_init_state(encoder_hidden, decoder_hidden_size, name='encoder_decoder_hidden'):
        #     decoder_hidden = tf.layers.dense(inputs=encoder_hidden, units=decoder_hidden_size,
        #                                          activation=tf.nn.relu, name=name)
        #     return decoder_hidden
        #
        # encoder_depth_ = len(encoder_states[0])
        # # init_state = [LSTMStateTuple(c=tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=1),
        # #                              h=tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=1))
        # #               for i in range(encoder_depth_)]
        # init_state = [[tf.concat([encoder_states[0][i].c, encoder_states[1][i].c], axis=1),
        #                tf.concat([encoder_states[0][i].h, encoder_states[1][i].h], axis=1)]
        #               for i in range(encoder_depth_)]
        # init_state1 = [
        #     [encoder2decoder_init_state(state[0], self.options['decoder_num_hidden'], name="enc_c2dec_c_%d" % (i+1)),
        #      encoder2decoder_init_state(state[1], self.options['decoder_num_hidden'], name="enc_h2dec_h_%d" % (i+1))]
        #                for i, state in enumerate(init_state)]
        # init_state2 = [LSTMStateTuple(c=eh_state[0], h=eh_state[1]) for eh_state in init_state1]
        #
        # return tuple(init_state2)

    # def decoder_output_layer(self):
