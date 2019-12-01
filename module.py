import tensorflow as tf
import tensorflow.python as tp
import tensorflow.python.keras.backend as K


class Ranger(tf.keras.optimizers.Optimizer):
    """
        tf.keras optimizer, combining RAdam (https://arxiv.org/abs/1908.03265) and Lookahead (https://arxiv.org/abs/1907.0861) optimization
        can be fed into the model.compile method of a tf.keras model as an optimizer
        ...
        Attributes
        ----------
        learning_rate / lr : float
            step size to take for RAdam optimizer (depending on gradient)
        beta_1 : float
            parameter that specifies the exponentially moving average length for momentum (0<=beta_1<=1)
        beta_2 : float
            parameter that specifies the exponentially moving average length for variance (0<=beta_2<=1)
        epsilon : float
            small number to cause stability for variance division
        weight_decay : float
            number with which the weights of the model are multiplied each iteration (0<=weight_decay<=1)
        amsgrad : bool
            parameter that specifies whether to use amsgrad version of Adam (https://arxiv.org/abs/1904.03590)
        total_steps : int
            total number of training steps
        warmup_proportion : float
            the proportion of updated over which the learning rate is increased from min learning rate to learning rate (0<=warmup_proportion<=1)
        min_lr : float
            learning rate at which the optimizer starts
        k : int
            parameter that specifies after how many steps the lookahead step backwards should be applied
        alpha : float
            parameter that specifies how much in the direction of the fast weights should be moved (0<=alpha<=1)
    """
    def __init__(self,
                 learning_rate=1e-3,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-5,
                 weight_decay=0.,
                 amsgrad=False,
                 total_steps=0,
                 warmup_proportion=0.1,
                 min_lr=0.,
                 k=6,
                 alpha=0.5,
                 **kwargs):
        class RAdam(tf.keras.optimizers.Optimizer):
            def __init__(self):
                super(RAdam, self).__init__(name='RAdam')
                self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
                self._set_hyper('beta_1', beta_1)
                self._set_hyper('beta_2', beta_2)
                self._set_hyper('decay', self._initial_decay)
                self._set_hyper('weight_decay', weight_decay)
                self._set_hyper('total_steps', float(total_steps))
                self._set_hyper('warmup_proportion', warmup_proportion)
                self._set_hyper('min_lr', min_lr)
                self.epsilon = epsilon or K.epsilon()
                self.amsgrad = amsgrad
                self._initial_weight_decay = weight_decay
                self._initial_total_steps = total_steps

            def _create_slots(self, var_list):
                for var in var_list:
                    self.add_slot(var, 'm')
                for var in var_list:
                    self.add_slot(var, 'v')
                if self.amsgrad:
                    for var in var_list:
                        self.add_slot(var, 'vhat')

            def set_weights(self, weights):
                params = self.weights
                num_vars = int((len(params) - 1) / 2)
                if len(weights) == 3 * num_vars + 1:
                    weights = weights[:len(params)]
                super(RAdam, self).set_weights(weights)

            def _resource_apply_dense(self, grad, var):
                var_dtype = var.dtype.base_dtype
                lr_t = self._decayed_lr(var_dtype)
                m = self.get_slot(var, 'm')
                v = self.get_slot(var, 'v')
                beta_1_t = self._get_hyper('beta_1', var_dtype)
                beta_2_t = self._get_hyper('beta_2', var_dtype)
                epsilon_t = tp.ops.convert_to_tensor(self.epsilon, var_dtype)
                local_step = tp.math_ops.cast(self.iterations + 1, var_dtype)
                beta_1_power = tp.math_ops.pow(beta_1_t, local_step)
                beta_2_power = tp.math_ops.pow(beta_2_t, local_step)
                if self._initial_total_steps > 0:
                    total_steps = self._get_hyper('total_steps', var_dtype)
                    warmup_steps = total_steps * self._get_hyper('warmup_proportion', var_dtype)
                    min_lr = self._get_hyper('min_lr', var_dtype)
                    decay_steps = K.maximum(total_steps - warmup_steps, 1)
                    decay_rate = (min_lr - lr_t) / decay_steps
                    lr_t = tf.where(
                        local_step <= warmup_steps,
                        lr_t * (local_step / warmup_steps),
                        lr_t + decay_rate * K.minimum(local_step - warmup_steps, decay_steps),
                    )
                sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
                sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
                m_t = tp.state_ops.assign(m, beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking)
                m_corr_t = m_t / (1.0 - beta_1_power)

                v_t = tp.state_ops.assign(v, beta_2_t * v + (1.0 - beta_2_t) * tp.math_ops.square(grad), use_locking=self._use_locking)
                if self.amsgrad:
                    vhat = self.get_slot(var, 'vhat')
                    vhat_t = tp.state_ops.assign(vhat, tp.math_ops.maximum(vhat, v_t), use_locking=self._use_locking)
                    v_corr_t = tp.math_ops.sqrt(vhat_t / (1.0 - beta_2_power))
                else:
                    vhat_t = None
                    v_corr_t = tp.math_ops.sqrt(v_t / (1.0 - beta_2_power))
                r_t = tp.math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) / (sma_inf - 2.0) * sma_inf / sma_t)
                var_t = tf.where(sma_t > 4.0, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)
                if self._initial_weight_decay > 0.0:
                    var_t += self._get_hyper('weight_decay', var_dtype) * var
                var_update = tp.state_ops.assign_sub(var,
                                                  lr_t * var_t,
                                                  use_locking=self._use_locking)
                updates = [var_update, m_t, v_t]
                if self.amsgrad:
                    updates.append(vhat_t)
                return tp.control_flow_ops.group(*updates)

            def _resource_apply_sparse(self, grad, var, indices):
                var_dtype = var.dtype.base_dtype
                lr_t = self._decayed_lr(var_dtype)
                beta_1_t = self._get_hyper('beta_1', var_dtype)
                beta_2_t = self._get_hyper('beta_2', var_dtype)
                epsilon_t = tp.ops.convert_to_tensor(self.epsilon, var_dtype)
                local_step = tp.math_ops.cast(self.iterations + 1, var_dtype)
                beta_1_power = tp.math_ops.pow(beta_1_t, local_step)
                beta_2_power = tp.math_ops.pow(beta_2_t, local_step)
                if self._initial_total_steps > 0:
                    total_steps = self._get_hyper('total_steps', var_dtype)
                    warmup_steps = total_steps * self._get_hyper('warmup_proportion', var_dtype)
                    min_lr = self._get_hyper('min_lr', var_dtype)
                    decay_steps = K.maximum(total_steps - warmup_steps, 1)
                    decay_rate = (min_lr - lr_t) / decay_steps
                    lr_t = tf.where(
                        local_step <= warmup_steps,
                        lr_t * (local_step / warmup_steps),
                        lr_t + decay_rate * K.minimum(local_step - warmup_steps, decay_steps),
                    )
                sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
                sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)
                m = self.get_slot(var, 'm')
                m_scaled_g_values = grad * (1 - beta_1_t)
                m_t = tp.state_ops.assign(m, m * beta_1_t, use_locking=self._use_locking)
                with tp.ops.control_dependencies([m_t]):
                    m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
                m_corr_t = m_t / (1.0 - beta_1_power)
                v = self.get_slot(var, 'v')
                v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
                v_t = tp.state_ops.assign(v, v * beta_2_t, use_locking=self._use_locking)
                with tp.ops.control_dependencies([v_t]):
                    v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)
                if self.amsgrad:
                    vhat = self.get_slot(var, 'vhat')
                    vhat_t = tp.state_ops.assign(vhat, tp.math_ops.maximum(vhat, v_t), use_locking=self._use_locking)
                    v_corr_t = tp.math_ops.sqrt(vhat_t / (1.0 - beta_2_power))
                else:
                    vhat_t = None
                    v_corr_t = tp.math_ops.sqrt(v_t / (1.0 - beta_2_power))
                r_t = tp.math_ops.sqrt((sma_t - 4.0) / (sma_inf - 4.0) * (sma_t - 2.0) / (sma_inf - 2.0) * sma_inf / sma_t)
                var_t = tf.where(sma_t >= 5.0, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t)
                if self._initial_weight_decay > 0.0:
                    var_t += self._get_hyper('weight_decay', var_dtype) * var
                var_update = self._resource_scatter_add(var, indices, tf.gather(-lr_t * var_t, indices))
                updates = [var_update, m_t, v_t]
                if self.amsgrad:
                    updates.append(vhat_t)
                return tp.control_flow_ops.group(*updates)

            def get_config(self):
                config = super(RAdam, self).get_config()
                config.update({
                    'learning_rate': self._serialize_hyperparameter('learning_rate'),
                    'beta_1': self._serialize_hyperparameter('beta_1'),
                    'beta_2': self._serialize_hyperparameter('beta_2'),
                    'decay': self._serialize_hyperparameter('decay'),
                    'weight_decay': self._serialize_hyperparameter('weight_decay'),
                    'epsilon': self.epsilon,
                    'amsgrad': self.amsgrad,
                    'total_steps': self._serialize_hyperparameter('total_steps'),
                    'warmup_proportion': self._serialize_hyperparameter('warmup_proportion'),
                    'min_lr': self._serialize_hyperparameter('min_lr'),
                })
                return config
        
        super(Ranger, self).__init__(name='Ranger')
        learning_rate = kwargs.get('lr', learning_rate)
        self._optimizer = RAdam()
        self._set_hyper('k', k)
        self._set_hyper('alpha', alpha)
        self._initialized = False

    def _create_slots(self, var_list):
        self._optimizer._create_slots(var_list=var_list)
        for var in var_list:
            self.add_slot(var, 'slow')

    def _create_hypers(self):
        self._optimizer._create_hypers()

    def _prepare(self, var_list):
        return self._optimizer._prepare(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None):
        self._optimizer._iterations = self.iterations
        return super(Ranger, self).apply_gradients(grads_and_vars, name)

    def _init_op(self, var):
        slow_var = self.get_slot(var, 'slow')
        return slow_var.assign(
            tf.where(
                tf.equal(self.iterations,
                         tf.constant(0, dtype=self.iterations.dtype)),
                var,
                slow_var,
            ),
            use_locking=self._use_locking)

    def _look_ahead_op(self, var):
        var_dtype = var.dtype.base_dtype
        slow_var = self.get_slot(var, 'slow')
        local_step = tf.cast(self.iterations + 1, tf.dtypes.int64)
        k = self._get_hyper('k', tf.dtypes.int64)
        alpha = self._get_hyper('alpha', var_dtype)
        step_back = slow_var + alpha * (var - slow_var)
        sync_cond = tf.equal(
            tf.math.floordiv(local_step, k) * k,
            local_step)
        with tf.control_dependencies([step_back]):
            slow_update = slow_var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    slow_var,
                ),
                use_locking=self._use_locking)
            var_update = var.assign(
                tf.where(
                    sync_cond,
                    step_back,
                    var,
                ),
                use_locking=self._use_locking)
        return tf.group(slow_update, var_update)

    @property
    def weights(self):
        return self._weights + self._optimizer.weights

    def _resource_apply_dense(self, grad, var):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_dense(grad, var)  # pylint: disable=protected-access
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def _resource_apply_sparse(self, grad, var, indices):
        init_op = self._init_op(var)
        with tf.control_dependencies([init_op]):
            train_op = self._optimizer._resource_apply_sparse(  # pylint: disable=protected-access
                grad, var, indices)
            with tf.control_dependencies([train_op]):
                look_ahead_op = self._look_ahead_op(var)
        return tf.group(init_op, train_op, look_ahead_op)

    def get_config(self):
        config = {
            'optimizer': tf.keras.optimizers.serialize(self._optimizer),
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decay': self._serialize_hyperparameter('decay'),
            'weight_decay': self._serialize_hyperparameter('weight_decay'),
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad,
            'total_steps': self._serialize_hyperparameter('total_steps'),
            'warmup_proportion': self._serialize_hyperparameter('warmup_proportion'),
            'min_lr': self._serialize_hyperparameter('min_lr'),
            'k': self._serialize_hyperparameter('k'),
            'alpha': self._serialize_hyperparameter('alpha'),
        }
        base_config = super(Ranger, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)
