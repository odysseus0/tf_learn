import tensorflow as tf

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)
y_pred = linear_model(x)

loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optim = tf.train.AdamOptimizer(0.01)
train = optim.minimize(loss)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

