import tensorflow as tf
import numpy as np
import model as m
from config import *
import time

from utils import DataLoader, draw_strokes_random_color, DataGenerator

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)

args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1

model = m.Model()
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
# loss_fn = m.compute_custom_loss
# for e in range(args.num_epochs):
#     print("epoch %d" % e)
#     data_loader.reset_batch_pointer()
#     for b in range(data_loader.num_batches):
#         tic = time.time()
#         with tf.GradientTape() as tape:

#             x, y, c_vec, c = data_loader.next_batch()
            
#             out = model([x, c_vec])
#             loss_value = loss_fn(y, out)

#         grads = tape.gradient(loss_value, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         if b % 1 == 0:
#             print('batches %d/%s, loss %g -- time: %s' % (b+1, data_loader.num_batches, loss_value, str(round(time.time() - tic, 2))))

#     model.save_weights('lstm_validator/checkpoint')

gen = DataGenerator(data_loader)

model.compile(optimizer, loss=m.CustomLoss(reduction=tf.keras.losses.Reduction.NONE), run_eagerly=False)
model.fit(gen, epochs=args.num_epochs, batch_size=args.batch_size, verbose=1)
model.save_weights('lstm_validator/checkpoint')