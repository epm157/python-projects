import tensorflow as tf
import matplotlib.pyplot as plt

w = tf.Variable(5.)

print(w.numpy())


def get_loss(w):
    return w**2

def get_grad(w):
    with tf.GradientTape() as tape:
        loss = get_loss(w)
    grad = tape.gradient(loss, w)
    return grad

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

losses = []

for i in range(1000):
    grad = get_grad(w)
    optimizer.apply_gradients(zip([grad], [w]))
    # print(get_loss(w).numpy())
    losses.append(get_loss(w))




'''
losses = []
for i in range(500):
    g = get_grad(w)
    optimizer.apply_gradients(zip([g], [w]))
    losses.append(get_loss(w))
'''

plt.plot(losses)
plt.show()
print(w.numpy())