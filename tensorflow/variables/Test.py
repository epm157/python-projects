import tensorflow as tf



my_variable = tf.Variable(tf.zeros([1., 2., 3.]))
print(my_variable)
print(my_variable.numpy())


#with tf.device('/devices:GPU:1'):
    #v = tf.Variable(tf.zeros([10, 10]))

v = tf.Variable(0.0)
w = v+1
print(w)

v.assign_add(1.0)
print(v)
print(v.read_value())


class MyModuleOne(tf.Module):
    def __init__(self):
        self.v0 = tf.Variable(1.0)
        self.vs = [tf.Variable(x) for x in range(10)]

class MyOtherModule(tf.Module):
    def __init__(self):
        self.m = MyModuleOne()
        self.v = tf.Variable(10)

m = MyOtherModule()
print(len(m.variables))



