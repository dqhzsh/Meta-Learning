from tensorflow import keras


class Model(keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = keras.layers.Dense(64, activation=keras.activations.tanh, input_shape=(1, ))
        self.l2 = keras.layers.Dense(64, activation=keras.activations.tanh)
        self.out = keras.layers.Dense(1)

    def call(self, x, training=None, mask=None):
        x = self.l1(x)
        x = self.l2(x)
        y = self.out(x)
        return y

    def build(self, input_shape):
        self.l1.build(input_shape)
        # 计算第一层的输出形状
        output_shape_l1 = self.l1.compute_output_shape(input_shape)
        # 根据第一层的输出形状构建第二层
        self.l2.build(output_shape_l1)
        # 计算第二层的输出形状
        output_shape_l2 = self.l2.compute_output_shape(output_shape_l1)
        # 根据第二层的输出形状构建输出层
        self.out.build(output_shape_l2)
