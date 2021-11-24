import tensorflow as tf
import tensorflow_addons as tfa

initializer = tf.random_normal_initializer(0., 0.02)

BASE_SIZE = 64


class AlwaysDropout(tf.keras.layers.Dropout):
    '''
    Dropout that is active during training and evaluation
    '''
    def call(self, x):
        return super().call(x, training=True)


def downsample_block(filters, kernel_size, dropout=False, norm=None, use_spectral_norm=False):
    conv1 = tf.keras.layers.Conv2D(
        filters, kernel_size, strides=2, padding='same', kernel_initializer=initializer)
    if use_spectral_norm:
        conv1 = tfa.layers.SpectralNormalization(conv1)
    block = tf.keras.Sequential()
    block.add(conv1)
    if norm == 'BatchNorm':
        block.add(tf.keras.layers.BatchNormalization())
    elif norm == 'InstanceNorm':
        block.add(tfa.layers.InstanceNormalization())
    elif norm == 'LayerNorm':
        block.add(tf.keras.layers.LayerNormalization())
    elif norm is not None:
        raise Exception(f'Unknown normalization: {norm}')
    if dropout:
        block.add(AlwaysDropout(0.5))
    block.add(tf.keras.layers.LeakyReLU())
    return block

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()
        
    def build(self, input_shape):
        self.conv.build(input_shape)
        # self.inst_norm.build(self.conv.output.shape)
        super().build(input_shape)
        
    def call(self, x):
        x = self.conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        return x

class DSConv(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()
        self.conv_block = ConvBlock(filters, strides=strides)
        
    def build(self, input_shape):
        # self.depthwise_conv.build(input_shape)
        # self.inst_norm.build(input_shape)
        # self.conv_block.build(input_shape)
        super().build(input_shape)
        
    def call(self, x):
        x = self.depthwise_conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv_block(x)
        return x
    
class IRB(tf.keras.layers.Layer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same')
        self.inst_norm = tfa.layers.InstanceNormalization()
        self.conv_block = ConvBlock(filters=512, strides=1, kernel_size=1)
        
        self.conv = tf.keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same')
        self.inst_norm2 = tfa.layers.InstanceNormalization()
        
    def build(self, input_shape):
        # self.depthwise_conv.build(input_shape)
        # self.inst_norm.build(input_shape)
        # self.conv_block.build(input_shape)
        # self.conv.build(input_shape)
        # self.inst_norm2.build(input_shape)
        super().build(input_shape)
        
    def call(self, x):
        inp = x
        
        x = self.conv_block(x)
        x = self.depthwise_conv(x)
        x = self.inst_norm(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv(x)
        x = self.inst_norm2(x)
        
        return x + inp
    
class UpConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs) -> None:
        super().__init__(**kwargs)
        self.DSConv = DSConv(filters)

        
    def build(self, input_shape):
        self.DSConv.build(input_shape)
        super().build(input_shape)
        
    def call(self, x):
        x = tf.keras.layers.UpSampling2D(interpolation='bilinear')(x)
        x = self.DSConv(x)        
        return x
    
class DownConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs) -> None:
        super().__init__(**kwargs)
        self.DSConv1 = DSConv(filters, strides=1)
        self.DSConv2 = DSConv(filters, strides=2)
        
    def build(self, input_shape):
        # self.DSConv1.build(input_shape)
        # self.DSConv2.build(input_shape)
        super().build(input_shape)
        
    def call(self, x):
        # x1 = tf.nn.avg_pool(x, 2, 2, padding='VALID')
        x1 = tf.nn.avg_pool2d(x, 2, 2, padding='VALID')
        x1 = self.DSConv1(x1)        
        x2 = self.DSConv2(x)
        
        return x1 + x2

def ae_model(input_shape) -> tf.keras.Model:
    image = tf.keras.layers.Input(shape=input_shape)
    x = image
    
    x = ConvBlock(64)(x)
    x = ConvBlock(64)(x)
    
    x = DownConv(128)(x)
    x = ConvBlock(128)(x)
    x = DSConv(128)(x)
    
    x = DownConv(256)(x)
    x = ConvBlock(256)(x)
    
    for _ in range(8):
        x = IRB()(x)
    
    x = ConvBlock(256)(x)
    
    x = UpConv(128)(x)
    x = DSConv(128)(x)
    x = ConvBlock(128)(x)

    x = UpConv(64)(x)
    x = ConvBlock(64)(x)
    x = ConvBlock(64)(x)
    
    x = tf.keras.layers.Conv2D(3, kernel_size=3, activation='tanh', padding='same')(x)

    
    return tf.keras.Model(inputs=image, outputs=x)




def patch_gan_discriminator():
    layers = [
        downsample_block(BASE_SIZE, kernel_size=4, use_spectral_norm=True),
        downsample_block(BASE_SIZE*2, kernel_size=4,
                         use_spectral_norm=True, norm='InstanceNorm'),
        downsample_block(BASE_SIZE*4, kernel_size=4,
                         use_spectral_norm=True, norm='InstanceNorm'),
        downsample_block(BASE_SIZE*8, kernel_size=4,
                         use_spectral_norm=True, norm='InstanceNorm'),
        tfa.layers.SpectralNormalization(
            tf.keras.layers.Conv2D(1, kernel_size=4, strides=2)),
        tf.keras.layers.Activation(tf.keras.activations.sigmoid),
        # tf.keras.layers.GlobalAveragePooling2D()
    ]
    # layers = [tfa.layers.SpectralNormalization(l) if type(
    #     l) == tf.keras.layers.Conv2D else l for l in layers]
    model = tf.keras.models.Sequential(layers)
    return model
