import tensorflow as tf

content_layers = [
    # 'block4_conv1'
    'block4_conv4'
]


def vgg(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    net = tf.keras.applications.vgg19.VGG19(
        input_tensor=input,
        weights='imagenet',
        include_top=False
    )

    output = None

    for layer in net.layers:
        if layer.name in content_layers:
            output = layer.output
            layer.trainable = False
            break

    return tf.keras.models.Model(input, output)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

# def GramMatrix(x):
#     fi = tf.transpose(x, perm=(0, 3, 1, 2))
#     fi = tf.reshape(x, (fi.shape[0], fi.shape[1], fi.shape[2]*fi.shape[3]))
#     size = fi.shape[2]
#     return tf.linalg.matmul(fi, fi, transpose_b=True)/size

def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)

class PerceptualLoss:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.model = vgg(image_shape)

    def calculate_loss(self, original_image, image_tensor):
        content_output = self.model(original_image)
        prediction = self.model(image_tensor)
        content_loss = tf.math.reduce_mean(
            tf.math.square(content_output-prediction))
        return content_loss

    def content_loss(self, img_embed_a, img_embed_b):
        return tf.math.reduce_mean(tf.math.abs(img_embed_a-img_embed_b))

    def style_loss(self, img_embed_a, img_embed_b):
        gram_a = gram(img_embed_a)
        gram_b = gram(img_embed_b)
        return tf.math.reduce_mean(tf.math.abs(gram_a-gram_b))

    def __call__(self, image):
        '''
        image: 4D tensor, batch of RGB images in range [-1, 1]
        '''
        # convert images to be in correct format. First convert to rgb in range [0, 255]
        image = (image+1.0)*0.5*255.0
        # use preprocess function to convert to bgr and center data
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return self.model(image)
