import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Layer


def create_MLP2_model(input_dimension=20,
                      hidden1_units=20,
                      dropout_rate=0.3,
                      objective_function='binary_crossentropy'):
    inputs = Input(batch_shape=(None, input_dimension))
    hidden1 = Dense(name='H1', units=hidden1_units, input_dim=input_dimension, activation='relu')(inputs)
    dropout1 = Dropout(name='D1', rate=dropout_rate)(hidden1)
    output = Dense(name='Output', units=1, activation='sigmoid')(dropout1)

    model = Model(name='1hidden_layer', inputs=inputs, outputs=output)
    model.compile(loss=objective_function, optimizer='adam')
    return model


def create_MLP3_model(input_dimension=20,
                      hidden1_units=20,
                      hidden2_units=10,
                      dropout_rate=0.3,
                      objective_function='binary_crossentropy'):
    inputs = Input(batch_shape=(None, input_dimension))
    hidden1 = Dense(name='H1', units=hidden1_units, input_dim=input_dimension, activation='relu')(inputs)
    dropout1 = Dropout(name='D1', rate=dropout_rate)(hidden1)
    hidden2 = Dense(name='H2', units=hidden2_units, activation='relu')(dropout1)
    dropout2 = Dropout(name='D2', rate=dropout_rate)(hidden2)
    output = Dense(name='Output', units=1, activation='sigmoid')(dropout2)

    model = Model(name='2hidden_layer', inputs=inputs, outputs=output)
    model.compile(loss=objective_function, optimizer='adam')
    return model


def create_AE_model(input_dimension=20,
                    intermediate=25,
                    bottleneck=15,
                    objective_function='mean_squared_error'):
    inputs = Input(batch_shape=(None, input_dimension))
    encoder_h = Dense(name='encoder_hidden', units=intermediate, input_dim=input_dimension, activation='relu')(inputs)
    latent = Dense(name='bottleneck', units=bottleneck, activation='relu')(encoder_h)
    decoder_h = Dense(name='decoder_hidden', units=intermediate, activation='relu')(latent)
    outputs = Dense(name='outputs', units=input_dimension, activation='tanh')(decoder_h)

    model = Model(name='AutoEncoder', inputs=inputs, outputs=outputs)
    model.compile(loss=objective_function, optimizer='adam')
    return model


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    def __init__(self, input_dimension, intermediate, bottleneck, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.__create_model(input_dimension, intermediate, bottleneck)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def __create_model(self, input_dim, intermediate, bottleneck):
        # build the encoder
        inputs = Input(batch_shape=(None, input_dim))
        encoder_h = Dense(units=intermediate, input_dim=input_dim, activation='relu', name='encoder_hidden')(inputs)
        z_mean = Dense(bottleneck, name='z_mean')(encoder_h)
        z_log_var = Dense(bottleneck, name='z_log_var')(encoder_h)
        z = Sampling()([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        # build the decoder
        latent_inputs = Input(shape=(bottleneck,))
        decoder_h = Dense(intermediate, activation='relu', name='decoder_hidden')(latent_inputs)
        decoder_outputs = Dense(input_dim, activation='tanh')(decoder_h)
        self.decoder = Model(latent_inputs, decoder_outputs, name='decoder')

    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    def test_step(self, data):
        x, y = data
        total_loss = self.__calculate_loss(x, y)
        self.loss_tracker.update_state(total_loss)
        return self.__get_loss_dictionary()

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss = self.__calculate_loss(x, y)
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.loss_tracker.update_state(total_loss)
        return self.__get_loss_dictionary()

    def __get_loss_dictionary(self):
        return {
            "loss": self.loss_tracker.result(),
        }

    def __calculate_loss(self, x, y):
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.keras.losses.mean_squared_error(y, reconstruction)
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=-1))
        return reconstruction_loss + kl_loss

    def call(self, input_features):
        _, _, z = self.encoder(input_features)
        return self.decoder(z)


def create_VAE_model(input_dimension=20, intermediate=25, bottleneck=15):
    vae_model = VAE(input_dimension=input_dimension,
                    intermediate=intermediate,
                    bottleneck=bottleneck,
                    name='VariationalAutoEncoder')
    vae_model.compile(optimizer='adam')
    return vae_model

# create_VAE_model().summary()
# create_AE_model().summary()
# create_MLP2_model().summary()
# create_MLP3_model().summary()
