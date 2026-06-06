from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from . import config



class RailwayAutoencoder:
    def __init__(self):
        self.input_shape = config.IMAGE_SHAPE + (3,)
        self.latent_dim = 512

    def build_encoder(self):
        """
        Creates the encoder using a truncated ResNet50.
        """
        encoder_input = layers.Input(shape=self.input_shape)

        base_model = ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_tensor=encoder_input
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Extract features from a middle layer
        encoder_output = base_model.get_layer('conv4_block6_out').output
        
        # Convolution of 1x1 to create the latent bottleneck
        bottleneck = layers.Conv2D(self.latent_dim, (1, 1), padding='same', name='bottleneck')(encoder_output)
        bottleneck = layers.BatchNormalization()(bottleneck)
        bottleneck = layers.LeakyReLU(name='latent_representation')(bottleneck)
        
        return models.Model(base_model.input, bottleneck, name="Encoder")

    def build_decoder(self, encoder_output_shape):
        """
        Symmetric decoder to reconstruct the image.
        """
        decoder_input = layers.Input(shape=encoder_output_shape[1:])
        
        x = decoder_input
        
        # Upsampling Block 1: 16x16 -> 32x32
        x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        # Upsampling Block 2: 32x32 -> 64x64
        x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        # Upsampling Block 3: 64x64 -> 128x128
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        # Upsampling Block 4: 128x128 -> 256x256
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        
        # Final Reconstruction Head
        output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='reconstruction')(x)
        
        return models.Model(decoder_input, output, name="Decoder")

    def build(self):
        """
        Assembles the full Autoencoder.
        """
        # Build
        encoder = self.build_encoder()
        decoder = self.build_decoder(encoder.output_shape)
        
        # Connect
        autoencoder_input = layers.Input(shape=self.input_shape)
        latent = encoder(autoencoder_input)
        reconstruction = decoder(latent)
        
        return models.Model(autoencoder_input, reconstruction, name="Railway_Autoencoder")



def get_model():
    factory = RailwayAutoencoder()
    return factory.build()