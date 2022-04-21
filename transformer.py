from torch import nn
import torch.nn.functional as F
import torch

class VisualTransformer(nn.Module):
    """
    Class representing a visual transformer through the PyTorch DL framework
    """

    def __init__(self, class_num=10) -> None:
        """
        Initializes the transformer's various layers (encoder, decoder, positional, and dense)
        Inherits from nn.Module initializer function.
        :param class_num: number of classes that will be predicted by the model.
            Default 10 for the 10 classes represented by CIFAR10
        """
         
        super().__init__()
        
        self.embed_size = 32*32*3/(8*3)
        self.window_size = 8*3
        self.class_num = class_num
        
        ## --- ENCODER --- ##
        self.encoder_pos_embed = Positional_Encoding_Layer(self.window_size, self.embed_size)
        
        # Implements self-attention with 16 attention heads
        self.encoding_layer = torch.nn.TransformerEncoderLayer(self.embed_size, 16, \
                                                     activation='gelu', dropout = 0.0)
        self.encoder = torch.nn.TransformerEncoder(self.encoding_layer, 5)
        
        
        ## --- DECODER --- ##
        self.decoder_pos_embed = Positional_Encoding_Layer(self.window_size, self.embed_size)
        
        # Implements self-attention with 8 attention heads
        self.decoding_layer = torch.nn.TransformerDecoderLayer(self.embed_size, 8, \
                                                     activation='gelu', dropout = 0.0)
        self.decoder = torch.nn.TransformerDecoder(self.decoding_layer, 5)
        
        
        ## --- DENSE --- ##
        self.dense1 = nn.Linear(self.embed_size * self.window_size, 480)
        self.dense2 = nn.Linear(480, 128)
        self.dense3 = nn.Linear(128, 30)
        self.dense4 = nn.Linear(30, self.class_num)
        
    
    def forward(self, inputs):
        """
        Performs forward propagation through the defined layers for given input
        :param inputs: Input to the Visual Transformer. Dimension (batch size, 3, 32, 32)
        :return: (batch size, 10) torch tensor output of convolution and 
        subsequent linear layers. Size 10 given the 10 predicted classes.
        """
        
        num_batches = inputs.shape[0]
        
        # Convert from [batch_size, Channels, H, W] to [batch_size, window_size, embedding_size]
        enc_in = torch.reshape(inputs, (num_batches, self.window_size, self.embed_size))
        # Encoder takes in [batch_size, window_size, embedding_size]
        
        # Add positional embeddings for encoder
        positioned_enc = self.encoder_pos_embed(enc_in)
        
        # Pass through encoder
        enc_out = self.encoder(positioned_enc)
        
        # Add positional embeddings for decoder
        positioned_dec = self.decoder_pos_embed(enc_out)
        
        # Pass through decoder
        dec_out = self.decoder(positioned_dec)
        
        # Finally, pass through linear layers (RELU activation for each except last)
        flat = nn.Flatten()(dec_out)
        final_out = F.relu(self.dense1(flat))
        final_out = F.relu(self.dense2(final_out))
        final_out = F.relu(self.dense3(final_out))
        final_out = F.softmax(self.dense4(final_out)) # Softmax for last layer
        
        return final_out
    
    
class Positional_Encoding_Layer(nn.Module):
    """ Class for the trainable Positional Embedding Layer """
    
    def __init__(self, window_size, emb_size):
        """ Initializes trainable positional embeddings to add to the input """
        super(Positional_Encoding_Layer, self).__init__()
        
        self.pos_embed = nn.Parameter([window_size, emb_size])

    def forward(self, word_embeds):
        """ Adds (trainable) positional embeddings to the input """
        positioned = word_embeds + self.pos_embed
        return positioned
    