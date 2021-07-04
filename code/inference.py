
from __future__ import print_function

import os

import torch
from torch.autograd import Variable
import numpy as np
from torch import nn
import math
import torch
from torchvision import transforms
import torch
import torch.nn.functional as F
import time


torch.manual_seed(0)

import cv2 
import numpy as np
import torch.nn.functional as F 
import io
from PIL import Image 
#from model_def import Model
import json
import math
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'
# Network definition
#from model_def import Model


# title Model v3 (CNN + 1 SE Block +positional embedding) encoder + Decoder(2_lstms +attn) 
class Encoder_2(nn.Module):
    def __init__(self,image_embedding_size = 18) -> None:
        super(Encoder_2, self).__init__()
        self.image_embedding_size = image_embedding_size
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2 , bias = False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),


            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias = False),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(512),
            nn.SiLU(),


            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias = False),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            SEBlock(1024 , r = 2),
            nn.MaxPool2d(kernel_size=3, stride=2),


        )
        self.avgpool = nn.AdaptiveAvgPool2d((self.image_embedding_size,self.image_embedding_size))
        
      
    def forward(self, x: torch.Tensor) -> tuple :
        x = self.features(x)
        x = self.avgpool(x)
        x = x.permute(0,2,3,1) #results in shape [batch_size , image_embedding_size , image_embedding_size , 3]
        x = x.view(x.shape[0] , -1 , x.shape[-1]) # results in [batch_size , 14*14 , 2048]

        return x

class SEBlock(nn.Module):
  """Squeeze-and-excitation block"""
  def __init__(self, n_in, r=2):
    super().__init__()

    self.squeeze = nn.AdaptiveAvgPool2d(1)
    self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
                                    nn.SiLU(),
                                    nn.Conv2d(n_in//r, n_in, kernel_size=1),
                                    nn.Sigmoid())
  
  def forward(self, x):
    y = self.squeeze(x)
    y = self.excitation(y)
    return x * y


class AttentionNn(nn.Module):
    """ A Nn to learn the attention mapping """
    def __init__(self ,encoder_output_dim =1024, decoder_output_dim =512, attention_hidden_dim=512  ):
        """
        parameters:
            encoder_output_dim : dim of encoded image features embedding
            decoder_output_dim : dim of decoder output token embedding
            attention_hidden_dim : dim of attention neural network's hidden layer
        """
        super(AttentionNn , self).__init__()
        # transformation to same space
        self.transformed_enc = nn.Linear(encoder_output_dim , attention_hidden_dim)
        self.transformed_dec = nn.Linear(decoder_output_dim , attention_hidden_dim)

        self.attention_map = nn.Linear(attention_hidden_dim , 1 ) 
        self.tanh = nn.Tanh()

        self.soft = nn.Softmax(dim = 1)                                 

    def forward(self , image_embedding , latex_token_embedding):
            """
            Performs the attention mapping
            Parameters :
                image_embedding : output of  Encoder of shape [batch_size  ,num_pix ,encoder_output_dim]
                latex_token_embedding : previous token output from decoder [batch_size , decoder_output_dim]
            """
            enc_transformed = self.transformed_enc(image_embedding) #results in [batch_size , num_pixels , attention_hidden_dim]

            dec_transformed = self.transformed_dec(latex_token_embedding) #results in [batch_size  , attention_hidden_dim]

            summed = enc_transformed + dec_transformed.unsqueeze(1)#[batch_size , num_pixels , attention_hidden_dim]

            attn = self.attention_map(self.tanh(summed)).squeeze(2) #results in #results in [batch_size ,num_pixels ,1]
            alpha_distribution = self.soft(attn)
            # applying attention on  pixels and summing to get the  weighted-pixel embedding
            attended_feature_embedding = torch.sum(image_embedding * alpha_distribution.unsqueeze(2) , dim = 1) # results in [batch_size , encoder_output_dim]
            return  attended_feature_embedding , alpha_distribution  
class Decoder(nn.Module):
    """
    Decoder that decodes attended feature embeddings to output a latex string of Mathematical expression in image
    """
    def __init__(self  ,
                 vocab_size  ,decoder_hidden_dim = 512, encoder_output_dim =1024,token_embedding_size =128 ,dropout = 0.2  ) : 
        """
        Parameters : 
            token_embedding_size : dim for token embeddings
            decoder_hidden_dim : dim to use in LSTM hidden layers
            vocab_size : size of vocabulary 
            encoder_output_dim : dimension of encoder's output vector
            dropout : drop probability for Dropout layers
        """
        super(Decoder , self).__init__()
      


        self.token_embedding_size = token_embedding_size
        self.decoder_hidden_dim = decoder_hidden_dim
        self.vocab_size = vocab_size
        self.encoder_output_dim = encoder_output_dim
        self.dropout = dropout

        self.token_embeddings = nn.Embedding(self.vocab_size , self.token_embedding_size)
        self.attention = AttentionNn()
        self.dropout = nn.Dropout(p = self.dropout)

        self.lstm1 = nn.LSTMCell(self.encoder_output_dim + self.token_embedding_size, self.decoder_hidden_dim ,bias = True)
        self.lstm2 = nn.LSTMCell(self.decoder_hidden_dim , self.decoder_hidden_dim)

        self.lstm_h0 = nn.Linear(self.encoder_output_dim , self.decoder_hidden_dim)
        self.lstm_c0 = nn.Linear(self.encoder_output_dim , self.decoder_hidden_dim)

         
        self.gate_transform = nn.Linear(decoder_hidden_dim , encoder_output_dim )
        self.gate = nn.Sigmoid()

        # final layer to find score over vocabulary
        self.final = nn.Linear(decoder_hidden_dim , vocab_size)

    def initialize_hidden_states(self , encoder_output):
            """
            this method initializes hidden states for lstm cell based on encoded images 
            """
            mean_encoding = encoder_output.mean(dim = 1)
            h0 = self.lstm_h0(mean_encoding)
            c0 = self.lstm_c0(mean_encoding)

            return h0 , c0 
    def forward(self, features, latexes ):
        
        #vectorize the embeds
        embeds = self.token_embeddings(latexes)
       
        # Initialize LSTM state
        h1, c1 = self.initialize_hidden_states(features)  # (batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(latexes[0])-1 #Exclude the last one
        batch_size = latexes.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length,num_features).to(device)
        h2 , c2 =  h1.clone() , c1.clone() 
        for s in range(seq_length):
            attention_weighted_encoding ,alpha = self.attention(features, h2)
            lstm1_input = torch.cat((embeds[:, s], attention_weighted_encoding), dim=1)
            gate = self.gate(self.gate_transform(self.dropout(h2)) )
            attention_weighted_encoding = gate * attention_weighted_encoding

            
            h1, c1 = self.lstm1(lstm1_input, (h1, c1))

            h2 , c2 = self.lstm2 (h1 , (h2, c2))
            output = self.final(self.dropout(h2))
            
            preds[:,s] = output
            alphas[:,s] = alpha  
        
        
        return preds, alphas 

  

class Model(nn.Module):
    def __init__(self, vocab_size=124 ,device = "cpu" ,decoder_hidden_dim=512 ,token_embedding_size = 256):
        super().__init__()
        self.device = device
        self.encoder = Encoder_2()
        self.decoder = Decoder(
             vocab_size  ,
             decoder_hidden_dim = decoder_hidden_dim,
              encoder_output_dim =1024,
              token_embedding_size =token_embedding_size 
        )
    
    def forward(self, images, latexes ):
            features = self.encoder(images)
            outputs = self.decoder(self.add_positional_features(features), latexes)
            return outputs


    def add_positional_features(self , tensor: torch.Tensor,
                                min_timescale: float = 1.0,
                                max_timescale: float = 1.0e4):
        """
        Implements the frequency-based positional encoding
        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The largest timescale to use.
        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """
        _, timesteps, hidden_dim = tensor.size()

        timestep_range = self.get_range_vector(timesteps, tensor.device).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(
            num_timescales, tensor.device).data.float()

        log_timescale_increments = math.log(
            float(max_timescale) / float(min_timescale)) / float(num_timescales - 1)
        inverse_timescales = min_timescale * \
            torch.exp(timescale_range * -log_timescale_increments)

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.randn(
            scaled_time.size(0), 2*scaled_time.size(1), device=tensor.device)
        sinusoids[:, ::2] = torch.sin(scaled_time)
        sinusoids[:, 1::2] = torch.sin(scaled_time)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat(
                [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1)
        return tensor + sinusoids.unsqueeze(0)


    def get_range_vector(self , size: int, device) -> torch.Tensor:
        return torch.arange(0, size, dtype=torch.long, device=device)    
    
    
    def generate_latex(self ,image,max_len=64 ):
        
      device = self.device
      # Inference part
      # Given the image features generate the latex
      with torch.no_grad():
        self.encoder.eval()
        self.decoder.eval()
        features = self.add_positional_features(self.encoder(image))
        batch_size = features.size(0)
        h1, c1 = self.decoder.initialize_hidden_states(features)  # (batch_size, decoder_dim)
        
        alphas = []
        
        #starting input
        word = torch.tensor([116]).view(1,-1).to(device)
        embeds = self.decoder.token_embeddings(word).squeeze(1) #batch_size x emb_dim
        Latexes = [116]
        h2 , c2 = h1.clone(),c1.clone()
        for i in range(max_len):
            #awe is attended feature embedding
            awe, alpha = self.decoder.attention(features, h2)  # (batch_size, encoder_output_dim), (batch_size, num_pixels)
            #store the alpha score
            alphas.append(alpha.cpu().detach().numpy())
            
            gate = self.decoder.gate(self.decoder.gate_transform(h2))  # gating scalar, (batch_size, encoder_output_dim)
            awe = gate * awe
            h1, c1 = self.decoder.lstm1(torch.cat([embeds, awe], dim=1), (h1, c1))  # (batch_size, decoder_dim)
            
            h2, c2 = self.decoder.lstm2(h1, (h2, c2)) 
          
            output = self.decoder.final(self.decoder.dropout(h2))
            #select the token with most val
            predicted_word_idx = output.argmax(dim=1)

            #save the generated token
            Latexes.append(predicted_word_idx.item())
            
            #end if <EOS detected>
            if predicted_word_idx.item() ==  118 :#"</s>":
                break
            
            #send generated token as the next token
            embeds = self.decoder.token_embeddings(predicted_word_idx)
            

            

        #covert the vocab idx to words and return sentence
        return torch.LongTensor(Latexes  )













def model_fn(model_dir):
    print("In model_fn. Model directory is -")
    print(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    with open(os.path.join(model_dir, "model.pt"), "rb") as f:
        print("Loading the mnist model")
        checkpoint = torch.load(f , map_location =device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print('MODEL-LOADED')
        logger.info('model loaded successfully')
    return model



def preprocess_one_image(image):
    """
    Process one single image.
    """
    # get image from the request

    image = np.array(image)
    # Load image, grayscale, Otsu's threshold
    image = torch.from_numpy(image.transpose(2,0,1))
    image = F.interpolate(image.unsqueeze(0) , (550 , 1500) , mode = "nearest").squeeze()
    image = image.numpy().transpose(1,2,0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)

    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,51,9)       

    # Morph open to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours and remove small noise
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        
        area = cv2.contourArea(c)
        if area < 50:
            cv2.drawContours(opening, [c], -1, 0, -1)

    # Invert and apply slight Gaussian blur
    result = 255 - opening
    result = cv2.GaussianBlur(result, (3,3), 0)
    result = np.repeat(result[:,:,np.newaxis] ,3 , axis = -1)
    result = torch.from_numpy(result.transpose(2,0,1)).float()/255
    result = F.pad(result , pad =  (math.ceil((1600-result.shape[2])//2),
                                  math.ceil((1600-result.shape[2])//2),
                                  math.ceil((600-result.shape[1])//2),
                                  math.ceil((600-result.shape[1])//2))  ,value = 1)

    result = F.pad(result , pad = (0,1600-result.shape[2],0,600-result.shape[1])  ,value = 1)
    return result.unsqueeze(0)

def load_reverse_vocab():
    
    with open("./code/reverse_vocab.json")  as f :
        vocab = json.load(f)
    return vocab


def post_process_tok_to_str(Latexes ):
  
    r_vocab=load_reverse_vocab()
    decoded = [r_vocab[str(tok)] for tok in Latexes.tolist() if tok not in {116,117,118}]   
    decoded = "".join(decoded)
    latex = r'{}'.format(decoded)

    return latex
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    # process an image uploaded to the endpoint
    #if content_type == JPEG_CONTENT_TYPE: return io.BytesIO(request_body)
    logger.debug(f'Request body CONTENT-TYPE is: {content_type}')
    logger.debug(f'Request body TYPE is: {type(request_body)}')
    #if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body)).read()
    if content_type == JPEG_CONTENT_TYPE: return Image.open(io.BytesIO(request_body))
    

def predict_fn(input_object, model):
    logger.info("Calling model")
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        input_object=preprocess_one_image(input_object)
        latexes=model.generate_latex(input_object)
        latexes=post_process_tok_to_str(latexes)
        return latexes
    
# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JSON_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    if accept == JSON_CONTENT_TYPE: 
        logger.debug(f'Returning response {json.dumps(prediction)}')
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))