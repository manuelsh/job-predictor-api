from flask import Flask, request, Response

from model_class import *
from utils import *
import torch.nn as nn
import torch

# Main parameters
device = 'cpu'
model_file_name = 'model-parameters/model.tar'
hidden_size = 500
encoder_n_layers = 3
decoder_n_layers = 3
dropout = 0.0
attn_model = 'dot'
max_length = 4  # Maximum sentence length to consider
SOS_token = 1  # Start-of-sentence token

# Instantiate vocabulary
voc = Voc()

# Load model parameters and vocabulary
checkpoint = torch.load(model_file_name, map_location=torch.device(device))
encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']

# Initialize encoder & decoder models & searcher
embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
searcher = GreedySearchDecoder(encoder, decoder, device, SOS_token)

# Associate loaded parameters
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

# Instantiate flask
app = Flask(__name__)

@app.route("/predicted_job")
def predict():
    desc = request.args.get("previous_job")
    res = get_output(encoder, decoder, searcher, voc, max_length, device, input_sentence=desc)
    return Response(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("80"), debug=True)
