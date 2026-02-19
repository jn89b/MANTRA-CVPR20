import torch
import torch.nn as nn
from typing import Tuple

class model_encdec(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    
    Updated to 3D trajectories (x,y,z).
    Input:
      past   (B, Tp, 3)
      future (B, Tf, 3)
    Output:
      future_pred (B, Tf, 3)

    Notes:
    - This is the original repo style (MLP + GRU) and kept structurally identical,
      only changing coord dims from 2 -> 3.
    """
    def __init__(self, settings):
        super(model_encdec, self).__init__()

        self.name_model = "autoencoder"
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]

        # === 3D ===
        channel_in = 3          # was 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # temporal encoding
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        # encoder-decoder
        self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        self.encoder_fut = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)

        # decoder consumes concatenated (past_state, fut_state) => 2K
        self.decoder = nn.GRU(self.dim_embedding_key * 2, self.dim_embedding_key * 2, 1, batch_first=False)

        # Output is 3D displacement
        self.FC_output = nn.Linear(self.dim_embedding_key * 2, 3)   # was 2

        # activation function
        self.relu = nn.ReLU()

        # weight initialization: kaiming (matches your style)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)

        # GRU params (kept consistent with your original style)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)

        nn.init.kaiming_normal_(self.FC_output.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.encoder_fut.bias_ih_l0)
        nn.init.zeros_(self.encoder_fut.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)

    def forward(self, past, future):
        """
        Forward pass that encodes past and future and decodes the future.

        :param past:   (B, past_len, 3)
        :param future: (B, future_len, 3)
        :return: prediction (B, future_len, 3)
        """
        dim_batch = past.size(0)
        device = past.device

        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 2, device=device)
        prediction = torch.empty(0, device=device)

        # present is last past point (B,1,3)
        present = past[:, -1, :3].unsqueeze(1)

        # temporal encoding for past: (B,T,3) -> (B,3,T) -> conv -> (B,16,T) -> (B,T,16)
        past_t = torch.transpose(past, 1, 2)
        past_embed = self.relu(self.conv_past(past_t))
        past_embed = torch.transpose(past_embed, 1, 2)

        # temporal encoding for future
        fut_t = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(fut_t))
        future_embed = torch.transpose(future_embed, 1, 2)

        # sequence encoding
        _, state_past = self.encoder_past(past_embed)     # (1,B,K)
        _, state_fut = self.encoder_fut(future_embed)     # (1,B,K)

        # state concatenation and decoding
        state_conc = torch.cat((state_past, state_fut), 2)  # (1,B,2K)
        input_fut = state_conc
        state_dec = zero_padding

        for _ in range(self.future_len):
            output_decoder, state_dec = self.decoder(input_fut, state_dec)  # output_decoder: (1,B,2K)
            displacement_next = self.FC_output(output_decoder)              # (1,B,3)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)  # (B,1,3)
            prediction = torch.cat((prediction, coords_next), 1)            # (B,t,3)
            present = coords_next
            input_fut = zero_padding

        return prediction