import torch
from torch import nn
from torch.nn import functional as F

class MLPModel(nn.Module):
    def __init__(self, config):
        super(MLPModel, self).__init__()

        # Get the configurations from the dictionary
        hidden_size = config.get('hidden_size', 768)
        num_classes = config.get('num_classes', 4)
        dropout_rate = config.get('dropout_rate', 0.3)
        mlp_hidden_dims = config.get('mlp_hidden_dims', [512, 256])
        activation_fn = config.get('activation_fn', 'relu')

        # Choose activation function
        if activation_fn == 'relu':
            activation = nn.ReLU()
        elif activation_fn == 'gelu':
            activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Define the MLP architecture based on the configuration
        layers = []
        input_size = hidden_size * 2  # For concatenated embeddings
        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_dim
        
        layers.append(nn.Linear(input_size, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, embed1, embed2):
        # Concatenate the two CLS embeddings
        if len(embed1.shape) == 3:
            combined_embedding = torch.cat((embed1[:,0,:], embed2[:,0,:]), dim=1)  # Shape: (batch_size, 768*2)
        else:
            combined_embedding = torch.cat((embed1, embed2), dim=1)

        # Pass the concatenated embeddings through the MLP
        logits = self.mlp(combined_embedding)
        return logits


class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()

        # Get the configurations from the dictionary
        hidden_size = config.get('hidden_size', 768)
        num_classes = config.get('num_classes', 4)
        dropout_rate = config.get('dropout_rate', 0.3)
        conv_window_size = config.get('conv_window_size', [3, 5, 7])
        out_conv_list_dim = config.get('out_conv_list_dim', 128)
        activation_fn = config.get('activation_fn', 'gelu')

        # Choose activation function
        if activation_fn == 'relu':
            self.activation = nn.ReLU()
        elif activation_fn == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        # Define the convolutional layers based on the configuration
        self.conv_list = nn.ModuleList([
            nn.Conv1d(hidden_size, out_conv_list_dim, w, padding=(w-1)//2)
            for w in conv_window_size
        ])
        
        self.mlp = nn.Linear(out_conv_list_dim * len(conv_window_size), num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embed1, embed2):
        # Concatenate the two CLS embeddings
        combined_embedding = torch.cat((embed1, embed2), dim=1)  # Shape: (batch_size, msl*2, 768)

        conv_outputs = []
        for conv_layer in self.conv_list:
            conv_output = self.activation(conv_layer(combined_embedding.transpose(1, 2)))
            conv_output, _ = torch.max(conv_output, dim=-1)  # Max pooling across sequence length
            conv_outputs.append(conv_output)

        # Concatenate the conv outputs and apply dropout
        conv_outputs = self.dropout(torch.cat(conv_outputs, dim=-1))

        # Pass through the MLP to get logits
        logits = self.mlp(conv_outputs)
        return logits
