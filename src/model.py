import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


def fc_net(input_dim, hidden_dim, output_dim):
    layers = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, output_dim),
    ]
    net = nn.Sequential(*layers)
    return net


class DeepHit(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        input_dim=12,
        hidden_dim_body=64,
        hidden_dim_head=32,
        output_dim=10,
        discrete_time=10,
        nr_event=2,
        residual=False,
    ):
        super().__init__()
        self.output_time = discrete_time
        self.nr_event = nr_event
        self.body = fc_net(input_dim, hidden_dim_body, input_dim)
        self.head1 = fc_net(input_dim, hidden_dim_head, output_dim)
        self.head2 = fc_net(input_dim, hidden_dim_head, output_dim)
        self.residual = residual

    def forward(self, x):
        input = self.body(x)

        if self.residual:
            # print("Shape input:",input.shape)
            output1 = self.head1(input + x)  # Residual connection from body to head1
            # print("Shape output1:",output1.shape)
            output2 = self.head2(input + x)  # Residual connection from body to head2
            # print("Shape output2:",output2.shape)

        else:
            output1 = self.head1(input)
            output2 = self.head2(input)

        stacked = torch.stack((output1, output2), dim=1)  # Stack the outputs
        # print("Shape stacked1:",stacked.shape)

        stacked = stacked.view(stacked.size(0), -1)  # Reshape to flatten the stacked tensor
        # print("Shape stacked2:",stacked.shape)

        output = F.softmax(stacked, dim=1)
        # print("Shape softmax:",output.shape)

        # print("Shape final:",output.view(stacked.shape[0], self.nr_event, -1).shape)

        return output.view(stacked.shape[0], self.nr_event, -1)


if __name__ == "__main__":
    pass
