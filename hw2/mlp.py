import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
            self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        super().__init__()
        all_dims = [in_dim, *dims]
        layers = []
        layer = 0
        for in_dim, out_dim in zip(all_dims[:-1], all_dims[1:]):
            if not (type(nonlins[layer]) is str):
                linear = nn.Linear(in_dim, out_dim, bias=True)
                linear.weight.requires_grad = True
                linear.bias.requires_grad = True
                layers += [
                    linear,
                    nonlins[layer]
                ]
                print(linear)
            else:
                linear = nn.Linear(in_dim, out_dim, bias=True)
                linear.weight.requires_grad = True
                linear.bias.requires_grad = True
                layers += [
                    linear,
                    ACTIVATIONS[nonlins[layer]]()
                ]
            layer += 1

        # Sequential is a container for layers
        self.fc_layers = nn.Sequential(*layers)
        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======
        assert x.shape[1] == self.in_dim

        y_l = torch.stack([self.fc_layers(x_i) for x_i in torch.unbind(x, dim=0)], dim=0)
        assert y_l.shape[1] == self.out_dim
        return y_l
        # ========================
