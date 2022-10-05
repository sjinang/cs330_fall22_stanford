"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    sparse: boolean, optional
        Use sparse gradients.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 sparse=False, embedding_sharing=True):

        super().__init__()

        self.embedding_dim = embedding_dim

        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************
        self.U_mf_embedding = ScaledEmbedding(num_users,embedding_dim)
        self.Q_mf_embedding = ScaledEmbedding(num_items,embedding_dim)
        self.A_mf_embedding = ZeroEmbedding(num_users,1)
        self.B_mf_embedding = ZeroEmbedding(num_items,1)

        self.U_mlp_embedding = self.U_mf_embedding
        self.Q_mlp_embedding = self.Q_mf_embedding
        # self.A_mlp_embedding = self.A_mf_embedding
        # self.B_mlp_embedding = self.B_mf_embedding

        if embedding_sharing is False:
            self.U_mlp_embedding = ScaledEmbedding(num_users,embedding_dim)
            self.Q_mlp_embedding = ScaledEmbedding(num_items,embedding_dim)
            # self.A_mlp_embedding = ZeroEmbedding(num_users,1)
            # self.B_mlp_embedding = ZeroEmbedding(num_items,1)



        self.MLP_layers=[]
        for l in range(len(layer_sizes)-1):
            self.MLP_layers.append(nn.Linear(layer_sizes[l],layer_sizes[l+1]))
            self.MLP_layers.append(nn.ReLU())
        self.MLP_layers.append(nn.Linear(layer_sizes[-1],1))
        self.MLP_layers = nn.ModuleList(self.MLP_layers)

        #********************************************************
        #********************************************************
        #********************************************************

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        #********************************************************
        #******************* YOUR CODE HERE *********************
        #********************************************************

        # Matrix Factorization Task
        # print(self.U_mf_embedding.weight==self.U_mlp_embedding.weight)

        u_mf = self.U_mf_embedding(user_ids.clone())
        q_mf = self.Q_mf_embedding(item_ids.clone())
        a_mf = self.A_mf_embedding(user_ids.clone())
        b_mf = self.B_mf_embedding(item_ids.clone())
        
        predictions = torch.sum(u_mf*q_mf,dim=-1).squeeze() + a_mf.squeeze() + b_mf.squeeze()
        
        # Regression Task
        
        u_mlp = self.U_mlp_embedding(user_ids.clone())
        q_mlp = self.Q_mlp_embedding(item_ids.clone())

        mlp_out = torch.cat((u_mlp,q_mlp,u_mlp*q_mlp),dim=-1)
        for layer in self.MLP_layers:
            mlp_out = layer(mlp_out)
        score = mlp_out.squeeze()
        



        #********************************************************
        #********************************************************
        #********************************************************
        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score