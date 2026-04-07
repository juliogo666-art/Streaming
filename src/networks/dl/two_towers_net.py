import torch
import torch.nn as nn
import torch.nn.functional as F

class UserTower(nn.Module):
    """
    Torre de Usuario: Codifica la identidad (ID) y atributos en un vector denso.
    """
    def __init__(self, num_users, embedding_dim=64, hidden_layers=[128, 64]):
        super(UserTower, self).__init__()
        self.user_embed = nn.Embedding(num_users, embedding_dim)
        
        layers = []
        input_dim = embedding_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = h
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_ids):
        # [batch_size, embedding_dim]
        x = self.user_embed(user_ids)
        # [batch_size, last_hidden_dim]
        return self.mlp(x)


class ItemTower(nn.Module):
    """
    Torre de Items (Películas): Codifica el contenido en el mismo espacio vectorial que el usuario.
    """
    def __init__(self, num_items, embedding_dim=64, hidden_layers=[128, 64]):
        super(ItemTower, self).__init__()
        self.item_embed = nn.Embedding(num_items, embedding_dim)
        
        layers = []
        input_dim = embedding_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = h
            
        self.mlp = nn.Sequential(*layers)

    def forward(self, item_ids):
        # [batch_size, embedding_dim]
        x = self.item_embed(item_ids)
        # [batch_size, last_hidden_dim]
        return self.mlp(x)


class TwoTowersModel(nn.Module):
    """
    Modelo Two-Towers (Bi-Encoder):
    Calcula la probabilidad de interacción mediante el producto escalar de los vectores
    de salida de las dos torres.
    """
    def __init__(self, num_users, num_items, embedding_dim=64, hidden_layers=[128, 64]):
        super(TwoTowersModel, self).__init__()
        self.user_tower = UserTower(num_users, embedding_dim, hidden_layers)
        self.item_tower = ItemTower(num_items, embedding_dim, hidden_layers)
        
    def forward(self, user_ids, item_ids):
        # 1. Obtener representaciones vectoriales (embeddings profundos)
        user_vector = self.user_tower(user_ids) # [B, D]
        item_vector = self.item_tower(item_ids) # [B, D]
        
        # 2. Producto escalar para medir similitud
        # [B, D] * [B, D] -> sum over D -> [B]
        dot_product = (user_vector * item_vector).sum(dim=1)
        
        return dot_product
