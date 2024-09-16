import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self,in_features,hidden_features,attention_layers,num_heads,linear_drop):
        super(Attention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.proj = nn.Linear(in_features, int(in_features*3))
        self.hard_parameter_sharing = Hardsharing(int(in_features*3), hidden_features, attention_layers, num_heads,linear_drop)
        self.L_layer = L_Layer(int(in_features*3), hidden_features)

    def forward(self, x):
        x = self.proj(x)
        features = self.hard_parameter_sharing(x)
        LE = self.L_layer(features)
        return LE

#================================ 辅助层 ==============================

class Hardsharing(nn.Module):
    def __init__(self,in_features,hidden_features,attention_layers,num_heads,linear_drop):
        super(Hardsharing, self).__init__()
        self.hidden_features = hidden_features
        self.num_heads = num_heads

        self.attentionblocks = nn.ModuleList(
            [Attentionlayer(in_features,hidden_features,num_heads,linear_drop)
             for i in range(attention_layers)])

    def forward(self,x):
        for block in self.attentionblocks:
            x = block(x)
        return x

class L_Layer(nn.Module):
    def __init__(self,in_features,hidden_features):
        super(L_Layer, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, int(hidden_features / 4)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 4), int(hidden_features / 8)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 8), int(hidden_features / 16)),
            nn.ReLU(),
            nn.Linear(int(hidden_features / 16), 1)
        )

    def forward(self, x):
        x = self.linear1(x)
        return x

class Attentionlayer(nn.Module):
    def __init__(self,in_features,hidden_features,num_heads,linear_drop):
        super(Attentionlayer,self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_heads = num_heads

        self.qkv = nn.Linear(in_features,in_features*3,bias=True)
        self.attn_drop = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)
        self.dk = (in_features // num_heads) ** -0.5
        self.proj = nn.Linear(in_features, in_features)
        self.proj_drop = nn.Dropout(linear_drop)
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, in_features)
        )
        self.drop = nn.Dropout(linear_drop)

    def forward(self,x):
        B,C = x.shape
        shortcut1 = x
        qkv = self.qkv(x)
        qkv = qkv.reshape(B,3,self.num_heads,C//self.num_heads)
        qkv = qkv.permute(1,0,2,3)
        q,k,v = qkv[0],qkv[1],qkv[2]

        attention = (q @ k.transpose(-2, -1))
        attention = attention * self.dk
        attention = self.softmax(attention)
        attention = self.attn_drop(attention)

        output = (attention @ v).transpose(1,2).reshape(B,C)
        output = self.proj(output)
        output = self.proj_drop(output)

        output = output + shortcut1

        shortcut2 = output
        output = self.linear1(output)
        output = self.drop(output)
        output = output + shortcut2

        return output