import torch
import numpy as np
import math
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

BATCH_SIZE = 64
EMB_FEATURE = 512
NUM_CLASSES = 8132

PATCH_SIZE = 10
IMG_WIDTH = 80
IGM_HEIGHT = 110
HEAD_NUM = 8
BLOCKS = 8
HIDDEN_RATIO = 1
MLP_SIZE = 4
PRUNED_RATIO = 0.

#positional encoding selection

# naive absolute positional encoding
ABS_EMB = False #origin ViT
# naive relative positional encoding
REL_EMB = False

#image Relative Positional Encoding(iRPE)-bias mode
REL_BIAS = False #relative bias
ABS_BIAS = False #absolute bias

#image Relative Positional Encoding(iRPE)-contextual mode
REL_Q = True #q interaction
REL_QK = False #qk interaction
REL_QKV = False #qkv interaction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Init_embedding(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE, \
                       img_width=IMG_WIDTH, img_height=IGM_HEIGHT):
        super(Init_embedding, self).__init__()
        self.N = int(img_width*img_height*HIDDEN_RATIO/patch_size**2)
        self.N_3 = self.N*3
        self.batch_size = batch_size
        #for projection
        self.proj_conv1 = nn.Conv2d(3, self.N, patch_size, stride=patch_size, padding = 0)
        self.class_token = nn.Parameter(torch.FloatTensor(1, 1, self.N))
        self.position = nn.Parameter(torch.FloatTensor(self.N + 1, self.N))
        #initialization
        torch.nn.init.xavier_uniform_(self.class_token, gain=1.0)
        torch.nn.init.xavier_uniform_(self.position, gain=1.0)

    def rel_pos_cal(self, len_q, len_k):
        dist_mat = np.zeros((len_q, len_k))
        for q in range(len_q):
            for k in range(len_k):
                dist_mat[q][k] = q-k
                try:
                    dist_mat[k][q] = k-q
                except:
                    pass
        rel_pos_embedding = self.position[torch.tensor(dist_mat).long(),torch.tensor(dist_mat.transpose()).long()]

        return rel_pos_embedding

    #CNN feature + class token + position embedding
    def forward(self, input):
        input = torch.tensor(input)
        input = (input.float()/255).clone().detach()
        x = self.proj_conv1(input) #[batch size, 88, 8, 11]
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        if HIDDEN_RATIO != 1:
            x = x.repeat(1,HIDDEN_RATIO,1)

        #absolute or relative pos embedding
        if ABS_EMB == True:
            self.class_tokens = repeat(self.class_token, '() n e -> b n e', b=self.batch_size)
            x = torch.cat([self.class_tokens, x], dim=1)
            x = x + self.position
        else:
            if REL_EMB == True:
                rel_pos_emb = self.rel_pos_cal(self.N, self.N)
                x = x + rel_pos_emb
            self.class_tokens = repeat(self.class_token, '() n e -> b n e', b=self.batch_size)
            x = torch.cat([self.class_tokens, x], dim=1) #[batch size, 89, 88]

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads=HEAD_NUM, patch_size=PATCH_SIZE, \
        batch_size=BATCH_SIZE, img_width=IMG_WIDTH, img_height=IGM_HEIGHT):
        super(MultiHeadAttention, self).__init__()
        self.batch_size = batch_size
        self.N = int(img_width*img_height*HIDDEN_RATIO/patch_size**2)
        self.N_3 = self.N*3
        self.heads = heads
        #for multi head
        self.q_transform = nn.Linear(self.N, self.N)
        self.k_transform = nn.Linear(self.N, self.N)
        self.v_transform = nn.Linear(self.N, self.N)
        self.proj_transform = nn.Linear(self.N, self.N)
        #for relative positional bias
        self.bias = 0.
        self.pos_embedding = nn.Parameter(torch.FloatTensor(self.N+1, self.N+1))
        torch.nn.init.xavier_uniform_(self.pos_embedding, gain=1.0)

    def rel_pos_cal(self, len_q, len_k):
        dist_mat = np.zeros((len_q, len_k))
        for q in range(len_q):
            for k in range(len_k):
                dist_mat[q][k] = q-k
                dist_mat[k][q] = k-q
        rel_pos_embedding = self.pos_embedding[torch.tensor(dist_mat).long(),torch.tensor(dist_mat.transpose()).long()]

        return rel_pos_embedding
        
    def forward(self, embedding):
        #multihead self attention
        query = self.q_transform(embedding) #[batch_size, 89, 88]
        key = self.k_transform(embedding) #[batch_size, 89, 88]
        value = self.v_transform (embedding) #[batch_size, 89, 88]

        #rel pos embedding - contextual mode
        if REL_Q == True:
            rel_pos_embedding = self.rel_pos_cal(self.N+1, self.N+1) #[89, 89]
            rel_pos_embedding = rel_pos_embedding[:,1:] #[89, 88]
            bias = query.matmul(torch.transpose(rel_pos_embedding,0,1)) #[batch_size, 89, 88]*[88, 89]
            self.bias  = torch.unsqueeze(bias, 1)
        elif REL_QK == True:
            rel_pos_embedding = self.rel_pos_cal(self.N+1, self.N+1) #[89, 89]
            rel_pos_embedding = rel_pos_embedding[:,1:] #[89, 88]
            bias_q = query.matmul(torch.transpose(rel_pos_embedding,0,1)) #[batch_size, 89, 88]*[88, 89]
            bias_k = key.matmul(torch.transpose(rel_pos_embedding,0,1)) #[batch_size, 89, 88]*[88, 89]
            bias = bias_q + bias_k
            self.bias  = torch.unsqueeze(bias, 1)
        elif REL_QKV == True:
            rel_pos_embedding = self.rel_pos_cal(self.N+1, self.N+1) #[89, 89]
            rel_pos_embedding = rel_pos_embedding[:,1:] #[89, 88]
            bias_q = query.matmul(torch.transpose(rel_pos_embedding,0,1)) #[batch_size, 89, 88]*[88, 89]
            bias_k = key.matmul(torch.transpose(rel_pos_embedding,0,1)) #[batch_size, 89, 88]*[88, 89]
            value = value + rel_pos_embedding #[batch_size, 89, 88]*[88, 89]
            bias = bias_q + bias_k
            self.bias = torch.unsqueeze(bias, 1)

        query = rearrange(self.q_transform(embedding), "b n (h d) -> b h n d", h=self.heads) #[batch_size, 8(heads), 89, 11]
        key = rearrange(self.k_transform(embedding), "b n (h d) -> b h n d", h=self.heads) #[batch_size, 8(heads), 89, 11]
        value  = rearrange(self.v_transform (embedding), "b n (h d) -> b h n d", h=self.heads) #[batch_size, 8(heads), 89, 11]
        attend = query.matmul(torch.transpose(key, 2,3)) #qeury*key^T

        if REL_BIAS == True:
            rel_pos_embedding = self.rel_pos_cal(self.N+1, self.N+1)
            self.bias = rel_pos_embedding 
        else:
            if ABS_BIAS == True:
                self.bias = self.pos_embedding
                
        attend = attend + self.bias

        attend_scale = attend/torch.sqrt(torch.tensor(self.N).float()) #scaling
        attend_scale = torch.softmax(attend_scale, dim=-1) #softmax
        output = attend_scale.matmul(value)
        output_reshape = torch.reshape(output,(self.batch_size, self.N+1, self.N))
        output_mhead = self.proj_transform(output_reshape)

        return output_mhead, query, key, value

class Sparse_ViT(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE, patch_size=PATCH_SIZE,\
                       img_width=IMG_WIDTH, img_height=IGM_HEIGHT,\
                       heads=HEAD_NUM, blocks=BLOCKS,\
                       in_features=EMB_FEATURE, num_classes=NUM_CLASSES):
        super(Sparse_ViT, self).__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.img_width = img_width
        self.img_height = img_height
        self.heads = heads
        self.blocks = blocks
        self.in_features = in_features
        self.out_feautures = self.num_classes = num_classes
        self.gelu = nn.GELU()

        self.input_embedding = Init_embedding(self.batch_size, self.patch_size, self.img_width, self.img_height)
        self.input_n = self.input_embedding.N
        self.output_n = self.input_embedding.N*MLP_SIZE
        self.arcface = ArcFace(64, 0.5, self.in_features, self.out_feautures)
        self.layer_norm = nn.LayerNorm(self.input_n, eps=1e-8)
        self.out_size = int(self.output_n*(1-PRUNED_RATIO))

        #for encoder
        for i in range(self.blocks):
            exec('self.multihead' + str(i+1) + ' = MultiHeadAttention(self.heads, \
                    self.patch_size, self.batch_size,self.img_width, self.img_height)')
            exec('self.MLP' + str(i+1) + '_exp = nn.Linear(self.input_n, self.out_size)')
            exec('self.MLP' + str(i+1) + '_slk = nn.Linear(self.out_size, self.input_n)')

        #for decoder
        self.outblock = nn.Linear(self.input_n*(self.input_n+1), int((1-PRUNED_RATIO)*self.in_features))
        self.loss_feature = nn.Linear(int((1-PRUNED_RATIO)*self.in_features), self.num_classes)
        
    def encoder_init(self, multihead, MLP_exp, MLP_slk, input):
        #multi head attention
        multi_embedding,_ ,_ ,_ = multihead(input)
        #add & layer norm1
        norm_embedding = self.layer_norm (input + multi_embedding)
        #forward transform
        output_exp = MLP_exp(norm_embedding)
        output_exp = self.gelu(output_exp)
        output_slk = MLP_slk(output_exp)
        return output_slk

    def decoder(self, embedding, out_trans):
        out_embedding = torch.flatten(embedding, start_dim=1)
        out_final = out_trans(out_embedding)
        return out_final

    #arcface : angular margin add
    def arcface_forward(self, image, label):
        input_embedding = self.input_embedding(image)
        #block1
        block1_out1 = self.encoder_init(self.multihead1, self.MLP1_exp, self.MLP1_slk, input_embedding)
        block1_out2 = self.encoder_init(self.multihead2, self.MLP2_exp, self.MLP2_slk, block1_out1)
        block1_out3 = self.encoder_init(self.multihead3, self.MLP3_exp, self.MLP3_slk, block1_out2)
        block1_out4 = self.encoder_init(self.multihead4, self.MLP4_exp, self.MLP4_slk, block1_out3)
        block1_out5 = self.encoder_init(self.multihead5, self.MLP5_exp, self.MLP5_slk, block1_out4)
        block1_out6 = self.encoder_init(self.multihead6, self.MLP6_exp, self.MLP6_slk, block1_out5)
        block1_out7 = self.encoder_init(self.multihead7, self.MLP7_exp, self.MLP7_slk, block1_out6)
        block1_out8 = self.encoder_init(self.multihead8, self.MLP8_exp, self.MLP8_slk, block1_out7)

        #output of block1
        output_block1 = self.decoder(block1_out8, self.outblock)

        margin_feature, feature = self.arcface.forward(output_block1, label)

        return margin_feature, feature

    #feature extractor
    def forward(self, image):
        input_embedding = self.input_embedding(image)
        #block1
        block1_out1 = self.encoder_init(self.multihead1, self.MLP1_exp, self.MLP1_slk, input_embedding)
        block1_out2 = self.encoder_init(self.multihead2, self.MLP2_exp, self.MLP2_slk, block1_out1)
        block1_out3 = self.encoder_init(self.multihead3, self.MLP3_exp, self.MLP3_slk, block1_out2)
        block1_out4 = self.encoder_init(self.multihead4, self.MLP4_exp, self.MLP4_slk, block1_out3)
        block1_out5 = self.encoder_init(self.multihead5, self.MLP5_exp, self.MLP5_slk, block1_out4)
        block1_out6 = self.encoder_init(self.multihead6, self.MLP6_exp, self.MLP6_slk, block1_out5)
        block1_out7 = self.encoder_init(self.multihead7, self.MLP7_exp, self.MLP7_slk, block1_out6)
        block1_out8 = self.encoder_init(self.multihead8, self.MLP8_exp, self.MLP8_slk, block1_out7)

        output_block1 = self.decoder(block1_out8, self.outblock)
        feature = output_block1

        return feature

class ArcFace(nn.Module):
    def __init__(self, s, margin, in_features, out_features):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = margin
        self.in_features = in_features
        self.out_feautures = out_features
        self.weight = nn.Parameter(torch.FloatTensor(self.out_feautures, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
    
    def forward(self, input_feature, label):
        cosine = F.linear(F.normalize(input_feature), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp((1.0 - torch.pow(cosine, 2)),1e-9,1))
        phi = cosine * self.cos_m - sine * self.sin_m
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        one_hot_zero = torch.zeros(cosine.size(), device='cuda')
        output_no = (one_hot_zero * phi) + ((1.0 - one_hot_zero) * cosine)  
        output_no *= self.s

        return output, output_no

    def test(self, input_feature):
        cosine = F.linear(F.normalize(input_feature), F.normalize(self.weight))
        a_cosine = torch.acos(cosine)
        feature = self.s*torch.cos(a_cosine)

        return feature

