
import torch
from torch import nn

from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from Models.BolT.util import windowBoldSignal


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 1,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class WindowAttention(nn.Module):

    def __init__(self, dim, windowSize, receptiveSize, numHeads, headDim=20, attentionBias=True, qkvBias=True, attnDrop=0., projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias
        
        maxDisparity = windowSize - 1 + (receptiveSize - windowSize)//2


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*maxDisparity+1, numHeads))  # maxDisparity, nH

        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # get pair-wise relative position index for each token inside the window
        coords_x = torch.arange(self.windowSize) # N
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize)//2 # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # N, M
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)


        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.cls_bias_sequence_up, std=.02)
        trunc_normal_(self.cls_bias_sequence_down, std=.02)
        trunc_normal_(self.cls_bias_self, std=.02)
        
        self.softmax = nn.Softmax(dim=-1)


        # for token painting
        self.attentionMaps = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.nW = None

    def save_attention_maps(self, attentionMaps):
        self.attentionMaps = attentionMaps

    def save_attention_gradients(self, grads):
        self.attentionGradients = grads

    def averageJuiceAcrossHeads(self, cam, grad):

        """
            Hacked from the original paper git repo ref: https://github.com/hila-chefer/Transformer-MM-Explainability
            cam : (numberOfHeads, n, m)
            grad : (numberOfHeads, n, m)
        """

        #cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        #grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam


    def getJuiceFlow(self, shiftSize): # NOTE THAT, this functions assumes there is only one subject to analyze. So if you want to keep using this implementation, generate relevancy maps one by one for each subject

        # infer the dynamic length
        dynamicLength = self.windowSize + (self.nW - 1) * shiftSize

        targetAttentionMaps = self.attentionMaps # (nW, h, n, m) 
        targetAttentionGradients = self.attentionGradients #self.attentionGradients # (nW h n m)        

        globalJuiceMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)
        normalizerMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)


        # aggregate(by averaging) the juice from all the windows
        for i in range(self.nW):

            # average the juices across heads
            window_averageJuice = self.averageJuiceAcrossHeads(targetAttentionMaps[i], targetAttentionGradients[i]) # of shape (1+windowSize, 1+receptiveSize)
            
            # now broadcast the juice to the global juice matrix.

            # set boundaries for overflowing focal attentions
            L = (self.receptiveSize-self.windowSize)//2

            overflow_left = abs(min(i*shiftSize - L, 0))
            overflow_right = max(i*shiftSize + self.windowSize + L - dynamicLength, 0)

            leftMarker_global = i*shiftSize - L + overflow_left
            rightMarker_global = i*shiftSize + self.windowSize + L - overflow_right
            
            leftMarker_window = overflow_left
            rightMarker_window = self.receptiveSize - overflow_right

                # first the cls it self
            globalJuiceMatrix[i, i] += window_averageJuice[0,0]
            normalizerMatrix[i, i] += 1
                # cls to bold tokens
            globalJuiceMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window])
                # bold tokens to cls
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += window_averageJuice[1:, 0]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += torch.ones_like(window_averageJuice[1:, 0])
                # bold tokens to bold tokens
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window])

        # to prevent divide by zero for those non-existent attention connections
        normalizerMatrix[normalizerMatrix == 0] = 1

        globalJuiceMatrix = globalJuiceMatrix / normalizerMatrix

        return globalJuiceMatrix

    def forward(self, x, x_, mask, nW, analysis=False):
        """
            Input:

            x: base BOLD tokens with shape of (B*num_windows, 1+windowSize, C), the first one is cls token
            x_: receptive BOLD tokens with shape of (B*num_windows, 1+receptiveSize, C), again the first one is cls token
            mask: (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize)
            nW: number of windows
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 

            Output:

            transX : attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the first one is cls token

        """


        B_, N, C = x.shape
        _, M, _ = x_.shape
        N = N-1
        M = M-1

        B = B_ // nW 

        mask_left, mask_right = mask

        # linear mapping
        q = self.q(x) # (batchSize * #windows, 1+N, C)
        k, v = self.kv(x_).chunk(2, dim=-1) # (batchSize * #windows, 1+M, C)

        # head seperation
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        attn = torch.matmul(q , k.transpose(-1, -2)) * self.scale # (batchSize*#windows, h, n, m)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, M
       

        if(self.attentionBias):
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down
        
        # mask the not matching queries and tokens here
        maskCount = mask_left.shape[0]
        # repate masks for batch and heads
        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)

        mask_value = max_neg_value(attn) 


        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW = nW)        
        
        # make sure masks do not overflow
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        attn[:, :maskCount].masked_fill_(mask_left==1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right==1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")


        attn = self.softmax(attn) # (b, h, n, m)

        if(analysis):
            self.save_attention_maps(attn.detach()) # save attention
            handle = attn.register_hook(self.save_attention_gradients) # save it's gradient
            self.nW = nW
            self.handle = handle

        attn = self.attnDrop(attn)

        x = torch.matmul(attn, v) # of shape (b_, h, n, d)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)
        x = self.projDrop(x)
        
        return x



class FusedWindowTransformer(nn.Module):

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads, headDim, mlpRatio, attentionBias, drop, attnDrop):
        
        super().__init__()


        self.attention = WindowAttention(dim=dim, windowSize=windowSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, attentionBias=attentionBias, attnDrop=attnDrop, projDrop=drop)
        
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

    def getJuiceFlow(self):  
        return self.attention.getJuiceFlow(self.shiftSize)

    def forward(self, x, cls, windowX, windowX_, mask, nW, analysis=False):
        """

            Input: 

            x : (B, T, C)
            cls : (B, nW, C)
            windowX: (B, 1+windowSize, C)
            windowX_ (B, 1+windowReceptiveSize, C)
            mask : (B, 1+windowSize, 1+windowReceptiveSize)
            nW : number of windows

            analysis : Boolean, it is set True only when you want to analyze the model, otherwise not important 

            Output:

            xTrans : (B, T, C)
            clsTrans : (B, nW, C)

        """

        # WINDOW ATTENTION
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW, analysis=analysis) # (B*nW, 1+windowSize, C)
        clsTrans = windowXTrans[:,:1] # (B*nW, 1, C)
        xTrans = windowXTrans[:,1:] # (B*nW, windowSize, C)
        
        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)
        # FUSION
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        
        # residual connections
        clsTrans = clsTrans + cls
        xTrans = xTrans + x

        # MLP layers
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))

        return xTrans, clsTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        
        """
        Input:
            windowedX : (batchSize, nW, windowLength, C)
            scatterWeights : (windowLength, )
        
        Output:
            destination: (batchSize, dynamicLength, C)
        
        """

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        
        device = windowedX.device


        destination = torch.zeros((batchSize, dynamicLength,  C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)

        indexes = torch.tensor([[j+(i*shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C)) # (batchSize, nW, windowSize, featureDim)

        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")

        destination.scatter_add_(dim=1, index=indexes, src=src)


        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C) # (batchSize, nW, windowLength, featureDim)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")

        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)

        destination = destination / scalerDestination


        return destination

    

class BolTransformerBlock(nn.Module):

    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):

        assert((receptiveSize-windowSize)%2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim, windowSize=windowSize, shiftSize=shiftSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, mlpRatio=mlpRatio, attentionBias=attentionBias, drop=drop, attnDrop=attnDrop)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)
        mask_right = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)

        for i in range(maskCount):
            if(self.remainder > 0):
                mask_left[i, :, 1:1+self.remainder-shiftSize*i] = 1
                if(-self.remainder+shiftSize*i > 0):
                    mask_right[maskCount-1-i, :, -self.remainder+shiftSize*i:] = 1

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)


    def getJuiceFlow(self):
        return self.transformer.getJuiceFlow()
    
    def forward(self, x, cls, analysis=False):
        """
        Input:
            x : (batchSize, dynamicLength, c)
            cls : (batchSize, nW, c)
        
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 


        Output:
            fusedX_trans : (batchSize, dynamicLength, c)
            cls_trans : (batchSize, nW, c)

        """

        B, Z, C = x.shape
        device = x.device

        #update z, incase some are dropped during windowing
        Z = self.windowSize + self.shiftSize * (cls.shape[1]-1)
        x = x[:, :Z]

        # form the padded x to be used for focal keys and values
        x_ = torch.cat([torch.zeros((B, self.remainder,C),device=device), x, torch.zeros((B, self.remainder,C), device=device)], dim=1) # (B, remainder+Z+remainder, C) 

        # window the sequences
        windowedX, _ = windowBoldSignal(x.transpose(2,1), self.windowSize, self.shiftSize) # (B, nW, C, windowSize)         
        windowedX = windowedX.transpose(2,3) # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(x_.transpose(2,1), self.receptiveSize, self.shiftSize) # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2,3) # (B, nW, receptiveSize, C)

        
        nW = windowedX.shape[1] # number of windows
    
        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX], dim = 2) # (B, nW, 1+windowSize, C)
        xcls = rearrange(xcls, "b nw l c -> (b nw) l c") # (B*nW, 1+windowSize, C) 
       
        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_], dim=2) # (B, nw, 1+receptiveSize, C)
        xcls_ = rearrange(xcls_, "b nw l c -> (b nw) l c") # (B*nW, 1+receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        # pass to fused window transformer
        fusedX_trans, cls_trans = self.transformer(x, cls, xcls, xcls_, masks, nW, analysis) # (B*nW, 1+windowSize, C)


        return fusedX_trans, cls_trans





