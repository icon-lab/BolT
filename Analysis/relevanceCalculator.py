
# hacked from https://github.com/hila-chefer/Transformer-MM-Explainability

import torch
import numpy as np

# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition

def normalizeCam(cam, nW):
    cam = cam / torch.max(cam)#torch.max(cam[:nW, :nW])
    return cam

def normalizeR(R):
    R_ = R - torch.eye(R.shape[0])
    R_ /= R.sum(dim=1, keepdim=True)
    return R_ + torch.eye(R.shape[0])

def generate_relevance_(model, input, index=None):

    device = next(model.parameters()).device

    output, cls = model(input, analysis=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)
        

    # accumulate gradients on attentions
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot = torch.sum(one_hot.to(device) * output)
    model.zero_grad()
    one_hot.backward()

    # construct the initial relevance matrix

    shiftSize = model.shiftSize
    windowSize = model.hyperParams.windowSize
    T = input.shape[-1] # number of bold tokens

    dynamicLength = ((T - windowSize) // shiftSize) * shiftSize + windowSize

    nW = model.last_numberOfWindows # number of cls tokens
    
    num_tokens = dynamicLength + nW 
    R = torch.eye(num_tokens, num_tokens)

    # now pass the relevance matrix through the blocks



    for block in model.blocks:

        cam = block.getJuiceFlow().cpu()
        
        R += apply_self_attention_rules(R, cam)
        R = normalizeR(R)
        


    del one_hot
    del output
    torch.cuda.empty_cache()

    # R.shape = (dynamicLength + nW, dynamicLength + nW)

    # get the part that the window cls tokens are interested in
    # here we have relevance of each window cls token to the bold tokens
    inputToken_relevances = R[:nW, nW:] # of shape (nW, dynamicLength)


    return inputToken_relevances # (nW, dynamicLength)


def generate_relevance(model, input, index=None):

    device = next(model.parameters()).device

    output, cls = model(input, analysis=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    # accumulate gradients on attentions
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot = torch.from_numpy(one_hot)
    one_hot = torch.sum(one_hot.to(device) * output)
    model.zero_grad()
    one_hot.backward()

    # construct the initial relevance matrix

    shiftSize = model.shiftSize
    windowSize = model.hyperParams.windowSize
    T = input.shape[-1] # number of bold tokens

    dynamicLength = ((T - windowSize) // shiftSize) * shiftSize + windowSize

    nW = model.last_numberOfWindows # number of cls tokens
    
    num_tokens = dynamicLength + nW 
    R = torch.eye(num_tokens, num_tokens)

    # now pass the relevance matrix through the blocks



    for block in model.blocks:

        cam = block.getJuiceFlow().cpu()
        R += apply_self_attention_rules(R, cam)


    del one_hot
    del output
    torch.cuda.empty_cache()

    # R.shape = (dynamicLength + nW, dynamicLength + nW)

    # get the part that the window cls tokens are interested in
    # here we have relevance of each window cls token to the bold tokens
    inputToken_relevances = R[:nW, nW:] # of shape (nW, dynamicLength)


    return inputToken_relevances # (nW, dynamicLength)

