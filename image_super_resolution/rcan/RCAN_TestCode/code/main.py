import torch
import torch_ipex

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

def dpcpp_fw_pre_hook(layer, input):
    if layer.__class__.__name__ == 'PixelShuffle':
        new_input = list()
        for i in input:
            new_input.append(i.to('cpu'))
        return tuple(new_input)

def dpcpp_fw_hook(layer, input, output):
    if layer.__class__.__name__ == 'PixelShuffle':
        return output.to('dpcpp')

def recursive(model, fw_hook, bw_hook=None, fw_pre_hook=None):
    has_children = False
    for layer in model.children():
        has_children = True
        recursive(layer, fw_hook, bw_hook, fw_pre_hook)
    if not has_children:
        model.register_forward_hook(fw_hook)
        if bw_hook:
            model.register_backward_hook(bw_hook)
        if fw_pre_hook:
            model.register_forward_pre_hook(fw_pre_hook)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    if args.sycl:
        recursive(model, dpcpp_fw_hook, fw_pre_hook=dpcpp_fw_pre_hook)
    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

