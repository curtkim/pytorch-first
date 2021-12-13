import torch

if __name__ == '__main__':
    checkpoint = torch.load("example.ckpt")
    print(checkpoint.keys())
    print(checkpoint['epoch'])          # 275
    print(checkpoint['global_step'])    # 514936
    # epoch, global_step, state_dict, callbacks, optimizer_states, lr_schedulers
