import torch
import os

def ckpt_to_pth(ckpt):
    states = torch.load(ckpt, map_location=lambda storage, loc: storage)[
            "state_dict"
        ]
    states = {k[6:]: v for k, v in states.items() if k.startswith("model.")}
    return states

def main():
    
    # Write the path to checkpoints here
    ckpt = '/data/jkuspalova/experiments/simple_short/last.ckpt'
    model_path = '/data/jkuspalova/experiments/simple_short/last.pth'
    torch.save(ckpt_to_pth(ckpt), model_path)
    
    

if __name__ == "__main__":
    main()