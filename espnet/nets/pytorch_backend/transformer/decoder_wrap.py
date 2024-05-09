import torch
from typing import Any, List, Tuple


class DecoderWrap(torch.nn.Module):
    def __init__(self, decoder, last_layer_dim):
        super(DecoderWrap, self).__init__()
        self.decoder = decoder
        self.last_layer = torch.nn.Linear(self.decoder.odim, last_layer_dim)
        
        
    def forward(self, tgt, tgt_mask, memory, memory_mask):
        #print(tgt.size())
        x = self.decoder.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoder.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.decoder.normalize_before:
            x = self.decoder.after_norm(x)
            
        if self.last_layer is not None:
            x = self.decoder.output_layer(x)
            x = self.last_layer(x)
            
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        x = self.decoder.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoder.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoder.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c
            )
            new_cache.append(x)

        if self.decoder.normalize_before:
            y = self.decoder.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if self.last_layer is not None:
            y = torch.log_softmax(self.last_layer(y), dim=-1)
            return y, new_cache

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        return self.decoder.score(ys, state, x)

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        return self.decoder.batch_score(ys, states, xs)