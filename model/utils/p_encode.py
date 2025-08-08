import torch

def positional_encoding(positions, freqs):
        freq_bands = (2 ** torch.arange(freqs).float()).to(positions.device)
        pts = (positions[..., None] * freq_bands).reshape(
                positions.shape[:-1] + (freqs * positions.shape[-1], 
            )
        )  
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts