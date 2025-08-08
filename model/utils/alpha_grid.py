import torch
import torch.nn.functional as F

class AlphaGridMask(torch.nn.Module):
    def __init__(self, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.aabb = aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0 / self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor(
            [
                alpha_volume.shape[-1],
                alpha_volume.shape[-2],
                alpha_volume.shape[-3]
            ]
        ).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(
            self.alpha_volume, 
            xyz_sampled.view(1, -1, 1, 1, 3), 
            align_corners = True
        ).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invgridSize - 1