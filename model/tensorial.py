import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.utils.render_module import MLPRender_Fea, MLPRender_PE
from model.utils.alpha_grid import AlphaGridMask
from model.utils.raw2alpha import raw2alpha

class TensorBase(torch.nn.Module):
    def __init__(self, config):
        super(TensorBase, self).__init__()

        self.device                 = "cuda" if torch.cuda.is_available() else "cpu" 

        self.density_n_comp         = config['density_n_comp']  # 8
        self.app_n_comp             = config['appearance_n_comp']  # 24
        self.app_dim                = config['app_dim'] # 27
        self.aabb                   = config['aabb'].to(self.device)
        self.alphaMask              = config['alphaMask'] # None
        

        self.density_shift          = config['density_shift'] # -10
        self.alphaMask_thres        = config['alphaMask_thres'] # 0.001
        self.distance_scale         = config['distance_scale'] # 25
        self.rayMarch_weight_thres  = config['rayMarch_weight_thres'] # 0.0001
        self.fea2denseAct           = config['fea2denseAct'] # 'softplus'

        self.near_far               = config['near_far'] # [2.0, 6.0]
        self.step_ratio             = config['step_ratio'] # 2.0

        self.matMode                = [[0, 1], [0, 2], [1, 2]]
        self.vecMode                = [2, 1, 0]
        self.comp_w                 = [1, 1, 1]

        self.update_stepSize(config['gridSize'])
        self.init_svd_volume(config['gridSize'][0], self.device)

        self.shadingMode            = config['shadingMode'] # 'MLP_PE'
        self.pos_pe                 = config['pos_pe'] # 6
        self.view_pe                = config['view_pe'] # 6
        self.fea_pe                 = config['fea_pe'] # 6
        self.featureC               = config['featureC'] # 128

        self.init_render_func(
            self.shadingMode, 
            self.pos_pe, 
            self.view_pe, 
            self.fea_pe, 
            self.featureC, 
            self.device
        )

        return
    
    def get_kwargs(self):
        return {
            'aabb':                     self.aabb,
            'gridSize':                 self.gridSize.tolist(),
            'density_n_comp':           self.density_n_comp,
            'appearance_n_comp':        self.app_n_comp,
            'app_dim':                  self.app_dim,

            'density_shift':            self.density_shift,
            'alphaMask_thres':          self.alphaMask_thres,
            'distance_scale':           self.distance_scale,
            'rayMarch_weight_thres':    self.rayMarch_weight_thres,
            'fea2denseAct':             self.fea2denseAct,

            'near_far':                 self.near_far,
            'step_ratio':               self.step_ratio,

            'shadingMode':              self.shadingMode,
            'pos_pe':                   self.pos_pe,
            'view_pe':                  self.view_pe,
            'fea_pe':                   self.fea_pe,
            'featureC':                 self.featureC
        }

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(
                self.app_dim, view_pe, pos_pe, featureC
            ).to(device)
        
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(
                self.app_dim, view_pe, fea_pe, featureC
            ).to(device)
        
        else: exit()
        return 

    def update_stepSize(self, gridSize):
        self.aabbSize       = self.aabb[1] - self.aabb[0]
        self.invaabbSize    = 2.0 / self.aabbSize
        self.gridSize       = torch.LongTensor(gridSize).to(self.device)
        self.units          = self.aabbSize / (self.gridSize-1)
        self.stepSize       = torch.mean(self.units)*self.step_ratio
        self.aabbDiag       = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples       = int((self.aabbDiag / self.stepSize).item()) + 1
        return

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device = device)
        )
        
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device = device)
        )
        
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias = False, device = device)

    def compute_features(self, xyz_sampled):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]], 
                xyz_sampled[..., self.matMode[1]], 
                xyz_sampled[..., self.matMode[2]]
            )
        ).detach()

        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]], 
                xyz_sampled[..., self.vecMode[1]], 
                xyz_sampled[..., self.vecMode[2]]
            )
        )
        
        coordinate_line = torch.stack(
            (
                torch.zeros_like(coordinate_line), coordinate_line
            ), dim = -1
        ).detach()

        plane_feats = F.grid_sample(
            self.plane_coef[:, -self.density_n_comp:], 
            coordinate_plane, 
            align_corners = True
        ).view(-1, *xyz_sampled.shape[:1])
        
        line_feats = F.grid_sample(
            self.line_coef[:, -self.density_n_comp:], 
            coordinate_line, 
            align_corners=True
        ).view(-1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim = 0)
        
        plane_feats = F.grid_sample(
            self.plane_coef[:, :self.app_n_comp], 
            coordinate_plane, 
            align_corners = True
        ).view(3 * self.app_n_comp, -1)
        
        line_feats = F.grid_sample(
            self.line_coef[:, :self.app_n_comp], 
            coordinate_line, 
            align_corners = True
        ).view(3 * self.app_n_comp, -1)
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features
    
    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]], 
                xyz_sampled[..., self.matMode[1]], 
                xyz_sampled[..., self.matMode[2]]
            )
        ).detach().view(3, -1, 1, 2)
        
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]], 
                xyz_sampled[..., self.vecMode[1]], 
                xyz_sampled[..., self.vecMode[2]]
            )
        )
        
        coordinate_line = torch.stack(
            (
                torch.zeros_like(coordinate_line), 
                coordinate_line
            ), dim=-1
        ).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(
            self.plane_coef[:, -self.density_n_comp:], 
            coordinate_plane, 
            align_corners = True
        ).view(-1, *xyz_sampled.shape[:1])
        
        line_feats = F.grid_sample(
            self.line_coef[:, -self.density_n_comp:], 
            coordinate_line, 
            align_corners = True
        ).view(-1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack(
            (
                xyz_sampled[..., self.matMode[0]], 
                xyz_sampled[..., self.matMode[1]], 
                xyz_sampled[..., self.matMode[2]]
            )
        ).detach().view(3, -1, 1, 2)
        
        coordinate_line = torch.stack(
            (
                xyz_sampled[..., self.vecMode[0]], 
                xyz_sampled[..., self.vecMode[1]], 
                xyz_sampled[..., self.vecMode[2]]
            )
        )
        
        coordinate_line = torch.stack(
            (
                torch.zeros_like(coordinate_line), 
                coordinate_line
            ), dim = -1
        ).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(
            self.plane_coef[:, :self.app_n_comp], 
            coordinate_plane, 
            align_corners = True
        ).view(3 * self.app_n_comp, -1)
        
        line_feats = F.grid_sample(
            self.line_coef[:, :self.app_n_comp], 
            coordinate_line, 
            align_corners = True
        ).view(3 * self.app_n_comp, -1)
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return app_features
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [
            {
                'params': self.line_coef, 
                'lr': lr_init_spatialxyz
            }, 
            {
                'params': self.plane_coef, 
                'lr': lr_init_spatialxyz
            },
            {
                'params': self.basis_mat.parameters(), 
                'lr': lr_init_network
            }
        ]
        
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params': self.renderModule.parameters(), 'lr': lr_init_network}]
        
        return grad_vars

    def sample_ray_ndc(self, rays_o, rays_d, is_train = True, N_samples = -1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim = -1)
        
        return rays_pts, interpx, ~mask_outbbox
    
    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
            mask_inbbox = (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        
        return mask_filtered

    def sample_ray(self, rays_o, rays_d, is_train = True, N_samples = -1):
        
        N_samples = N_samples if N_samples > 0 else self.nSamples
        
        stepsize = self.stepSize
        near, far = self.near_far
        
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min = near, max = far)

        rng = torch.arange(N_samples)[None].float()
        
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)    
            rng += torch.rand_like(rng[:, [0]])
        
        step = stepsize * rng.to(rays_o.device)
        interpx = t_min[..., None] + step

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim = -1)

        return rays_pts, interpx, ~mask_outbbox

    @torch.no_grad()
    def getDenseAlpha(self, gridSize = None):
        
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, gridSize[0]),
                torch.linspace(0, 1, gridSize[1]),
                torch.linspace(0, 1, gridSize[2]),
                ), -1).to(self.device)
        
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200, 200, 200)):

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None, None]

        ks = 3
        alpha = F.max_pool3d(
            alpha, 
            kernel_size = ks, 
            padding = ks // 2, 
            stride = 1
        ).view(gridSize[::-1])
        
        alpha[alpha >= self.alphaMask_thres] = 1
        alpha[alpha < self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.aabb, alpha)

        valid_xyz = dense_xyz[alpha > 0.5]

        xyz_min,xyz_max = valid_xyz.amin(0), valid_xyz.amax(0)

        new_aabb = torch.stack(
            (
                xyz_min, 
                xyz_max
            )
        )

        return new_aabb


    def feature2density(self, density_features):
        return F.softplus(density_features + self.density_shift)
        

    def compute_alpha(self, xyz_locs, length = 1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype = bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device = xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma * length).view(xyz_locs.shape[:-1])

        return alpha
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    plane_coef[i].data, 
                    size = (res_target[mat_id_1], res_target[mat_id_0]), 
                    mode = 'bilinear',
                    align_corners = True
                )
            )
            
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(
                    line_coef[i].data, 
                    size = (res_target[vec_id], 1), 
                    mode = 'bilinear', 
                    align_corners = True
                )
            )

        return plane_coef, line_coef
    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        
        scale = res_target[0] / self.line_coef.shape[2] 
        
        plane_coef = F.interpolate(
            self.plane_coef.detach().data, 
            scale_factor = scale, 
            mode = 'bilinear',
            align_corners = True
        )
        
        line_coef = F.interpolate(
            self.line_coef.detach().data, 
            size = (res_target[0], 1), 
            mode = 'bilinear', 
            align_corners = True
        )

        self.plane_coef, self.line_coef = torch.nn.Parameter(plane_coef), torch.nn.Parameter(line_coef)
        
        self.update_stepSize(res_target)

        return


    def forward(self, rays_chunk, white_bg = True, is_train = False, ndc_ray = False, N_samples = -1):
        
        viewdirs = rays_chunk[:, 3:6]
        
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(
                rays_chunk[:, :3], 
                viewdirs, 
                is_train = is_train, 
                N_samples = N_samples
            )
            
            dists = torch.cat(
                (
                    z_vals[:, 1:] - z_vals[:, :-1], 
                    torch.zeros_like(z_vals[:, :1])
                ), dim = -1
            )
            
            rays_norm = torch.norm(viewdirs, dim = -1, keepdim = True)
            
            dists = dists * rays_norm
            
            viewdirs = viewdirs / rays_norm
        
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(
                rays_chunk[:, :3], 
                viewdirs, 
                is_train = is_train,
                N_samples = N_samples
            )

            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device = xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device = xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,)) < 0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map 
    
    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {
            'kwargs': kwargs, 
            'state_dict': self.state_dict()
        }
        
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            
            ckpt.update(
                {
                    'alphaMask.shape': alpha_volume.shape
                }
            )
            
            ckpt.update(
                {
                    'alphaMask.mask': np.packbits(
                        alpha_volume.reshape(-1)
                    )
                }
            )
            ckpt.update(
                {
                    'alphaMask.aabb': 
                    self.alphaMask.aabb.cpu()
                }
            )
        torch.save(ckpt, path)

    def load(self, ckpt):
        
        if 'alphaMask.aabb' in ckpt.keys():
            
            length = np.prod(
                ckpt['alphaMask.shape']
            )
            
            alpha_volume = torch.from_numpy(
                np.unpackbits(
                    ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape'])
                )
            
            self.alphaMask = AlphaGridMask(
                self.device, 
                ckpt['alphaMask.aabb'].to(self.device), 
                alpha_volume.float().to(self.device)
            )
        
        self.load_state_dict(ckpt['state_dict'])