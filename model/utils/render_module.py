import torch
from model.utils.p_encode import positional_encoding
    
class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe = 6, feape = 6, featureC = 128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2 * viewpe * 3 + 2 * feape * inChanel + 3 + inChanel
        self.viewpe, self.feape = viewpe, feape

        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace = True), 
            layer2, torch.nn.ReLU(inplace = True), 
            layer3
        )

        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        
        indata = [features, viewdirs]
        
        if self.feape > 0: indata += [positional_encoding(features, self.feape)]
        
        if self.viewpe > 0: indata += [positional_encoding(viewdirs, self.viewpe)]
        
        mlp_in = torch.cat(indata, dim = -1)
        
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe = 6, pospe = 6, featureC = 128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3 + 2 * viewpe * 3) + (3 + 2 * pospe * 3)  + inChanel
        self.viewpe, self.pospe = viewpe, pospe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC, 3)

        self.mlp = torch.nn.Sequential(
            layer1, torch.nn.ReLU(inplace = True), 
            layer2, torch.nn.ReLU(inplace = True), 
            layer3
        )
        
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        
        indata = [features, viewdirs]
        
        if self.pospe > 0: indata += [positional_encoding(pts, self.pospe)]
        
        if self.viewpe > 0: indata += [positional_encoding(viewdirs, self.viewpe)]
        
        mlp_in = torch.cat(indata, dim = -1)
        
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb