import torch
import torch.nn as nn
import torchsummary

from layers import TransformerEncoder

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True, joint:bool=False):
        super(ViT, self).__init__()
        # hidden=384
        self.joint = joint
        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)
    
        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )


    def forward(self, x, joint_size=None, eval_method=None):
        if self.joint:
            tokens = self._to_words(x)
            if self.training:
                tokens = tokens.reshape(tokens.shape[0] // joint_size, tokens.shape[1] * joint_size , -1)
            else:
                if eval_method == 'zeros':
                    zeros = torch.zeros(tokens.shape[0], tokens.shape[1] * (joint_size - 1), tokens.shape[2]).cuda()
                    tokens = torch.cat([tokens, zeros], dim=1)
                elif eval_method == "single":
                    pass
                else:
                    raise not implemented
            out = self.emb(tokens)
            if out.shape[1] != self.pos_emb.shape[1]:
                out += self.pos_emb.repeat(1, joint_size, 1)
            else:
                out += self.pos_emb
            out = self.enc(out)                              
            out = self.fc(out)
            if self.training:
                return out.reshape(out.shape[0] * out.shape[1], -1)
            else:
                return out[:, :self.patch**2, :].mean(dim=1)
        out = self._to_words(x)
        out = self.emb(out)            
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out


if __name__ == "__main__":
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViT(in_c=c, num_classes= 10, img_size=h, patch=16, dropout=0.1, num_layers=7, hidden=384, head=12, mlp_hidden=384, is_cls_token=False)
    # out = net(x)
    # out.mean().backward()
    torchsummary.summary(net, (c,h,w))
    # print(out.shape)
    