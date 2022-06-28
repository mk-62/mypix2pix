import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, filters = 64):
        super(UNet, self).__init__()
        ch = filters

        self.enc_conv0 = nn.Sequential( #256(3)->128(ch)
            nn.Conv2d( 3, ch, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True))
        
        self.enc_conv1 = nn.Sequential( #128(ch)->64(2ch)
            nn.Conv2d(ch, ch*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, True))

        self.enc_conv2 = nn.Sequential( #64(2ch)->32(4ch)
            nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, True))
        
        self.enc_conv3 = nn.Sequential( #32(4ch)->16(8ch)
            nn.Conv2d(ch*4, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True))
        
        self.enc_conv4 = nn.Sequential( #16(8ch)->8(8ch)
            nn.Conv2d(ch*8, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True))
        
        self.enc_conv5 = nn.Sequential( #8(8ch)->4(8ch)
            nn.Conv2d(ch*8, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True))
        
        self.enc_conv6 = nn.Sequential( #4(8ch)->2(8ch)
            nn.Conv2d(ch*8, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True))
        
        self.enc_conv7 = nn.Sequential( #2(8ch)->1(8ch)
            nn.Conv2d(ch*8, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, True))
        
        self.dec_conv7 = nn.Sequential( #1(8ch)->2(8ch)
            nn.ConvTranspose2d(ch*8, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.ReLU(True))
        
        self.dec_conv6 = nn.Sequential( #2(8ch+8ch)->4(8ch)
            nn.ConvTranspose2d(ch*16, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.Dropout(0.5), nn.ReLU(True))
        
        self.dec_conv5 = nn.Sequential( #4(8ch+8ch)->8(8ch)
            nn.ConvTranspose2d(ch*16, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.Dropout(0.5), nn.ReLU(True))

        self.dec_conv4 = nn.Sequential( #8(8ch+8ch)->16(8ch)
            nn.ConvTranspose2d(ch*16, ch*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.Dropout(0.5), nn.ReLU(True))
        
        self.dec_conv3 = nn.Sequential( #16(8ch+8ch)->32(4ch)
            nn.ConvTranspose2d(ch*16, ch*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.ReLU(True))
        
        self.dec_conv2 = nn.Sequential( #32(4ch+4ch)->64(2ch)
            nn.ConvTranspose2d(ch*8, ch*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.ReLU(True))

        self.dec_conv1 = nn.Sequential( #64(2ch+2ch)->128(ch)
            nn.ConvTranspose2d(ch*4, ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(True))

        self.dec_conv0 = nn.Sequential( #128(ch+ch)->256(3)
            nn.ConvTranspose2d(ch*2, 3, 4, stride=2, padding=1, bias=True),
            nn.Tanh())

    @staticmethod
    def filters(state_dict):
        return state_dict['enc_conv0.0.weight'].shape[0]

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        e5 = self.enc_conv5(e4)
        e6 = self.enc_conv6(e5)
        # bottleneck
        e7 = self.enc_conv7(e6)
        d7 = self.dec_conv7(e7)
        # decoder        
        d6 = self.dec_conv6(torch.cat((d7,e6),1))
        d5 = self.dec_conv5(torch.cat((d6,e5),1))
        d4 = self.dec_conv4(torch.cat((d5,e4),1))
        d3 = self.dec_conv3(torch.cat((d4,e3),1))
        d2 = self.dec_conv2(torch.cat((d3,e2),1))
        d1 = self.dec_conv1(torch.cat((d2,e1),1))
        d0 = self.dec_conv0(torch.cat((d1,e0),1))
        return d0

    def params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self, scale = 0.02):
        def init(m):
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data, 0.0, scale)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
        self.enc_conv0.apply(init)
        self.enc_conv1.apply(init)
        self.enc_conv2.apply(init)
        self.enc_conv3.apply(init)
        self.enc_conv4.apply(init)
        self.enc_conv5.apply(init)
        self.enc_conv6.apply(init)
        self.enc_conv7.apply(init)
        self.dec_conv7.apply(init)
        self.dec_conv6.apply(init)
        self.dec_conv5.apply(init)
        self.dec_conv4.apply(init)
        self.dec_conv3.apply(init)
        self.dec_conv2.apply(init)
        self.dec_conv1.apply(init)
        self.dec_conv0.apply(init)

class PatchGAN(nn.Module):
    def __init__(self, filters = 64):
        super(PatchGAN, self).__init__()
        ch = filters

        self.enc_conv0 = nn.Sequential(#out: ch x 128 x 128
            nn.Conv2d(6, ch, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True))

        self.enc_conv1 = nn.Sequential(#out: 2ch x 64 x 64
            nn.Conv2d(ch, ch*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*2), nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_conv2 = nn.Sequential(#out: 4ch x 32 x 32
            nn.Conv2d(ch*2, ch*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch*4), nn.LeakyReLU(0.2, inplace=True))
        
        self.enc_conv3 = nn.Sequential(#out: 8ch x 31 x 31
            nn.Conv2d(ch*4, ch*8, kernel_size=4, padding=1, bias=False),
            nn.BatchNorm2d(ch*8), nn.LeakyReLU(0.2, inplace=True)) 
        
        self.enc_conv4 = nn.Sequential(#out: 1 x 30 x 30
            nn.Conv2d(ch*8, 1, kernel_size=4, padding=1, bias=True))

    def forward(self, x):
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(e0)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)        
        return e4

    def params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self, scale = 0.02):
        def init(m):
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                torch.nn.init.normal_(m.weight.data, 0.0, scale)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
        self.enc_conv0.apply(init)
        self.enc_conv1.apply(init)
        self.enc_conv2.apply(init)
        self.enc_conv3.apply(init)
        self.enc_conv4.apply(init)        