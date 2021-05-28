from torch import nn

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.t1 = nn.Sequential(
            # in: 3 * 64 * 64
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # out: 64 * 32 * 32
        )

        self.t2 = nn.Sequential(
            # in: 64 * 32 * 32
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # out: 128 * 16 * 16
        )

        self.t3 = nn.Sequential(
            # in: 128 * 16 * 16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # out: 256 * 8 * 8
        )

        self.t4 = nn.Sequential(
            # in: 256 * 8 * 8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            # out: 512 * 4 * 4
        )

        self.t5 = nn.Sequential(
            # in: 512 * 4 * 4
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 * 1 * 1
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        return x




