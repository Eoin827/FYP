import torch
import yaml

from ..IHCApproxNH.classes import WaveNet
from ..IHCApproxNH.utils import utils

with open("../IHCApproxNH/config/config31rfa3-1fullSet.yaml", "r") as ymlfile:
    conf = yaml.safe_load(ymlfile)  #
ihcogramMax = torch.tensor(1.33)
ihcogramMax = utils.comp(ihcogramMax, conf["scaleWeight"], conf["scaleType"])
fs = 16000

skipLength = (2 ** conf["nLayers"]) * conf["nStacks"]

IHCogram = WaveNet.WaveNet(
    conf["nLayers"],
    conf["nStacks"],
    conf["nChannels"],
    conf["nResChannels"],
    conf["nSkipChannels"],
    conf["numOutputLayers"],
)

IHCogram.load_state_dict(
    torch.load(
        "../IHCApproxNH/model/musan31rfa3-1fullSet_20231014-145738.pt",
        map_location=torch.device("cuda:0"),
        weights_only=True,
    )
)

for param in IHCogram.parameters():
    param.requires_grad = False


# maybe need to permute ????
def forward(x):
    with torch.no_grad():
        pred = IHCogram(x)

        pred = pred * ihcogramMax
        return utils.invcomp(pred, conf["scaleWeight"], conf["scaleType"])
