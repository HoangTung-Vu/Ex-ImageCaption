import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from models.CvTIC import EncoderWrapperCvT
import torch

projection = torch.nn.Linear(384,100)
enc = EncoderWrapperCvT(projection = projection)
dummy_input = torch.randn(1,3,224,224)

out_en = enc(dummy_input)
print(out_en.shape)