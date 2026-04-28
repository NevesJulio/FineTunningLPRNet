import torch
from model.LPRNet import build_lprnet

device = torch.device("cpu")

num_classes = 68  # ou len(CHARS)

model = build_lprnet(
    lpr_max_len=8,
    phase=False,
    class_num=num_classes,
    dropout_rate=0
)

model.load_state_dict(torch.load("weights/Final_LPRNet_model.pth", map_location=device))
model.eval()

dummy = torch.randn(1, 3, 24, 94)

torch.onnx.export(
    model,
    dummy,
    "lprnet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    dynamic_axes=None
)

print("ONNX exportado com sucesso!")