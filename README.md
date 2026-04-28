FineTunning/
│
├── lp_autogenerate/        # Dataset (imagens organizadas em batches)
├── model/                  # Arquitetura LPRNet
├── data/                   # Loader + CHARS + preprocessing
├── weights/               # Checkpoints (.pth)
│   └── Final_LPRNet_model.pth
│
├── train_LPRNet.py        # Script de treino
├── test_LPRNet.py         # Script de avaliação
├── export_onnx.py         # Exportação para ONNX
├── lprnet.pth             # Modelo base pré-treinado
├── README.md



Criar ambiente
python -m venv lpr_env
lpr_env\Scripts\activate


Instalar dependências
pip install torch torchvision numpy opencv-python imutils
pip install onnx onnxsim onnxruntime

Rodar no terminal

python train_LPRNet.py ^
--train_img_dirs "lp_autogenerate/train_batch0,lp_autogenerate/train_batch1,...,lp_autogenerate/train_batch15" ^
--test_img_dirs "lp_autogenerate/train_batch0" ^
--pretrained_model lprnet.pth ^
--learning_rate 0.001 ^
--max_epoch 15 ^
--train_batch_size 32 ^
--num_workers 0
