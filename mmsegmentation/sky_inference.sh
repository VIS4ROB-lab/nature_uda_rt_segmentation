# python3 tools/inferencer.py -m mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640 -w "./pretrained/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth" -p "./data/apple_farm_winter" --image x -b 1 --mask --mask_cls 2 --resize 2

python3 tools/inferencer.py -m mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640 -w "./pretrained/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth" -p "./data/apple_farm_winter/" --image x -b 1 --mask --mask_cls 2 --resize 1
