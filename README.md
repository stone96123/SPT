# [AAAI 2024] Occluded Person Re-identification via Saliency-Guided Patch Transfer
The official repository for Occluded Person Re-identification via Saliency-Guided Patch Transfer [[pdf]]([https://arxiv.org/pdf/2411.01225](https://ojs.aaai.org/index.php/AAAI/article/view/28312))

### Prepare Datasets

```bash
mkdir data
```
Download the person datasets [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Occluded-Duke](https://github.com/lightas/Occluded-DukeMTMC-Dataset), and the [Occluded_REID](https://github.com/wangguanan/light-reid/blob/master/reid_datasets.md), 
Then unzip them and rename them under the directory like

```
data
├── Occluded_Duke
│   └── images ..
├── Occluded_REID
│   └── images ..
├── market1501
│   └── images ..
└── dukemtmcreid
    └── images ..
```

### Installation

```bash
pip install -r requirements.txt
```

### Prepare ViT Pre-trained Models

You need to download the ImageNet pretrained transformer model : [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

## Training

We utilize 1 3090 GPU for training.

You can train the SPT with:

**First step:**

Train the SPS model:

```bash
python train_pt.py --config_file configs/vit_base.yml MODEL.DEVICE_ID "('your device id')"
```

**Second step:**

Train the ReID model:

```bash
python train_pt.py --config_file configs/vit_base.yml MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained SPS checkpoints')"
```

**Some examples:**
```bash
# Occluded_Duke: 
python train_pt.py --config_file configs/OCC_Duke/vit_base.yml MODEL.DEVICE_ID "('0')"
python train.py --config_file configs/OCC_Duke/vit_base.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT "('./logs/occ_duke_vit_base/sps.pth')"
```

## Evaluation

```bash
python test.py --config_file 'choose which config to test' MODEL.DEVICE_ID "('your device id')" TEST.WEIGHT "('your path of trained checkpoints')"
```

**Some examples:**
```bash
# OCC_Duke
python test.py --config_file configs/OCC_Duke/dpm.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/occ_duke_vit_base/transformer_150.pth'
```

## Citation
Please kindly cite this paper in your publications if it helps your research:
```bash
@inproceedings{tan2024occluded,
  title={Occluded person re-identification via saliency-guided patch transfer},
  author={Tan, Lei and Xia, Jiaer and Liu, Wenfeng and Dai, Pingyang and Wu, Yongjian and Cao, Liujuan},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={38},
  number={5},
  pages={5070--5078},
  year={2024}
}
```

## Acknowledgement
Our code is based on [TransReID](https://github.com/damo-cv/TransReID)[1]

## References
[1]Shuting He, Hao Luo, Pichao Wang, Fan Wang, Hao Li, and Wei Jiang. 2021. Transreid: Transformer-based object re-identification. In Proceedings of the IEEE/CVF
International Conference on Computer Vision. 15013–15022.

## Contact

If you have any question, please feel free to contact us. E-mail: [tanlei@stu.xmu.edu.cn](mailto:tanlei@stu.xmu.edu.cn)

