```bash
$ git clone https://github.com/BICLab/EMS-YOLO.git
$ pip install -r requirements.txt
```

</details>

### Pretrained Checkpoints

We provide the best and the last trained model based on EMS-Res34 on the COCO dataset.

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [COCO_EMS-ResNet34](https://drive.google.com/drive/folders/1mry8sdED6ncqxajmQROKBECpcrmXStpB?usp=sharing) .

The relevant parameter files are in the `runs/train`.


### Training & Addition
<details open>
<summary>Train</summary>

The relevant code for the Gen1 dataset is at `/g1-resnet`. It needs to be replaced or added to the appropriate root folder.

For gen1 dataset:

```python
python path/to/train_g1.py --weights ***.pt --img 640
```
For coco dataset:
```python
python train.py
```
</details>


Calculating the spiking rate:

Dependencies can be downloaded from [Visualizer](https://github.com/luo3300612/Visualizer).
```python
python calculate_fr.py
```

### Contact Information


```shell
@inproceedings{su2023deep,
  title={Deep Directly-Trained Spiking Neural Networks for Object Detection},
  author={Su, Qiaoyi and Chou, Yuhong and Hu, Yifan and Li, Jianing and Mei, Shijie and Zhang, Ziyang and Li, Guoqi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6555--6565},
  year={2023}
}
```

<p>
YOLOv3  is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development. 
 
 <b>Our code is also implemented in this framework, so please remember to cite their work.</b>
</p>


