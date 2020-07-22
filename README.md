# soma-experiment



## Segmentation
https://github.com/YBIGTA/pytorch-hair-segmentation

```Workspace``` <br>
AWS Ubuntu Server 16.04 LTS (HVM), SSD Volume Type - ami-04781752c9b20ea41 (64비트 x86) >> gpu instance g4dn.xlarge <br>
cpu : 2세대 인텔 제온 확장형(Cascade Lake) 프로세서 (cores : 4) <br>
ram : 16GB <br>
ssd : 1 x 125GB (SSD) 기본 + 30GB(EBS) 추가 <br>
gpu : NVIDIA T4 Tensor 코어 GPU x1 <br>

***Setting Environment*** <br>
```sh .../segmentation/segmentation_setup.sh```

***training*** <br>
(face segmentation) <br> 
```sh .../face-segmentation/run_train.sh```
<br>

(hair segmentation) <br>
```sh .../hair-segmentation/run_train.sh```
<br><br>

***test*** <br>
(face segmentation) <br> 
```sh .../face-segmentation/run_demo.sh```
<br>

(hair segmentation) <br>
```sh .../hair-segmentation/run_demo.sh```
<br><br>

## Orientation
https://github.com/papagina/HairNet_orient2D

```Workspace``` <br>
Windows10 Docker Container created by Ubuntu 16.04 docker image
