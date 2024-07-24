# nohup > running_log.log python train.py \
python train.py \
    --data coco.yaml \
    --epochs 300 \
    --weights '' \
    --cfg tiny_yolov3.yaml \
    --batch-size 8 \
    --device 3