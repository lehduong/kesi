python train.py -a resnet56 \
-d cifar10 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar10/resnet56 && \

python train.py -a resnet56 \
-d cifar100 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar100/resnet56 && \

python train.py -a resnet110 \
-d cifar10 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar10/resnet110 && \

python train.py -a resnet110 \
-d cifar100 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar100/resnet110 && \

python train.py -a preresnet164 \
-d cifar10 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar10/preresnet164 && \

python train.py -a preresnet164 \
-d cifar100 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar100/preresnet164 && \

python train.py -a vgg16 \
-d cifar10 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar10/vgg16 && \

python train.py -a vgg16 \
-d cifar100 \
--epochs 300 \
--schedule 150 225 \
--gamma 0.1 \
--wd 1e-4 \
--save checkpoints/pretrained/cifar100/vgg16 && \

python train.py -a wrn_16_8 \
-d cifar10 \
--epochs 200 \
--schedule 60 120 160 \
--gamma 0.2 \
--wd 5e-4 \
--save checkpoints/pretrained/cifar10/wrn_16_8 && \

python train.py -a wrn_16_8 \
-d cifar100 \
--epochs 200 \
--schedule 60 120 160 \
--gamma 0.2 \
--wd 5e-4 \
--save checkpoints/pretrained/cifar100/wrn_16_8