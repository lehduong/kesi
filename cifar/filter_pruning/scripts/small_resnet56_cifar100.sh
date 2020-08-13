# prune 1
echo "PRUNE 1" &&
python residualprune.py --dataset cifar100 \
--arch resnet56 \
--model checkpoints/pretrained/cifar100/resnet56/model_best.pth.tar \
--save checkpoints/pruned/cifar100/resnet56/prune_1 &&
# finetune 1
python finetune.py --lr 0.001 \
--schedule 50 \
--refine checkpoints/pruned/cifar100/resnet56/prune_1/pruned.pth.tar \
--dataset cifar100 \
--arch resnet56 \
--no-onecycle \
--save checkpoints/pruned/cifar100/resnet56/prune_1 &&
# prune 2
echo "PRUNE 2" &&
python residualprune.py --dataset cifar100 \
--arch resnet56 \
--model checkpoints/pruned/cifar100/resnet56/prune_1/checkpoint.pth.tar \
--save checkpoints/pruned/cifar100/resnet56/prune_2 &&
# finetune 2
python finetune.py --lr 0.001 \
--schedule 50 \
--refine checkpoints/pruned/cifar100/resnet56/prune_2/pruned.pth.tar \
--dataset cifar100 \
--arch resnet56 \
--no-onecycle \
--save checkpoints/pruned/cifar100/resnet56/prune_2 &&
# prune 3
echo "PRUNE 3" &&
python residualprune.py --dataset cifar100 \
--arch resnet56 \
--model checkpoints/pruned/cifar100/resnet56/prune_2/checkpoint.pth.tar \
--save checkpoints/pruned/cifar100/resnet56/prune_3 &&
# finetune 3
python finetune.py --lr 0.001 \
--schedule 50 \
--refine checkpoints/pruned/cifar100/resnet56/prune_3/pruned.pth.tar \
--dataset cifar100 \
--arch resnet56 \
--no-onecycle \
--save checkpoints/pruned/cifar100/resnet56/prune_3 &&
# prune 4
echo "PRUNE 4" &&
python residualprune.py --dataset cifar100 \
--arch resnet56 \
--model checkpoints/pruned/cifar100/resnet56/prune_3/checkpoint.pth.tar \
--save checkpoints/pruned/cifar100/resnet56/prune_4 &&
# finetune 4
python finetune.py --lr 0.001 \
--schedule 50 \
--refine checkpoints/pruned/cifar100/resnet56/prune_4/pruned.pth.tar \
--dataset cifar100 \
--arch resnet56 \
--no-onecycle \
--save checkpoints/pruned/cifar100/resnet56/prune_4 &&
# prune 5
echo "PRUNE 5" &&
python residualprune.py --dataset cifar100 \
--arch resnet56 \
--model checkpoints/pruned/cifar100/resnet56/prune_4/checkpoint.pth.tar \
--save checkpoints/pruned/cifar100/resnet56/prune_5 &&
# finetune 4
python finetune.py --lr 0.001 \
--schedule 50 \
--refine checkpoints/pruned/cifar100/resnet56/prune_5/pruned.pth.tar \
--dataset cifar100 \
--arch resnet56 \
--no-onecycle \
--save checkpoints/pruned/cifar100/resnet56/prune_5 &&
# ensemble finetune
echo "ENSEMBLE FINETUNE" &&
python ensemble_finetune.py --lr 0.1 \
--batch-size 128 \
--gamma 0.2 \
--schedule 20 30 \
--wd 1e-4 \
--refine checkpoints/pruned/cifar100/resnet56/prune_5/checkpoint.pth.tar \
--dataset cifar100 --save checkpoints/pruned/cifar100/resnet56/snapshot_ensemble --arch resnet56 \
--teachers checkpoints/pruned/cifar100/resnet56/prune_5/checkpoint.pth.tar \
checkpoints/pruned/cifar100/resnet56/prune_4/checkpoint.pth.tar \
checkpoints/pruned/cifar100/resnet56/prune_3/checkpoint.pth.tar \
checkpoints/pruned/cifar100/resnet56/prune_2/checkpoint.pth.tar \
checkpoints/pruned/cifar100/resnet56/prune_1/checkpoint.pth.tar \
checkpoints/pretrained/cifar100/resnet56/model_best.pth.tar