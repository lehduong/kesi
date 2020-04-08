# prune 1
echo "PRUNE 1" &&
python vggprune.py --dataset cifar100 \
                   --arch vgg16 \
                   --model checkpoints/model_best.pth.tar \
                   --save prune_1 && 
python finetune.py --lr 0.01 \
                   --gamma 0.2 \
                   --schedule 50 \
                   --refine prune_1/pruned.pth.tar \
                   --dataset cifar100 \
                   --arch vgg16 \
                   --save prune_1 &&
# prune 2
echo "PRUNE 2" &&
python vggprune.py --dataset cifar100 \
                   --arch vgg16 \
                   --model prune_1/checkpoint.pth.tar \
                   --save prune_2 && 
python finetune.py --lr 0.01 \
                   --gamma 0.2 \
                   --schedule 50 \
                   --refine prune_2/pruned.pth.tar \
                   --dataset cifar100 \
                   --arch vgg16 \
                   --save prune_2 &&
# prune 3
echo "PRUNE 3" &&
python vggprune.py --dataset cifar100 \
                   --arch vgg16 \
                   --model prune_2/checkpoint.pth.tar \
                   --save prune_3 && 
python finetune.py --lr 0.01 \
                   --gamma 0.2 \
                   --schedule 50 \
                   --refine prune_3/pruned.pth.tar \
                   --dataset cifar100 \
                   --arch vgg16 \
                   --save prune_3 &&
# prune 4
echo "PRUNE 4" &&
python vggprune.py --dataset cifar100 \
                   --arch vgg16 \
                   --model prune_3/checkpoint.pth.tar \
                   --save prune_4 && 
python finetune.py --lr 0.01 \
                   --gamma 0.2 \
                   --schedule 50 \
                   --refine prune_4/pruned.pth.tar \
                   --dataset cifar100 \
                   --arch vgg16 \
                   --save prune_4 &&
# prune 5
echo "PRUNE 5" &&
python vggprune.py --dataset cifar100 \
                   --arch vgg16 \
                   --model prune_4/checkpoint.pth.tar \
                   --save prune_5 && 
python finetune.py --lr 0.01 \
                   --gamma 0.2 \
                   --schedule 50 \
                   --refine prune_5/pruned.pth.tar \
                   --dataset cifar100 \
                   --arch vgg16 \
                   --save prune_5 &&
# ensemble finetune
echo "ENSEMBLE FINETUNE" &&
python ensemble_finetune.py --lr 0.01 \
                            --batch-size 128 \
                            --gamma 0.2 \
                            --schedule 20 30 \
                            --wd 1e-4 \
                            --refine prune_5/finetuned.pth.tar \
                            --dataset cifar100 --save snapshot_ensemble --arch vgg16