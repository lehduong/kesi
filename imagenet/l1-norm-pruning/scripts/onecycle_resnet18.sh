echo "PRUNE: 1" &&
python residualprune.py /content/tiny-imagenet-200 \
                        --save prune_1 --arch resnet18 --test-batch-size 128 \
                        --model checkpoints/checkpoint.pth.tar \
                        --num_classes 200 &&
python main_finetune.py /content/tiny-imagenet-200 \
                        --save prune_1 --arch resnet18 --lr 0.1 --num_classes 200 -b 128 --refine \
                        prune_1/pruned.pth.tar --print-freq 100 --schedule 8 15 20 --epochs 25 &&
echo "PRUNE: 2" &&
python residualprune.py /content/tiny-imagenet-200 \
                        --save prune_2 --arch resnet18 --test-batch-size 128 \
                        --model prune_1/checkpoint.pth.tar \
                        --num_classes 200 &&
python main_finetune.py /content/tiny-imagenet-200 \
                        --save prune_2 --arch resnet18 --lr 0.1 --num_classes 200 -b 128 --refine \
                        prune_2/pruned.pth.tar --print-freq 100 --schedule 8 15 20 --epochs 25 &&
echo "PRUNE: 3" &&
python residualprune.py /content/tiny-imagenet-200 \
                        --save prune_3 --arch resnet18 --test-batch-size 128 \
                        --model prune_2/checkpoint.pth.tar \
                        --num_classes 200 &&
python main_finetune.py /content/tiny-imagenet-200 \
                        --save prune_3 --arch resnet18 --lr 0.1 --num_classes 200 -b 128 --refine \
                        prune_3/pruned.pth.tar --print-freq 100 --schedule 8 15 20 --epochs 25 &&
echo "PRUNE: 4" &&
python residualprune.py /content/tiny-imagenet-200 \
                        --save prune_4 --arch resnet18 --test-batch-size 128 \
                        --model prune_3/checkpoint.pth.tar \
                        --num_classes 200 &&
python main_finetune.py /content/tiny-imagenet-200 \
                        --save prune_4 --arch resnet18 --lr 0.1 --num_classes 200 -b 128 --refine \
                        prune_4/pruned.pth.tar --print-freq 100 --schedule 8 15 20 --epochs 25 &&
echo "ENSEMBLE FINETUNE" &&
python ensemble_finetune.py /content/tiny-imagenet-200 \
                        --save kd --arch resnet18 --lr 0.001 --num_classes 200 -b 128 --refine \
                        prune_4/checkpoint.pth.tar --print-freq 100 --schedule 8 15 20 --epochs 25