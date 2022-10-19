runs=10

dir=/persist/kjordan/ffcv-imagenet
for gpu in 0 1 2 3 4 5 6 7
do
    CUDA_VISIBLE_DEVICES=$gpu conda run -n ffcv python $dir/train_imagenet.py --config-file $dir/rn18.yaml \
        --data.train_dataset=/persist/kjordan/data_image/ffcv_imagenet/train_300_0_100.ffcv \
        --data.val_dataset=/persist/kjordan/data_image/ffcv_imagenet/val.ffcv \
        --data.num_workers=12 --logging.folder=$dir/logs/ --logging.log_level=2 \
        --training.runs=$runs 2> $dir/err$gpu.txt &
done
wait
echo ALLDONE

