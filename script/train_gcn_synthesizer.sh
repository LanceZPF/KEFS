python trainer.py --manualSeed 806 \
--cls_weight 0.01 --nclass_all 21 --syn_num 500 --val_every 1 \
--cuda --netG_name MLP_G \
--netD_name MLP_D --nepoch 150 --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 \
--critic_iter 5 \
--dataset voc --batch_size 2 --nz 300 --attSize 300 --resSize 1024 \
--lr 0.00001 \
--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
--pretrain_classifier /data5/zpf/RRFS-main-new/mmdetection/tools/work_dirs/faster_rcnn_r101_gcn/epoch_4.pth \
--class_embedding /data5/zpf/RRFS-main-new/VOC/fasttext_synonym.npy \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--lz_ratio 0 \
--outname checkpoints/VOC \
--lambda_contra 0.001 --tau 0.1 --num_negative 10 \
--radius 0.0000001 --inter_weight 0.001 --inter_temp 0.1\