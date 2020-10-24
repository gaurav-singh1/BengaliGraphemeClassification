export CUDA_VISIBLE_DEVICES=0
export IMG_HEIGHT=137
export IMG_WIDTH=137
export EPOCHS=50
export TRAIN_BATCH_SIZE=32
export TEST_BATCH_SIZE=16
export MODEL_MEAN="(0.4, 0.4, 0.4)"
export MODEL_STD="(0.2,0.2,0.2)"
export TRAINING_FOLDS="(0, 1, 2, 3)"
export VALIDATION_FOLDS="(4,)"
export BASE_MODEL="resnet34"
python train.py

