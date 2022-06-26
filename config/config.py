from easydict import EasyDict as edict

config = edict()
config.dataset = "webface" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
# batch size per GPU
config.lr = 0.1
config.output = "output/cls_10k_kT_only" # train model output folder
config.output_real = "r50_webface_COSFace/31614backbone.pth" # train model output folder

config.global_step=0 # step to resume
config.s=64.0
config.m=0.35
config.std=0.05
config.sample=10


config.loss="CosFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss

# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet50"
config.SE=False # SEModule



if config.dataset == "webface":
    config.rec = "data/validation/" #"/home/psiebke/faces_webface_112x112"
    config.data_path="data/sg2cond_aligned"
    config.num_classes = 10572
    config.num_image = 95150
    config.num_epoch = 64
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step= 2000
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch])
    config.lr_func = lr_step_func
