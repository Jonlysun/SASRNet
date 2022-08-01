from pathlib import Path
import socket
import platform
import getpass

HOSTNAME = socket.gethostname()
PLATFORM = platform.system()
USER = getpass.getuser()

train_device = "cuda:0"
eval_device = "cuda:0"

### Path
Tanks_and_Temples_path = "/home/user2/datasets/Tanks_and_Temples_HLBIC_FR/LR/x4/"
ETH_path = "/home/user2/datasets/ETH_v1/LR/"
Project_Path = "/home/user2/SASRNet"

train = True
save_alpha = False
if train == True:
    tat_root = Path(Tanks_and_Temples_path)
else:
    tat_root = Path(ETH_path)

### TOP PRIORITY
depth_train_only = False
feature_refine = True
vs_help_sr = True
sr_help_vs = True

### Train phrase
if depth_train_only:
    experiment_name = "last-200000-ssimalpha-dp-only"
else:
    experiment_name = "last-200000-ssimalpha-final"

### Hyper-parameter
min_img_width = 100000
max_img_width = -1
min_img_height = 100000
max_img_height = -1
train_scale = 0.50

train_iters = 200000
lr = 1e-4
pad_width = 4
ycbcr = False
train_batch_size = 1
eval_batch_size = 1
train_patch = 100
nbs = 3
scale = 4
pin_memory = True
manul_save = 5000
manul_save_model = 10000
clamp = False
data_augment = False
sample_patch = False
net = 'warp_offset'
num_res_blocks = '16+16+8+4'
nf = 64
use_perceptual = False
use_gan = False
use_edge_loss = False
use_sr_loss = False
use_pretrained_sr = False
# pretrained_sr_path = './pretrained_result/state_dict/best_model.pth'
pretrained_dp_esti = 'experiments/last-200000-ssimalpha-dp-only/net_0000000000199999.params'

pretrained_requires_grad = False
use_ip_offset = True
use_pcd_align_sr = True
use_sr = False
use_unfold = True
single = False
bi_rnn = True

beta1 = 0.9
beta2 = 0.99
T_period = [train_iters, train_iters]
restarts = [train_iters]
restart_weights = [1]
eta_min = 1e-7
Refine_RBs = 20

gan_type = "LSGAN"
TTSR_loss_per_begin = 0
T_period_gan = [train_iters-TTSR_loss_per_begin, train_iters-TTSR_loss_per_begin]
restarts_gan = [train_iters-TTSR_loss_per_begin]
restart_weights_gan = [1]

pretrain_iters = 100000
pretrain_lr = 1e-4
pretrain_beta1 = 0.9
pretrain_beta2 = 0.99
pretrain_T_period = [100000, 100000]
pretrain_restarts = [100000]
pretrain_restart_weights = [1]
pretrain_eta_min = 1e-7

tat_eval_tracks = {}

if train == True:
    tat_train_sets = [
        "training/Barn",
        "training/Caterpillar",
        "training/Church",
        "training/Courthouse",
        "training/Ignatius",
        "training/Meetingroom",
        "intermediate/Family",
        "intermediate/Francis",
        "intermediate/Horse",
        "intermediate/Lighthouse",
        "intermediate/Panther",
        "advanced/Auditorium",
        "advanced/Ballroom",
        "advanced/Museum",
        "advanced/Temple",
        "advanced/Courtroom",
        "advanced/Palace",
    ]

    tat_eval_sets = [
        "intermediate/M60",
        "training/Truck",
        "intermediate/Playground",
        "intermediate/Train",
    ]

    tat_eval_tracks['intermediate/M60'] = [i for i in range(313)]
    tat_eval_tracks['training/Truck'] = [i for i in range(251)]
    tat_eval_tracks['intermediate/Playground'] = [i for i in range(307)]
    tat_eval_tracks['intermediate/Train'] = [i for i in range(301)]
else:

    tat_eval_sets = [
        "delivery_area",
        "electro",
        "forest",
        "playground",
        "terrains"
    ]

    tat_eval_tracks['delivery_area'] = [i for i in range(237)]
    tat_eval_tracks['electro'] = [i for i in range(300)]
    tat_eval_tracks['forest'] = [i for i in range(257)]
    tat_eval_tracks['playground'] = [i for i in range(240)]
    tat_eval_tracks['terrains'] = [i for i in range(165)]


