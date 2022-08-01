import torch
import numpy as np
import sys
import cv2
import logging
from pathlib import Path
import PIL

import dataset
import modules
import losses

sys.path.append("../")
import co
import ext
import config
import co.utils as co_utils
from scipy import misc
import os

class Worker(co.mytorch.Worker):
    def __init__(
        self,
        train_dsets,
        eval_dsets="",
        train_n_nbs=1,
        train_nbs_mode="argmax",
        train_scale=1,
        train_patch=64,
        eval_n_nbs=1,
        eval_scale=-1,
        n_train_iters=config.train_iters,  # 750000
        num_workers=8,
        **kwargs,
    ):
        super().__init__(
            n_train_iters=n_train_iters,
            num_workers=num_workers,
            train_device=config.train_device,
            eval_device=config.eval_device,
            **kwargs,
        )

        self.train_dsets = train_dsets
        self.eval_dsets = eval_dsets
        self.train_n_nbs = train_n_nbs
        self.train_nbs_mode = train_nbs_mode
        self.train_scale = train_scale
        self.train_patch = train_patch
        self.eval_n_nbs = eval_n_nbs
        self.eval_scale = train_scale if eval_scale <= 0 else eval_scale
        self.bwd_depth_thresh = 0.01
        self.invalid_depth_to_inf = True

        self.train_loss = losses.ImageSimilarityLoss()
        self.l1_loss = torch.nn.L1Loss().cuda()

        self.eval_loss = self.train_loss

    def get_pw_dataset(
        self,
        *,
        name,
        ibr_dir,
        im_size,
        patch,
        pad_width,
        n_nbs,
        nbs_mode,
        train,
        tgt_ind=None,
        n_max_sources=-1,
    ):
        logging.info(f"  create dataset for {name}")
        im_paths = sorted(ibr_dir.glob(f"im_*.png"))
        im_paths += sorted(ibr_dir.glob(f"im_*.jpg"))
        im_paths += sorted(ibr_dir.glob(f"im_*.jpeg"))
        dm_paths = sorted(ibr_dir.glob("dm_*.npy"))
        count_paths = sorted(ibr_dir.glob("count_0*.npy"))
        counts = []
        for count_path in count_paths:
            counts.append(np.load(count_path))
        counts = np.array(counts)
        Ks = np.load(ibr_dir / "Ks.npy")
        Rs = np.load(ibr_dir / "Rs.npy")
        ts = np.load(ibr_dir / "ts.npy")

        if tgt_ind is None:
            tgt_ind = np.arange(len(im_paths))
            src_ind = np.arange(len(im_paths))
        else:
            # src_ind = [
            #     idx for idx in range(len(im_paths)) if idx not in tgt_ind
            # ]
            src_ind = tgt_ind
        # counts = counts[tgt_ind]
        # counts = counts[:, src_ind]

        counts = None

        dset = dataset.Dataset(
            name=name,
            tgt_im_paths=[im_paths[idx] for idx in tgt_ind],
            tgt_dm_paths=[dm_paths[idx] for idx in tgt_ind],
            tgt_Ks=Ks[tgt_ind],
            tgt_Rs=Rs[tgt_ind],
            tgt_ts=ts[tgt_ind],
            tgt_counts=counts,
            src_im_paths=[im_paths[idx] for idx in src_ind],
            src_dm_paths=[dm_paths[idx] for idx in src_ind],
            src_Ks=Ks[src_ind],
            src_Rs=Rs[src_ind],
            src_ts=ts[src_ind],
            im_size=im_size,
            pad_width=pad_width,
            patch=patch,
            n_nbs=n_nbs,
            nbs_mode=nbs_mode,
            bwd_depth_thresh=self.bwd_depth_thresh,
            invalid_depth_to_inf=self.invalid_depth_to_inf,
            train=train,
            ycbcr=config.ycbcr
        )
        return dset


    def get_train_set_tat(self, dset):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.train_scale:.2f}"
        dset = self.get_pw_dataset(
            name=f'tat_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=config.pad_width,
            patch=(self.train_patch, self.train_patch),
            # patch=None,
            n_nbs=self.train_n_nbs,
            nbs_mode=self.train_nbs_mode,
            train=True,
        )
        return dset

    def get_train_set(self):
        logging.info("Create train datasets")
        dsets = co.mytorch.MultiDataset(name="train")
        if "tat" in self.train_dsets:
            for dset in config.tat_train_sets:
                dsets.append(self.get_train_set_tat(dset))
        return dsets

    def get_eval_set_tat(self, dset, mode):
        dense_dir = config.tat_root / dset / "dense"
        ibr_dir = dense_dir / f"ibr3d_pw_{self.eval_scale:.2f}"
        if mode == "all":
            tgt_ind = None
        elif mode == "subseq":
            tgt_ind = config.tat_eval_tracks[dset]
        else:
            raise Exception("invalid mode for get_eval_set_tat")
        dset = self.get_pw_dataset(
            name=f'tat_{mode}_{dset.replace("/", "_")}',
            ibr_dir=ibr_dir,
            im_size=None,
            pad_width=config.pad_width,
            patch=None,
            n_nbs=self.eval_n_nbs,
            nbs_mode="argmax",
            tgt_ind=tgt_ind,
            train=False,
        )
        return dset

    def get_eval_sets(self):
        logging.info("Create eval datasets")
        eval_sets = []
        if "tat" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                dset = self.get_eval_set_tat(dset, "all")
                eval_sets.append(dset)
        for dset in self.eval_dsets:
            if dset.startswith("tat-scene-"):
                dset = dset[len("tat-scene-") :]
                dset = self.get_eval_set_tat(dset, "all")
                eval_sets.append(dset)
        if "tat-subseq" in self.eval_dsets:
            for dset in config.tat_eval_sets:
                dset = self.get_eval_set_tat(dset, "subseq")
                eval_sets.append(dset)
        for dset in eval_sets:
            dset.logging_rate = 1
            dset.vis_ind = np.arange(len(dset))
        return eval_sets

    def copy_data(self, data, device, train):
        self.data = {}
        for k, v in data.items():
            v = v.cuda()
            self.data[k] = v.requires_grad_(requires_grad=False)

    def net_forward(self, net, train, iter):
        return net(**self.data)

    def loss_forward(self, output, train, iter):
        errs = {}
        if config.depth_train_only:
            key = 'rgb'
            inter_dm_pred = output["inter_dm_pred"]
            lr_dm_tgt = self.data["inter_dm"].view(*inter_dm_pred.shape)
            inter_dm_pred = inter_dm_pred[..., : lr_dm_tgt.shape[-2], : lr_dm_tgt.shape[-1]]
            dm_loss = self.l1_loss(inter_dm_pred, lr_dm_tgt)
            total_loss = [dm_loss]
            for lidx, loss in enumerate(total_loss):
                errs[key+f"{lidx}"] = loss / config.train_batch_size
        else:
            est = output["out"]
            inter_lr_pred = output["inter_lr_pred"]
            inter_dm_pred = output["inter_dm_pred"]

            tgt = self.data["tgts"].view(*est.shape)
            lr_tgt = self.data["inter_lr"].view(*inter_lr_pred.shape)
            lr_dm_tgt = self.data["inter_dm"].view(*inter_dm_pred.shape)
        
            est = est[..., : tgt.shape[-2], : tgt.shape[-1]]
            inter_lr_pred = inter_lr_pred[..., : lr_tgt.shape[-2], : lr_tgt.shape[-1]]
            inter_dm_pred = inter_dm_pred[..., : lr_dm_tgt.shape[-2], : lr_dm_tgt.shape[-1]]

            if config.use_sr_loss == True:
                sr_result = output["sr_result"]
                sr_result = sr_result.view(est.shape[0],est.shape[1]//2+1,est.shape[2],est.shape[3])
                sr_result = sr_result[..., : tgt.shape[-2], : tgt.shape[-1]]
        
            if config.ycbcr == True:
                key = 'ycbcr'
            else:
                key = 'rgb'
            if train:
                if config.use_sr_loss == True:
                    for lidx, loss in enumerate(self.train_loss(est, tgt, sr_result)):
                        errs[key+f"{lidx}"] = loss / config.train_batch_size
                else:
                    hr_loss = self.train_loss(est, tgt)
                    lr_loss = self.train_loss(inter_lr_pred, lr_tgt)
                    lr_loss = [each * 0.5 for each in lr_loss]
                    dm_loss = [self.l1_loss(inter_dm_pred, lr_dm_tgt) * 0.5]
                    total_loss = hr_loss + lr_loss
                    for lidx, loss in enumerate(total_loss):
                        errs[key+f"{lidx}"] = loss / config.train_batch_size
                    # errs[f"rgb{lidx}"] = loss
            else:
                est = torch.clamp(est, -1, 1)
                est = 255 * (est + 1) / 2
                est = est.type(torch.uint8)
                est = est.type(torch.float32)
                est = (est / 255 * 2) - 1

                errs[key] = self.eval_loss(est, tgt)

            output["out"] = est

        return errs

    def callback_eval_start(self, **kwargs):
        self.metric = None

    def im_to2np(self, im ,ycbcr):
        im = im.detach().to("cpu").numpy()
        if ycbcr == False:
            im = (np.clip(im, -1, 1) + 1) / 2
        else:
            im = np.clip(im, 0, 1)
        im = im.transpose(0, 2, 3, 1)
        return im

    def callback_eval_add(self, **kwargs):
        output = kwargs["output"]
        batch_idx = kwargs["batch_idx"]
        iter = kwargs["iter"]
        eval_set = kwargs["eval_set"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"


        B, N, C, H, W = self.data["tgts"].shape
        size=  self.data["HR_size"]
        ta = self.im_to2np(self.data["tgts"].view(B*N, C, H, W), ycbcr = config.ycbcr)[:, : size[0,0], : size[0,1], :]
        # write debug images
        out_dir = self.exp_out_root / f"{eval_set_name}_n{self.eval_n_nbs}" / f"{iter}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if config.depth_train_only:
            bidx = batch_idx
            lr_dm_pred = output["inter_dm_pred"].cpu().detach().numpy()
            out_LR_dm = lr_dm_pred[0].astype(np.uint8)
            out_LR_dm = cv2.applyColorMap(cv2.convertScaleAbs(out_LR_dm, alpha=15), cv2.COLORMAP_RAINBOW)
    
            if not os.path.exists(str(out_dir / f"{bidx:04d}")):
                os.makedirs(str(out_dir / f"{bidx:04d}"))
            save_path = out_dir / f"{bidx:04d}"/ f"tgt_lr_dm.png"
            cv2.imwrite(str(save_path), out_LR_dm)
    
        else:
            es = self.im_to2np(output["out"], ycbcr = config.ycbcr)[:, : size[0,0], : size[0,1], :]
            lr_es = self.im_to2np(output["inter_lr_pred"], ycbcr=config.ycbcr)[:, : (size[0, 0] // 4), : (size[0, 1] // 4), :]
            lr_dm = output["inter_dm_pred"].cpu().detach().numpy()


            if config.ycbcr == True:
                key = 'ycbcr'
                tgt_lr_rgb = self.im_to2np(self.data["tgt_lr_rgb"], ycbcr = False)
                tgt_fr_rgb = self.im_to2np(self.data["tgt_fr_rgb"], ycbcr= False)
                vec_length = 1
            else:
                key = 'rgb'
                vec_length = 3
            # record metrics
            if self.metric is None:
                self.metric = {}

                self.metric[key] = co.metric.MultipleMetric(
                    metrics=[
                        co.metric.DistanceMetric(p=1, vec_length=vec_length),
                    ]
                )

            self.metric[key].add(es, ta)

            if config.train == False and 'alphas' in output and config.save_alpha == True:
                alphas = torch.squeeze(output["alphas"])
                alphas = alphas.detach().to("cpu").numpy()

            out_LR_im = (255 * lr_es[0]).astype(np.uint8)
            out_LR_dm= lr_dm[0].astype(np.uint8)

            for b in range(ta.shape[0]):
                bidx = batch_idx
                out_im = (255 * es[b]).astype(np.uint8)

                if not os.path.exists(str(out_dir / f"{bidx:04d}")):
                    os.makedirs(str(out_dir / f"{bidx:04d}"))
                PIL.Image.fromarray(out_im).save(out_dir / f"{bidx:04d}"/ f"s{b:04d}_es.png")
                if b == 1:
                    out_LR_dm = cv2.applyColorMap(cv2.convertScaleAbs(out_LR_dm, alpha=5),cv2.COLORMAP_RAINBOW)
                    PIL.Image.fromarray(out_LR_im).save(out_dir / f"{bidx:04d}"/ f"s{b:04d}_es_lr.png")
                    # PIL.Image.fromarray(out_vmm).save(out_dir / f"{bidx:04d}"/ f"s{b:04d}_vmm.png")
                    save_path = out_dir / f"{bidx:04d}"/ f"s{b:04d}_lr_dm.png"
                    cv2.imwrite(str(save_path), out_LR_dm)
                    
    def callback_eval_stop(self, **kwargs):
        eval_set = kwargs["eval_set"]
        iter = kwargs["iter"]
        mean_loss = kwargs["mean_loss"]
        eval_set_name = eval_set.name.replace("/", "_")
        eval_set_name = f"{eval_set_name}_{self.eval_scale}"
        method = self.experiment_name + f"_n{self.eval_n_nbs}"

if __name__ == "__main__":
    parser = co.mytorch.get_parser()
    parser.add_argument("--net", type=str, default=config.net)
    parser.add_argument("--train-dsets", nargs="+", type=str, default=["tat"])
    parser.add_argument(
        "--eval-dsets", nargs="+", type=str, default=["tat"]
    )
    parser.add_argument("--train-n-nbs", type=int, default=config.nbs)
    parser.add_argument("--train-scale", type=float, default=config.train_scale)
    parser.add_argument("--train-patch", type=int, default=config.train_patch)
    parser.add_argument("--eval-n-nbs", type=int, default=config.nbs)
    parser.add_argument("--eval-scale", type=float, default=-1)
    parser.add_argument("--log-debug", type=str, nargs="*", default=[])
    args = parser.parse_args()

    experiment_name = config.experiment_name

    worker = Worker(
        experiments_root=args.experiments_root,
        experiment_name=experiment_name,
        train_dsets=args.train_dsets,
        eval_dsets=args.eval_dsets,
        train_n_nbs=args.train_n_nbs,
        train_scale=args.train_scale,
        train_patch=args.train_patch,
        eval_n_nbs=args.eval_n_nbs,
        eval_scale=args.eval_scale,
    )
    worker.log_debug = args.log_debug
    worker.save_frequency = co.mytorch.Frequency(hours=2)
    worker.eval_frequency = co.mytorch.Frequency(hours=2)
    worker.train_batch_size = config.train_batch_size
    worker.eval_batch_size = config.eval_batch_size
    worker.train_batch_acc_steps = 1

    worker_objects = co.mytorch.WorkerObjects(
        optim_f=lambda net: torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, betas=(config.beta1, config.beta2))
    )

    worker_objects.net_f = lambda: modules.get_net()

    worker.do(args, worker_objects)
