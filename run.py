import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from plot_cm import plot_cm_like_paper
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import DAGF
from trains.singleTask.distillnets import get_distillation_kernel, get_distillation_kernel_homo
from trains.singleTask.misc import softmax
import sys

from datetime import datetime      
now = datetime.now()
format = "%Y/%m/%d %H:%M:%S"
formatted_now = now.strftime(format)
formatted_now = str(formatted_now)+" - "

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('MMSA')

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('MMSA')
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger


def DAGF_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=1, verbose_level=1, mode = '', is_training = False 
):
    # Initialization
    model_name = model_name.upper()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else: # use default config files
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    if model_save_dir == "":
        model_save_dir = Path.home() / "MMSA" / "saved_models"
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    if res_save_dir == "":
        res_save_dir = Path.home() / "MMSA" / "results"
    Path(res_save_dir).mkdir(parents=True, exist_ok=True)
    if log_dir == "":
        log_dir = Path.home() / "MMSA" / "logs"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    

    args = get_config_regression(model_name, dataset_name, config_file)
    args.is_training = is_training  
    args.mode = mode # train or test
    args['model_save_path'] = Path(model_save_dir) / f"{args['model_name']}-{args['dataset_name']}.pth"
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'regression'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    res_save_dir = Path(res_save_dir) / "normal"
    res_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        result = _run(args, num_workers, is_tune)
        model_results.append(result)
    if args.is_training:
        criterions = list(model_results[0].keys())
        # save result to csv
        csv_file = res_save_dir / f"{dataset_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Time"]+["Model"] + criterions)
        # save results
        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))
        
        res = [formatted_now]+res 
        df.loc[len(df)] = res    
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")

def _get_from_dict(data, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _move_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_to_device(v, device) for v in x)
    return x


def _extract_DAGF_batch(batch_data):
    """
    适配常见 MMSA / DAGF 数据格式：
    batch_data['text'], batch_data['audio'], batch_data['vision'], batch_data['labels']
    """
    if isinstance(batch_data, dict):
        text = _get_from_dict(batch_data, ["text", "Text", "language", "words"])
        audio = _get_from_dict(batch_data, ["audio", "Audio", "acoustic"])
        video = _get_from_dict(batch_data, ["vision", "video", "Vision", "Video", "visual"])
        labels = _get_from_dict(batch_data, ["labels", "label", "Label", "target", "y"])

        if isinstance(labels, dict):
            labels = _get_from_dict(labels, ["M", "m", "label", "labels", "regression", "sentiment"])

        if text is None or audio is None or video is None or labels is None:
            raise KeyError(
                "无法从 batch_data 中自动提取 text/audio/video/labels。"
                f"当前 batch_data keys = {list(batch_data.keys())}"
            )

        return text, audio, video, labels

    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) < 4:
            raise ValueError(
                "batch_data 是 list/tuple，但长度小于 4，无法判断 text/audio/video/labels。"
            )

        text = batch_data[0]
        audio = batch_data[1]
        video = batch_data[2]
        labels = batch_data[3]
        return text, audio, video, labels

    raise TypeError(f"不支持的 batch_data 类型：{type(batch_data)}")


def _save_DAGF_confusion_matrix(model, test_loader, args, normalize=False):
    """
    对测试集重新前向推理一次，保存 7 类情感混淆矩阵。
    """
    model.eval()

    device = next(model.parameters()).device

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch_data in test_loader:
            text, audio, video, labels = _extract_DAGF_batch(batch_data)

            text = _move_to_device(text, device)
            audio = _move_to_device(audio, device)
            video = _move_to_device(video, device)
            labels = _move_to_device(labels, device)

            outputs = model(text, audio, video)

            if isinstance(outputs, dict):
                pred = outputs["output_logit"]
            else:
                pred = outputs

            pred = pred.detach().cpu().numpy().reshape(-1)
            true = labels.detach().cpu().numpy().reshape(-1)

            all_pred.append(pred)
            all_true.append(true)

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    save_dir = Path("./result/confusion_matrix")
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{args.dataset_name}_7class_confusion_matrix.png"

    plot_cm_like_paper(
        y_true_score=all_true,
        y_pred_score=all_pred,
        save_path=str(save_path),
        normalize=normalize
    )

    np.save(save_dir / f"{args.dataset_name}_y_true.npy", all_true)
    np.save(save_dir / f"{args.dataset_name}_y_pred.npy", all_pred)

    print(f"[Confusion Matrix] y_true saved to: {save_dir / f'{args.dataset_name}_y_true.npy'}")
    print(f"[Confusion Matrix] y_pred saved to: {save_dir / f'{args.dataset_name}_y_pred.npy'}")

def _run(args, num_workers=4, is_tune=False, from_sena=False): 

    dataloader = MMDataLoader(args, num_workers)

    if args.is_training:
        print("training for DAGF")

        
        args.gd_size_low = 64  
        args.w_losses_low = [1, 10]  
        args.metric_low = 'l1'  

        
        args.gd_size_high = 32  
        args.w_losses_high = [1, 10]  
        args.metric_high = 'l1'  

        to_idx = [0, 1, 2]  
        from_idx = [0, 1, 2]  
        assert len(from_idx) >= 1

        model = []
        model_DAGF = getattr(DAGF, 'DAGF')(args)

        model_distill_homo = getattr(get_distillation_kernel_homo, 'DistillationKernel')(n_classes=1,
                                                                               hidden_size=
                                                                               args.dst_feature_dim_nheads[0],
                                                                               gd_size=args.gd_size_low,
                                                                               to_idx=to_idx, from_idx=from_idx,
                                                                               gd_prior=softmax([0, 0, 1, 0, 1, 0], 0.25),
                                                                               gd_reg=10,
                                                                               w_losses=args.w_losses_low,
                                                                               metric=args.metric_low,
                                                                               alpha=1 / 8,
                                                                               hyp_params=args)

        model_distill_hetero = getattr(get_distillation_kernel, 'DistillationKernel')(n_classes=1,
                                                                                   hidden_size=
                                                                                   args.dst_feature_dim_nheads[0] * 2,
                                                                                   gd_size=args.gd_size_high,
                                                                                   to_idx=to_idx, from_idx=from_idx,
                                                                                   gd_prior=softmax([0, 0, 1, 0, 1, 1], 0.25),
                                                                                   gd_reg=10,
                                                                                   w_losses=args.w_losses_high,
                                                                                   metric=args.metric_high,
                                                                                   alpha=1 / 8,
                                                                                   hyp_params=args)

        model_DAGF = model_DAGF.cuda()

        model = [model_DAGF]         
    else:
        print("testing phase for DAGF")
        model = getattr(DAGF, 'DAGF')(args)
        model = model.cuda()

    trainer = ATIO().getTrain(args)


    #test
    if args.mode == 'test':
        model.load_state_dict(torch.load('./pt/DAGF' + str(args.dataset_name) + '.pth'), strict=False)
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

        _save_DAGF_confusion_matrix(
            model=model,
            test_loader=dataloader['test'],
            args=args,
            normalize=True
        )

        sys.stdout.flush()
        input('[Press Any Key to start another run]')
    #train
    else:
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)
        model[0].load_state_dict(torch.load('./pt/DAGF'+str(args.dataset_name)+'.pth'))

        results = trainer.do_test(model[0], dataloader['test'], mode="TEST")

        _save_DAGF_confusion_matrix(
            model=model[0],
            test_loader=dataloader['test'],
            args=args,
            normalize=True
        )

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results