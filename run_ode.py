import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from loguru import logger
from args import args
from model import ODEGCN
from utils import generate_dataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_rmse_np, masked_mape_np
import torch.nn.functional as F



args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_weights(h: int, lam: float = 0.2, device=None):
    """
    Create exponentially decaying weights for horizons 1..h:
      w_t = exp(-lam*(t-1)), then normalized to sum 1.
    Return shape (h,).
    """
    idx = torch.arange(h, dtype=torch.float32, device=device)
    w = torch.exp(-lam * idx)  # t-1 -> idx 0..h-1
    w = w / (w.sum() + 1e-12)
    return w  # (h,)

def train(loader, net, optimizer, device, horizon, loss_lambda,
          alpha: float = 0.7, smoothl1_beta: float = 1.0):
    net.train()
    total_loss = 0.0
    n_batches = 0
    weights = make_weights(horizon, lam=loss_lambda, device=device)  # (h,)

    for inputs, targets in tqdm(loader, desc=f"Train (h={horizon})", dynamic_ncols=True):
        inputs  = inputs.to(device)      # (B,N,T_in,F)
        targets = targets.to(device)     # (B,N,T_out)

        optimizer.zero_grad()
        outputs = net(inputs)            # (B,N,T_out)
        pred = outputs[:, :, :horizon]   # (B,N,h)
        true = targets[:, :, :horizon]   # (B,N,h)


        abs_err = torch.abs(pred - true)                       
        mae_w   = (abs_err * weights.view(1,1,-1)).mean()


        smooth = F.smooth_l1_loss(pred, true, beta=smoothl1_beta, reduction='none')  
        smooth_w = (smooth * weights.view(1,1,-1)).mean()

        loss = alpha * mae_w + (1.0 - alpha) * smooth_w

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)

@torch.no_grad()
def eval(loader, model, std, mean, device, horizon):
    """Evaluation function: returns (rmse, mae, mape) for the first horizon steps"""
    batch_rmse_loss = 0.0
    batch_mae_loss = 0.0
    batch_mape_loss = 0.0
    n_batches = 0

    model.eval()
    for idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Eval (h={horizon})", dynamic_ncols=True)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)                      # (B,N,T_out)

        pred = output[:, :, :horizon].detach().cpu().numpy() * std + mean   # unnorm
        true = targets[:, :, :horizon].detach().cpu().numpy() * std + mean

        # Masked metrics include their own masking logic; supply mask_value 0 as before
        mae_loss = masked_mae_np(true, pred, 0)
        rmse_loss = masked_rmse_np(true, pred, 0)
        mape_loss = masked_mape_np(true, pred, 0)

        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss
        n_batches += 1

    if n_batches == 0:
        return float('nan'), float('nan'), float('nan')
    return batch_rmse_loss / n_batches, batch_mae_loss / n_batches, batch_mape_loss / n_batches

def main(args):
    """Main function (Curriculum Learning)"""
    # 1. Data Loading
    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    logger.info(f"Dataset: {args.filename}")
    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Total time steps: {data.shape[0]}")



    # 2. Normalize adjacency matrices
    A_sp_wave = get_normalized_adj(sp_matrix).to(args.device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(args.device)

    # 3. Data loaders
    train_loader, valid_loader, test_loader = generate_dataset(data, args)

    # 4. Model Initialization
    from model_improved import ODEGCN_Improved
    net = ODEGCN_Improved(
        num_nodes=data.shape[1],
        num_features=data.shape[2],
        num_timesteps_input=args.his_length,
        num_timesteps_output=args.pred_length,
        A_sp_hat=A_sp_wave,
        A_se_hat=A_se_wave,
        hidden_channels=64,  
        num_stacks=2,  
        odeg_steps=12,  
        odeg_solver='rk4',
        dropout=0.1
    ).to(args.device)

    # 5. Optimizer / Loss / Scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    best_valid_rmse = float('inf')
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Curriculum schedule
    horizons = [min(3, args.pred_length), min(6, args.pred_length), min(9, args.pred_length), args.pred_length]

    # Epochs per stage
    epochs_per_stage = max(1, args.epochs // len(horizons))
    loss_lambda = getattr(args, 'loss_lambda', 0.2)  # Weight decay param for loss weights

    # 6. Curriculum Training Loop
    global_epoch = 0
    log_file = f"train_log_{args.filename}.txt"
    with open(log_file, "w") as f:
        f.write(
            "Stage\tEpoch\tHorizon\tTrain_loss\tTrain_RMSE\tTrain_MAE\tTrain_MAPE\tValid_RMSE\tValid_MAE\tValid_MAPE\n")

    for stage_idx, h in enumerate(horizons):
        logger.info(f"=== Curriculum Stage {stage_idx + 1}/{len(horizons)}: horizon = {h} ===")
        for epoch in range(1, epochs_per_stage + 1):
            global_epoch += 1
            logger.info(f"----- Stage {stage_idx + 1} Epoch {epoch} (global epoch {global_epoch}) -----")

            train_loss = train(train_loader, net, optimizer, args.device, horizon=h, loss_lambda=loss_lambda)
            logger.info(f"Train loss (h={h}): {train_loss:.6f}")

            # Evaluate on train and valid for current horizon
            train_rmse, train_mae, train_mape = eval(train_loader, net, std, mean, args.device, horizon=h)
            valid_rmse, valid_mae, valid_mape = eval(valid_loader, net, std, mean, args.device, horizon=h)

            logger.info(f"Train (h={h}) RMSE {train_rmse:.6f}, MAE {train_mae:.6f}, MAPE {train_mape:.6f}")
            logger.info(f"Valid (h={h}) RMSE {valid_rmse:.6f}, MAE {valid_mae:.6f}, MAPE {valid_mape:.6f}")


            with open(log_file, "a") as f:
                f.write(f"{stage_idx + 1}\t{epoch}\t{h}\t"
                        f"{train_loss:.6f}\t"
                        f"{train_rmse:.6f}\t{train_mae:.6f}\t{train_mape:.6f}\t"
                        f"{valid_rmse:.6f}\t{valid_mae:.6f}\t{valid_mape:.6f}\n")

            # Save best by validation RMSE (global across stages)
            if valid_rmse < best_valid_rmse:
                best_valid_rmse = valid_rmse
                logger.info(f"New best valid RMSE: {best_valid_rmse:.6f} — saving model.")
                torch.save(net.state_dict(), f'net_params_{args.filename}_{args.num_gpu}.pkl')

            # Step scheduler per epoch
            scheduler.step()

    net.load_state_dict(torch.load(f'net_params_{args.filename}_{args.num_gpu}.pkl'))
    test_rmse, test_mae, test_mape = eval(test_loader, net, std, mean, args.device)


if __name__ == '__main__':
    main(args)


