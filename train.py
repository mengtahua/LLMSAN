import os
# ===============================
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
from model import LLMSAN

# ===============================
# dataset & collate
# ===============================
class GRUDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            item["input_seq"],
            item["mean_embedding"],
            item["target"],
            item["prior_target"],
            item["article_ids"],
            item["pred_target"],
            item["pred_time"]
        )

def gru_collate_fn(batch):
    input_seq = torch.stack([b[0] for b in batch]).float()
    mean_embedding = torch.stack([b[1] for b in batch]).float()
    target = torch.stack([b[2] for b in batch]).float()
    prior_target = torch.stack([b[3] for b in batch]).float()
    article_ids = [b[4] for b in batch]
    pred_target = [b[5] for b in batch]
    pred_time = [b[6] for b in batch]
    return input_seq, mean_embedding, target, prior_target, article_ids, pred_target, pred_time

# ===============================

def extract_input_dims(data):
    sample = data[0]
    return (
        sample["input_seq"].shape[-1],
        sample["mean_embedding"].shape[-1],
        sample["prior_target"].shape[-1],
        sample["target"].shape[-1],
    )

# ===============================

def save_results_to_xlsx(model_name, input_ids, target_ids, target_times,
                         predicted_ids, prediction_errors,
                         save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    records = []

    for i in range(len(target_ids)):
        row = {f"time{t+1}_id": input_ids[i][t] if t < len(input_ids[i]) else None
               for t in range(12)}
        row["target_id"] = target_ids[i]
        row["target_post_time"] = (
            target_times[i].strftime("%Y-%m-%d %H:%M:%S")
            if not isinstance(target_times[i], str)
            else target_times[i]
        )
        row["predicted_id"] = predicted_ids[i]
        row["prediction_error"] = prediction_errors[i]

        records.append(row)

    df = pd.DataFrame(records)
    save_path = os.path.join(save_dir, model_name + ".xlsx")
    df.to_excel(save_path, index=False)
    print(f"[INFO] Results saved to {save_path}")

# ===============================
def train_and_eval(pt_name, config):
    
    ablation_type=config["ablation_type"]
    batch_size=config["batch_size"]
    hidden_dim = config["hidden_dim"]
    learning_rate=config["learning_rate"]
    epoch=config['epoch']
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    data = torch.load(os.path.join("Dataset", pt_name))
    seq_dim, mean_dim, prior_dim, out_dim = extract_input_dims(data)

    dataset = GRUDataset(data)

    g = torch.Generator()
    g.manual_seed(2025)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gru_collate_fn,
        generator=g
    )

    model = LLMSAN(seq_dim, mean_dim, prior_dim, hidden_dim, out_dim, ablation_type).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # =================
    # 训练
    # =================
    model.train()
    for epoch in range(epoch):
        total_loss = 0.0
        for input_seq, mean_embedding, target, prior_target, *_ in loader:
            input_seq = input_seq.to(device)
            mean_embedding = mean_embedding.to(device)
            target = target.to(device)
            prior_target = prior_target.to(device)

            pred, _, _ = model(input_seq, mean_embedding, prior_target)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * input_seq.size(0)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataset):.6f}")

    # =================
    # 推理
    # =================
    model.eval()
    input_ids_all, target_ids_all, target_times_all = [], [], []
    predicted_ids_all, errors_all = [], []

    with torch.no_grad():
        for batch in loader:
            input_seq, mean_embedding, target, prior_target, article_ids, pred_ids, pred_times = batch
            input_seq = input_seq.to(device)
            mean_embedding = mean_embedding.to(device)
            target = target.to(device)
            prior_target = prior_target.to(device)

            pred, _, _ = model(input_seq, mean_embedding, prior_target)
            errors = ((pred - target) ** 2).mean(dim=1).cpu().numpy()

            for i in range(len(pred_ids)):
                input_ids_all.append(article_ids[i])
                target_ids_all.append(pred_ids[i])
                target_times_all.append(pred_times[i])
                predicted_ids_all.append(f"pred_{pred_ids[i]}")
                errors_all.append(errors[i])
                

    save_results_to_xlsx(
        f"Model_{ablation_type}",
        input_ids_all,
        target_ids_all,
        target_times_all,
        predicted_ids_all,
        errors_all,
        save_dir=os.path.join("result", pt_name[:-3])
    )
