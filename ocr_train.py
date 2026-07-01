from model import *

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)                                         # (B, 1, 32, 128)
    target_lengths = torch.tensor([len(l) for l in labels])             # (B,)
    targets = pad_sequence(labels, batch_first=True, padding_value=0)   # (B, max_len)
    return images, targets, target_lengths

def greedy_decode(logits, idx2char):
    pred_ids = logits.argmax(dim=-1).squeeze(0)  # (32,) — argmax over vocab at each time step

    result = []
    prev = None
    for idx in pred_ids:
        idx = idx.item()
        if idx != 0 and idx != prev:  # skip blank (0) and repeated tokens
            result.append(idx2char[idx])
        prev = idx

    return ''.join(result)

def epoch_batches(images, labels, lengths, batch_size=32,shuffle=True):
    if shuffle:
        idx = torch.randperm(len(images))
    else:
        idx = torch.arange(len(images))
    for i in range(0, len(images), batch_size):
        batch_idx = idx[i:i+batch_size]
        yield images[batch_idx], labels[batch_idx], lengths[batch_idx]

def main():    
    ocr = OCRModel().to(device)
    #ocr.load_state_dict(torch.load("retrain_oxf/oxf_trplate_00010.pth")) # NO NEED TO JUST YET.
    scaler = GradScaler(device)
    #ocr = torch.compile(ocr)
    optimizer = torch.optim.AdamW(ocr.parameters(), lr=3e-4)
    train_cache = torch.load('realplate_train_64x128.pt') # I have to change the dataset cache when I fine tune the model. loading all to mem is not ideal
    val_cache = torch.load("realplate_val_64x128.pt")

    tr_images  = train_cache['images']    # already in float32, no transform needed
    tr_labels  = train_cache['labels']
    tr_lengths = train_cache['lengths']

    val_images  = val_cache['images']    # already in float32, no transform needed
    val_labels  = val_cache['labels']
    val_lengths = val_cache['lengths']

    # move entire dataset to GPU if it fits
    tr_images  = tr_images.to(device)
    tr_labels  = tr_labels.to(device)
    tr_lengths = tr_lengths.to(device)

    val_images  = val_images.to(device)
    val_labels  = val_labels.to(device)
    val_lengths = val_lengths.to(device)

    tr_num_batches = len(tr_images) // 32 # batch_size
    val_num_batches = len(val_images) // 32
    results = {"epoch":[],"train_loss" : [],"val_loss":[]}
    for epoch in tqdm(range(50)):
        ocr.train()
        total_train_loss = 0
        total_val_loss = 0
        for tr_img_batch, tr_target_batch, tr_target_lengths in tqdm(epoch_batches(tr_images, tr_labels, tr_lengths, batch_size=32, shuffle=True),total=tr_num_batches):
            with autocast(device_type=device):
                train_logits, train_loss = ocr(tr_img_batch, tr_target_batch, tr_target_lengths)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += train_loss.item()
        
        ocr.eval()
        for val_img_batch, val_target_batch, val_target_lengths in tqdm(epoch_batches(val_images, val_labels, val_lengths, batch_size=32, shuffle=False),total=val_num_batches):
            with torch.no_grad():
                with autocast(device_type=device):
                    val_logits, val_loss = ocr(val_img_batch, val_target_batch, val_target_lengths)
            total_val_loss += val_loss.item()
        
        avg_train_loss = total_train_loss / tr_num_batches
        avg_val_loss = total_val_loss / val_num_batches

        if epoch % 5 == 0:
            save_path = f"retrain_oxf/realtrplate_2k_{epoch:0005d}.pth"
            torch.save(ocr.state_dict(), save_path)
        print(f"Epoch: {epoch}, Train Loss : {avg_train_loss:.4f}, Val Loss : {avg_val_loss:.4f}")
        results["epoch"].append(epoch)
        results["train_loss"].append(avg_train_loss)
        results["val_loss"].append(avg_val_loss)
      
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"retrain_oxf/realplate_results_2k.csv")
    # I want to save after each iteration.
if __name__ == "__main__":
    main()
