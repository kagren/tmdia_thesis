from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import torch
import lightning as L
from copy import deepcopy
import os
import math
import torch.nn.functional as F
from lightly.loss import NegativeCosineSimilarity
import torch.nn as nn
from time import time
import torch.distributed as dist

# We want to be able to run from command line as well
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

@torch.no_grad()
def extract_model_features(model, dataset, batch_size, device, progressbar = False, upsample = True):
    
    network = model
    
    def show_progress_if_needed(it):
        
        if progressbar:
            return tqdm(it)
        else:
            return it
    
    try:
    
        network.eval()
        network.to(device)
    
        if upsample:
            # Upsample so that we have 50/50
            labels_unique, labels_counts = np.unique(dataset.df.cancer, return_counts=True)
            class_weights = [sum(labels_counts) / c for c in labels_counts]
            sample_weights = [class_weights[e] for e in dataset.df.cancer]
            sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(dataset))
        else:
            sampler = None

        # Encode all images
        data_loader = torch.utils.data.DataLoader(dataset, 
                                                  batch_size = batch_size, 
                                                  num_workers = 4, 
                                                  shuffle = False, 
                                                  drop_last = False, 
                                                  sampler = sampler,
                                                  pin_memory = True)
        feats, labels = [], []

        torch.manual_seed(42)

        for batch_imgs, batch_labels in show_progress_if_needed(data_loader):

            with autocast():
                batch_imgs = batch_imgs.to(device)
                batch_feats = network(batch_imgs)

            feats.append(batch_feats.detach().cpu().float())
            labels.append(batch_labels.detach().cpu().long())

        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)

        # Sort images by labels
        labels, idxs = labels.sort()
        feats = feats[idxs]
    finally:
        pass

    return torch.utils.data.TensorDataset(feats, labels)



class LogisticRegression(L.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

CHECKPOINT_PATH = "."

def train_logreg(batch_size, train_feats_data, test_feats_data, max_epochs=100, **kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
        accelerator="gpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        enable_progress_bar=False,
        check_val_every_n_epoch=10,
    )
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_feats_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4
    )

    L.seed_everything(42)  # To be reproducable
    model = LogisticRegression(**kwargs)

    trainer.fit(model, train_loader, test_loader)

    model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    del trainer

    return result["test"]

def train_knn(train_feats_data, test_feats_data, k = 20):

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    # DataLoader to handle the TensorDatasets
    train_loader = DataLoader(train_feats_data, batch_size=len(train_feats_data))
    test_loader = DataLoader(test_feats_data, batch_size=len(test_feats_data))

    # Function to convert data
    def extract_features_and_labels(loader):
        for data in loader:
            features, labels = data
            return features.numpy(), labels.numpy()

    # Extract features and labels from TensorDatasets
    X_train, y_train = extract_features_and_labels(train_loader)
    X_test, y_test = extract_features_and_labels(test_loader)

    knn = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn.fit(X_train, y_train)
    test_probabilities = knn.predict_proba(X_test)[:, 1]  # probabilities for the positive class
    auc_score = roc_auc_score(y_test, test_probabilities)

    return auc_score

# Calculate uniformity loss (Example using a simple dispersion measure)
def calculate_uniformity_loss(embeddings):
    # Simple regularization to maximize distance between embeddings
    # Negative of average cosine similarity
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    batch_size = embeddings.size(0)
    mask = ~torch.eye(batch_size, dtype=bool, device=embeddings.device)
    similarity_matrix = similarity_matrix.masked_select(mask).view(batch_size, -1)
    mean_similarity = similarity_matrix.mean()
    
    return -mean_similarity  # Maximizing angular distance

def add_jitter(embeddings, epsilon=1e-8):
    noise = torch.randn_like(embeddings) * epsilon
    return embeddings + noise

def normalize_embeddings(embeddings):
    norms = embeddings.norm(p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings / norms
    return normalized_embeddings

def calculate_angular_distances(embeddings):

    embeddings = add_jitter(embeddings)

    # Normalize embeddings to lie on the unit sphere
    embeddings = normalize_embeddings(embeddings)

    # Calculate pairwise cosine similarity
    # torch.cosine_similarity expects inputs of shape (1, D) and (N, D) to compute pairwise distances
    cosine_similarities = torch.cosine_similarity(embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2)
    
    # Ensure numerical stability before arc cosine
    cosine_similarities = torch.clamp(cosine_similarities, -1, 1)
    
    # Convert cosine similarities to angular distances
    angular_distances = torch.acos(cosine_similarities)

    return angular_distances


def calculate_histogram_loss(embeddings, num_bins=50):
    angular_distances = calculate_angular_distances(embeddings)

    # Remove self-comparisons by setting diagonal to NaN and then masking them out
    torch.diagonal(angular_distances)[:] = float('nan')
    valid_distances = angular_distances[~torch.isnan(angular_distances)]

    # Create histogram of valid distances
    hist = torch.histc(valid_distances, bins=num_bins, min=0, max=np.pi)

    # Define uniform target distribution
    expected_count = torch.full((num_bins,), hist.sum().item() / num_bins, device=embeddings.device)
    kl_divergence = torch.nn.functional.kl_div(torch.log(hist.clamp(min=1e-5)), expected_count.log(), reduction='batchmean')

    return kl_divergence

def matrix_sqrt_batch(sym_mat, eps=1e-6):
    """ Compute the square root of a batch of symmetric matrices with regularization. """
    batch_size = sym_mat.shape[0]
    sqrtmats = []
    for i in range(batch_size):
        s, u = torch.linalg.eigh(sym_mat[i] + eps * torch.eye(sym_mat.size(-1), device=sym_mat.device))
        s_sqrt = torch.sqrt(torch.clamp(s, min=eps))
        sqrtmats.append(u @ torch.diag(s_sqrt) @ u.T)
    return torch.stack(sqrtmats)

def wasserstein_uniformity_loss(embeddings, eps=1e-6):
    #print("embeddings.shape", embeddings.shape)
    
    n, m = embeddings.shape
    normalized_embeddings = normalize_embeddings(embeddings)
    
    # Calculate mean and covariance matrix for each batch
    mu = normalized_embeddings.mean(dim=0)

    #print("mu.shape", mu.shape)

    centered_embeddings = normalized_embeddings - mu
    sigma = centered_embeddings.T @ centered_embeddings / n

    # Compute the 2-Wasserstein distance for each batch
    term_mu = torch.sum(mu ** 2)
    term_trace1 = torch.trace(sigma)
    term_trace2 = -(2. / math.sqrt(m)) * torch.trace(torch.sqrt(sigma))

    distance = torch.sqrt(term_mu + 1 + term_trace1 + term_trace2)

    #print("term_mu", term_mu)
    #print("term_trace1", term_trace1)
    #print("term_trace2", term_trace2)    

    return -distance


"""
def calculate_histogram_loss(embeddings, num_bins=50, min_angle=0, max_angle=np.pi):
    # Calculate angular distances
    cosine_similarity = torch.matmul(embeddings, embeddings.T)
    angular_distances = torch.acos(torch.clamp(cosine_similarity, -1, 1))
    
    # Flatten the angular distances and remove self-comparisons
    batch_size = embeddings.size(0)
    #print("bs", batch_size)

    upper_tri_mask = torch.triu(torch.ones(batch_size, batch_size, device=embeddings.device), diagonal=1) == 1
    angular_distances = angular_distances[upper_tri_mask]
    
    # Create histogram of distances
    hist = torch.histc(angular_distances, bins=num_bins, min=min_angle, max=max_angle)
    
    # Calculate the KL divergence against a uniform distribution
    expected_count = len(angular_distances) / num_bins
    uniform_distribution = torch.full((num_bins,), expected_count, device=embeddings.device)

    print("hist", hist)
    print("uniform_distribution", uniform_distribution)
    kl_divergence = torch.nn.functional.kl_div(torch.log(hist + 1), uniform_distribution, reduction='sum')

    #print("kl", kl_divergence)
    
    return kl_divergence
"""

def train_model(logger, model, ds_train, eval_train_img_data, eval_test_img_data,
                batch_size, device, EPOCHS, do_save_model=True, mixup=False, sampler = None, 
                shuffle = True, model_checkpoint_filename = None, uniformity = False):
    
    num_workers = 6 if in_notebook() else 12
    
    torch.manual_seed(42)
        
    print("Device", device)
    print("Setting up loader ...")
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                           pin_memory=True, pin_memory_device=device, prefetch_factor=4, sampler=sampler)

    print("Setting up loader ... DONE")
    # SimSiam uses a symmetric negative cosine similarity loss
    criterion = NegativeCosineSimilarity()
    criterion_mse = nn.MSELoss()

    # scale the learning rate
    lr = 0.05 * batch_size / 256
    # use SGD with momentum and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


    #optimizer = torch.optim.AdamW(model.parameters(), 
    #                          lr=lr / 100, 
    #                          weight_decay=ADAMW_WEIGHT_DECAY)
    #optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
    #                                                       T_max=EPOCHS * batch_size * 3, 
    #                                                       eta_min=LEARNING_RATE / 100)
    scheduler = None
    
    scaler = GradScaler()
    best_eval_score = 0

    avg_loss = 0.0
    avg_output_std = 0.0
    
    avg_wasserstein_distance = 0.0

    if uniformity:
        uniform_loss = wasserstein_uniformity_loss #calculate_histogram_loss
    else:
        uniform_loss = None
    
    for epoch in tqdm(range(EPOCHS), desc='Epoch'):

        print("Epoch", epoch, "sampler", sampler)

        if sampler:
            sampler.set_epoch(epoch)

        model.train()
            
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, bs in enumerate(dl_train):

            loss_uniformity = None

            #
            # TODO: make sure that we do not create a mixup with the same image in the same batch
            #

            if mixup:
                (x0, x0hat), (x1, x1hat), y_cancer = bs

                # We now have two distinct images, each with two variations, create a third image
                # which is the mixup image
                alpha = np.random.beta(4, 4)

                if np.random.random() < 0.5:
                    alpha = 0

                xmixup = alpha * x0 + (1-alpha) * x1 # x0.shape = (batch_size, 1, m, n)
                
                # Using mixed precision training
                with autocast():
                    xmixup = xmixup.to(device)
                    x0hat = x0hat.to(device)
                    x1hat = x1hat.to(device)
                    
                    z0, p0, backbone0 = model(x0hat)
                    z1, p1, backbone1 = model(x1hat)
                    zmixup, pmixup, backbone2 = model(xmixup)

                    # apply the symmetric negative cosine similarity
                    # and run backpropagation
                    loss_blend = alpha
                    loss = loss_blend * criterion(z0, pmixup) + (1-loss_blend) * criterion(z1, pmixup) + \
                        loss_blend * criterion(zmixup, p0) + (1-loss_blend) * criterion(zmixup, p1)

                    loss = loss * 0.5

                    if uniformity:
                        loss_uniformity = uniform_loss(torch.concat((p0, p1, pmixup)))
                        loss = 0.9 * loss + 0.1 * loss_uniformity
                    
            else:

                (x0, x1), y_cancer = bs
                
                # Using mixed precision training
                with autocast():
                    x0 = x0.to(device)
                    x1 = x1.to(device)

                    # run the model on both transforms of the images
                    # we get projections (z0 and z1) and
                    # predictions (p0 and p1) as output
                    z0, p0, backbone0 = model(x0)
                    z1, p1, backbone1 = model(x1)

                    # apply the symmetric negative cosine similarity
                    # and run backpropagation
                    loss = criterion(z0, p1) + criterion(z1, p0)

                    if uniformity:
                        loss_uniformity = uniform_loss(torch.concat((p0, p1)))
                        loss = 0.9 * loss + 0.1 * loss_uniformity
                
            # scaler is needed to prevent "gradient underflow"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # calculate the per-dimension standard deviation of the outputs
            # we can use this later to check whether the embeddings are collapsing
            output = p0.detach()
            output = torch.nn.functional.normalize(output, dim=1)

            output_std = torch.std(output, 0)
            output_std = output_std.mean()

            # use moving averages to track the loss and standard deviation
            w = 0.9
            avg_loss = w * avg_loss + (1 - w) * loss.item()
            avg_output_std = w * avg_output_std + (1 - w) * output_std.item()

            avg_wasserstein_distance = w * avg_wasserstein_distance + (1-w) * wasserstein_uniformity_loss(output).item()
            
            # Logging ranking metrics
            mode = "train"

            if logger:
                log_dict = {}

                log_dict[mode + "_loss"] = loss.item()
                log_dict[mode + '_lr'] = lr if scheduler is None else scheduler.get_last_lr()[0]

                if uniformity:
                    log_dict[mode + "_loss_uniformity"] = loss_uniformity.item()
                    
                logger.log(log_dict)

            #logger.log({'loss': (loss.item()),
            #            'cancer_loss': cancer_loss.item(),
            #            'aux_loss': aux_loss.item(),
            #            'lr': scheduler.get_last_lr()[0],
            #            'epoch': epoch})
             
        # the level of collapse is large if the standard deviation of the l2
        # normalized output is much smaller than 1 / sqrt(dim)
        collapse_level = max(0.0, 1 - math.sqrt(model.module.out_dim) * avg_output_std)
        
        if logger:
            logger.log({mode + "_collapse" : collapse_level})
            logger.log({mode + "_wassterstein" : avg_wasserstein_distance})

        # print intermediate results
        print(
            f"[Epoch {epoch:3d}] "
            f"Loss = {avg_loss:.4f} | "
            f"Collapse Level: {collapse_level:.2f} / 1.00 |"
            f"Wasserstein distance: {avg_wasserstein_distance:.2f}"
        )
        
        if epoch % 2 == 0 and device == "cuda:0":
        

            print("Evaluating on test set ...")
            time_start = time()
            model.eval()

            with torch.no_grad():
                model_eval = nn.Sequential(model.module.backbone, nn.BatchNorm1d(model.module.out_dim * 2))

                print("Evaluating on test set ... Extracting features 1/2")
                np.random.seed(42)
                train_feats_simclr = extract_model_features(model_eval, eval_train_img_data, batch_size=batch_size*2, device=device)

                print("Evaluating on test set ... Extracting features 2/2")
                test_feats_simclr = extract_model_features(model_eval, eval_test_img_data, batch_size=batch_size*2, device=device)

            print("Evaluating on test set ... Training KNN")

            eval_result = train_knn(train_feats_simclr, test_feats_simclr)

            print("Evaluating on test set ... DONE, AUC", eval_result, f"in {time() - time_start} seconds")

            #print(logreg_results)
            if eval_result > best_eval_score:
                best_eval_score = eval_result

            if logger:
                logger.log({"test_knn_auc" : eval_result})
                logger.log({"test_knn_auc_max" : best_eval_score})

        if epoch % 10 == 0 and device == "cuda:0" and model_checkpoint_filename:
            fn = f"{model_checkpoint_filename}_epoch_{epoch}.pt"

            print("Saving model to", fn)
            torch.save(model.state_dict(), fn)

        if dist.is_initialized():
            print("DDP is running, wait for barrier")
            dist.barrier()

    return model

def infoNCE_loss(preds, temperature, mode="train"):
    
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(preds[:, None, :], preds[None, :, :], dim=-1)
    
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
    
    # InfoNCE loss
    cos_sim = cos_sim / temperature
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()
    
    # Get ranking position of positive example
    comb_sim = torch.cat(
        [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
        dim=-1,
    )
    sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
    
    return nll, sim_argsort


def train_model_simclr(logger, model, ds_train, eval_train_img_data, eval_test_img_data,
                batch_size, device, EPOCHS, do_save_model=True, mixup=False, sampler = None, 
                shuffle = True, model_checkpoint_filename = None, uniformity = False):
    
    num_workers = 6 if in_notebook() else 8
    
    torch.manual_seed(42)
        
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, 
                                           pin_memory=True, pin_memory_device=device, prefetch_factor=4, sampler = sampler)

    #optimizer = torch.optim.AdamW(model.parameters(), 
    #                          lr=LEARNING_RATE, 
    #                          weight_decay=ADAMW_WEIGHT_DECAY)
    # scale the learning rate
    lr = 0.05 * batch_size / 256
    # use SGD with momentum and weight decay
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            
    scaler = GradScaler()
    best_eval_score = 0
    avg_wasserstein_distance = 0
    avg_loss = 0
    scheduler = None
    
    for epoch in tqdm(range(EPOCHS), desc='Epoch'):

        model.train()
        with tqdm(dl_train, desc='Train', mininterval=60, maxinterval=120) as train_progress:
            
            optimizer.zero_grad(set_to_none=True)
            
            for bs in train_progress:

                if mixup:
                    (x0, x0hat), (x1, x1hat), _ = bs

                    # We now have two distinct images, each with two variations, create a third image
                    # which is the mixup image
                    alpha = np.random.beta(2, 2)

                    if np.random.random() < 0.25:
                        alpha = 0

                    xmixup = alpha * x0 + (1-alpha) * x1 # x0.shape = (batch_size, 1, m, n)

                    # The batch should contain three parts of images, the first part, which is
                    # the mixup image, the second part and third of the batch contain the 
                    # augmented variations of the source images
                    X = torch.cat((xmixup, x0hat, x1hat))
                    #X = torch.cat((x0, x0hat))

                else:
                    (x0, x1), _ = bs

                    # The batch should contain two parts of images representing the two alternatives 
                    # of each image with different augmentation
                    X = torch.cat((x0, x1))
                    

                # Using mixed precision training
                with autocast():

                    X = X.to(device, non_blocking=True)
                    
                    preds = model(X)
                    
                    if mixup:
                        slice = x0.shape[0]

                        preds_mixup_to_x0 = preds[:2*slice]
                        preds_mixup_to_x1 = torch.cat((preds[:slice], preds[2*slice:]))

                        #print(slice, preds.shape, preds_mixup_to_x0.shape, preds_mixup_to_x1.shape)

                        loss1, sim_argsort1 = infoNCE_loss(preds = preds_mixup_to_x0, temperature = 0.2)   
                        loss2, sim_argsort2 = infoNCE_loss(preds = preds_mixup_to_x1, temperature = 0.2) 

                        loss = alpha * loss1 + (1-alpha) * loss2
                        sim_argsort = torch.cat((sim_argsort1, sim_argsort2))

                    else:
                        #print(preds.shape)
                        loss, sim_argsort = infoNCE_loss(preds = preds, temperature = 0.2)                                                           

                # scaler is needed to prevent "gradient underflow"
                scaler.scale(loss).backward()
                scaler.step(optimizer)

                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                
                # use moving averages to track the loss and standard deviation
                w = 0.9
                avg_loss = w * avg_loss + (1 - w) * loss.item()
                avg_wasserstein_distance = w * avg_wasserstein_distance + (1-w) * wasserstein_uniformity_loss(preds).item()
                
                # Logging ranking metrics
                mode = "train"

                if logger:
                    log_dict = {}

                    log_dict[mode + "_loss"] = loss.item()
                    log_dict[mode + '_lr'] = lr if scheduler is None else scheduler.get_last_lr()[0]
                    log_dict[mode + "_acc_top1"] = (sim_argsort == 0).float().mean()
                    log_dict[mode + "_acc_top5"] = (sim_argsort < 5).float().mean()
                    log_dict[mode + "_acc_mean_pos"] = 1 + sim_argsort.float().mean()
                    log_dict[mode + "_wassterstein"] =  avg_wasserstein_distance

                    logger.log(log_dict)

        if epoch % 4 == 0 and device == "cuda:0":

            print("Evaluating on test set ...")
            time_start = time()
            model.eval()

            with torch.no_grad():
                model_eval = nn.Sequential(model.module.backbone, nn.BatchNorm1d(model.module.backbone_dim)) #model.module.out_dim * 2))

                print("Evaluating on test set ... Extracting features 1/2")
                np.random.seed(42)
                train_feats_simclr = extract_model_features(model_eval, eval_train_img_data, batch_size=batch_size, device=device)

                print("Evaluating on test set ... Extracting features 2/2")
                test_feats_simclr = extract_model_features(model_eval, eval_test_img_data, batch_size=batch_size, device=device)

            print("Evaluating on test set ... Training KNN")

            eval_result = train_knn(train_feats_simclr, test_feats_simclr)

            print("Evaluating on test set ... DONE, AUC", eval_result, f"in {time() - time_start} seconds")

            #print(logreg_results)
            if eval_result > best_eval_score:
                best_eval_score = eval_result

            if logger:
                logger.log({"test_knn_auc" : eval_result})
                logger.log({"test_knn_auc_max" : best_eval_score})

        if epoch % 10 == 0 and device == "cuda:0" and model_checkpoint_filename:
            fn = f"{model_checkpoint_filename}_epoch_{epoch}.pt"

            print("Saving model to", fn)
            torch.save(model.state_dict(), fn)

        if dist.is_initialized():
            print("DDP is running, wait for barrier")
            dist.barrier()

    return model