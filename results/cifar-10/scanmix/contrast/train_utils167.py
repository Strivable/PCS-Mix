import torch
import torch.nn as nn
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def selflabel_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = criterion(output, output_augmented)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)


def scanmix_train(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion,lambda_u,device,prototype_manager=None,lambda_proto=1.0):
    print(f'\n========== ENTERING scanmix_train for EPOCH {epoch} ==========')
    sys.stdout.flush()
    
    net.train()
    net2.eval()
    
    labeled_losses = AverageMeter('Labelled Loss', ':.4e')
    unlabeled_losses = AverageMeter('Unlabelled Loss', ':.4e')
    
    update_counter = [0]
    
    if prototype_manager is not None:
        proto_losses = AverageMeter('Prototype Loss', ':.4e')
        progress = ProgressMeter(len(labeled_trainloader),
            [labeled_losses, unlabeled_losses, proto_losses],
            prefix="Epoch: [{}]".format(epoch))
        print(f'\n*** PROTOTYPE MANAGER ACTIVE for Epoch {epoch} ***')
        sys.stdout.flush()
    else:
        progress = ProgressMeter(len(labeled_trainloader),
            [labeled_losses, unlabeled_losses],
            prefix="Epoch: [{}]".format(epoch))
        print(f'\n*** WARNING: NO PROTOTYPE MANAGER for Epoch {epoch} ***')
        sys.stdout.flush()
    
    if prototype_manager is not None:
        proto_criterion = nn.CrossEntropyLoss()
        temperature = 0.1

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//p['batch_size'])+1
    
    print(f'\n@@@ ABOUT TO START LOOP: {len(labeled_trainloader)} batches, prototype_manager={prototype_manager is not None} @@@')
    sys.stdout.flush()
    
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        if batch_idx == 0:
            if prototype_manager is not None:
                print(f'\n*** batch 0: prototype_manager is NOT None ***')
            else:
                print(f'\n*** batch 0: prototype_manager IS None! ***')
            sys.stdout.flush()
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, p['num_class']).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u, forward_pass='dm')
            outputs_u12 = net(inputs_u2, forward_pass='dm')
            outputs_u21 = net2(inputs_u, forward_pass='dm')
            outputs_u22 = net2(inputs_u2, forward_pass='dm')            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/p['T']) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x, forward_pass='dm')
            outputs_x2 = net(inputs_x2, forward_pass='dm')            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/p['T']) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(p['alpha'], p['alpha'])        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input, forward_pass='dm')
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:],lambda_u, epoch+batch_idx/num_iter, p['warmup'])
        
        # 提取特征并计算原型损失（使用上一个batch更新后的原型）
        L_proto = 0
        features_x_clean = None
        target_labels_x_clean = None
        if prototype_manager is not None:
            if batch_idx == 0:
                print(f'\n*** ENTERING PROTOTYPE CODE at batch 0, epoch {epoch} ***')
                sys.stdout.flush()
            try:
                # 提取当前batch的特征
                features_x = net(inputs_x, forward_pass='backbone')
                features_x2 = net(inputs_x2, forward_pass='backbone')
                features_x_clean = torch.cat([features_x, features_x2], dim=0)
                
                target_labels_x = targets_x.argmax(dim=1)
                target_labels_x_clean = torch.cat([target_labels_x, target_labels_x], dim=0)
                
                # 计算原型对比损失（参与反向传播）
                features_norm = F.normalize(features_x_clean, dim=1)
                prototypes_norm = F.normalize(prototype_manager.prototypes, dim=1)
                similarities = torch.mm(features_norm, prototypes_norm.t()) / temperature
                L_proto = proto_criterion(similarities, target_labels_x_clean)
                
                proto_losses.update(L_proto.item())
                
            except Exception as e:
                print(f'\n*** EXCEPTION in prototype code at batch {batch_idx}: {e} ***')
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
                raise
        
        # regularization
        prior = torch.ones(p['num_class'])/p['num_class']
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu + penalty
        if prototype_manager is not None:
            loss = loss + lambda_proto * L_proto
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'  [Batch {batch_idx}] Network backward completed')

        # 反向传播完成后，使用当前batch的特征更新原型
        if prototype_manager is not None:
            with torch.no_grad():
                if features_x_clean is not None and features_x_clean.size(0) > 0:
                    update_counter[0] += 1
                    if batch_idx % 100 == 0:
                        print(f'  [Batch {batch_idx}] Now updating prototypes (after backward)...')
                    prototype_manager.update_prototypes(
                        features=features_x_clean.detach(),
                        labels=target_labels_x_clean,
                        momentum=0.9
                    )
                    if update_counter[0] == 1:
                        print(f'\n*** FIRST PROTOTYPE UPDATE in Epoch {epoch} (after backward)! ***')
                        sys.stdout.flush()
                else:
                    print(f'  [WARNING] Empty labeled batch at iteration {batch_idx}')

        labeled_losses.update(Lx.item())
        unlabeled_losses.update(Lu.item())

        if batch_idx % 25 == 0:
            progress.display(batch_idx)
    
    if prototype_manager is not None:
        print(f'\n*** Epoch {epoch} completed: {update_counter[0]} prototype updates made ***')
        print(f'*** Prototype manager update_count: {prototype_manager.update_count} ***\n')
        sys.stdout.flush()

def scanmix_big_train(p,epoch,net,net2,optimizer,labeled_trainloader,unlabeled_trainloader,criterion,lambda_u,device):
    net.train()
    net2.eval() #fix one network and train the other
    
    final_loss = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(labeled_trainloader),
        [final_loss],
        prefix="Epoch: [{}]".format(epoch))

    unlabeled_train_iter = iter(unlabeled_trainloader)    
    num_iter = (len(labeled_trainloader.dataset)//p['batch_size'])+1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):      
        try:
            inputs_u, inputs_u2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = next(unlabeled_train_iter)                 
        batch_size = inputs_x.size(0)
        
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, p['num_class']).scatter_(1, labels_x.view(-1,1), 1)        
        w_x = w_x.view(-1,1).type(torch.FloatTensor) 

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.to(device), inputs_x2.to(device), labels_x.to(device), w_x.to(device)
        inputs_u, inputs_u2 = inputs_u.to(device), inputs_u2.to(device)

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u, forward_pass='dm')
            outputs_u12 = net(inputs_u2, forward_pass='dm')
            outputs_u21 = net2(inputs_u, forward_pass='dm')
            outputs_u22 = net2(inputs_u2, forward_pass='dm')            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4       
            ptu = pu**(1/p['T']) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach()       
            
            # label refinement of labeled samples
            outputs_x = net(inputs_x, forward_pass='dm')
            outputs_x2 = net(inputs_x2, forward_pass='dm')            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px              
            ptx = px**(1/p['T']) # temparature sharpening 
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       
        
        # mixmatch
        l = np.random.beta(p['alpha'], p['alpha'])        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a[:batch_size*2] + (1 - l) * input_b[:batch_size*2]        
        mixed_target = l * target_a[:batch_size*2] + (1 - l) * target_b[:batch_size*2]
                
        logits = net(mixed_input, forward_pass='dm')
        
        Lx = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * mixed_target, dim=1))
        
        prior = torch.ones(p['num_class'])/p['num_class']
        prior = prior.to(device)        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
       
        loss = Lx + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        final_loss.update(loss.item())

        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_warmup(epoch,net,optimizer,dataloader,criterion, conf_penalty, noise_mode, device):
    net.train()
    losses = AverageMeter('CE-Loss', ':.4e')
    progress = ProgressMeter(len(dataloader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):    
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad()
        with torch.no_grad():
            input_features = net(inputs, forward_pass='backbone')
        outputs = net(input_features, forward_pass='dm_head')      
        loss = criterion(outputs, labels)  
        if noise_mode=='asym' or 'semantic' in noise_mode:  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        losses.update(L.item()) 
        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_big_warmup(p,epoch,net,optimizer,dataloader,criterion, conf_penalty, noise_mode, device):
    net.train()
    losses = AverageMeter('CE-Loss', ':.4e')
    progress = ProgressMeter(len(dataloader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, batch in enumerate(dataloader): 
        inputs, labels = batch['image'].to(device), batch['target'].to(device)     
        optimizer.zero_grad()
        with torch.no_grad():
            input_features = net(inputs, forward_pass='backbone')
        outputs = net(input_features, forward_pass='dm_head')      
        loss = criterion(outputs, labels)  
        if p['dataset'] in  ['webvision', 'mini_imagenet_red', 'mini_imagenet32_red']:   
            L = loss
        else:
            raise NotImplementedError()
        L.backward()  
        optimizer.step() 
        losses.update(L.item()) 
        if batch_idx % 25 == 0:
            progress.display(batch_idx)

def scanmix_eval_train(args,model,all_loss,epoch,eval_loader,criterion,device):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset))    
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device) 
            outputs = model(inputs, forward_pass='dm')
            _, predicted = torch.max(outputs, 1) 
            loss = criterion(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                pl[index[b]]  = predicted[b]        
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalised losses for each image
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob,all_loss,pl


def scanmix_big_eval_train(p,args,model,epoch,eval_loader,criterion,device,output):    
    model.eval()
    losses = torch.zeros(len(eval_loader.dataset))
    pl = torch.zeros(len(eval_loader.dataset))    
    processed = AverageMeter('Eval train')
    progress = ProgressMeter(len(eval_loader),
        [processed],
        prefix="Epoch: [{}]".format(epoch))
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader): 
            inputs, targets = batch['image'].to(device), batch['target'].to(device) 
            index = batch['meta']['index']
            outputs = model(inputs, forward_pass='dm')
            _, predicted = torch.max(outputs, 1) 
            loss = criterion(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]
                pl[index[b]]  = predicted[b]
            if batch_idx % 25 == 0:
                progress.display(batch_idx)
    losses = (losses-losses.min())/(losses.max()-losses.min())    # normalised losses for each image
    input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    if (p['dataset'] == 'webvision'):
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    else:
        gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=1e-3)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    output['prob'] = prob
    output['pl'] = pl         



def scanmix_scan(train_loader, model, criterion, optimizer, epoch, device, update_cluster_head_only=False):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['anchor'].to(device, non_blocking=True)
        neighbors = batch['neighbor'].to(device, non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='sl_head')
            neighbors_output = model(neighbors_features, forward_pass='sl_head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors, forward_pass='sl')
            neighbors_output = model(neighbors, forward_pass='sl')     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = criterion(anchors_output, neighbors_output)
        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(total_loss.item())
        consistency_losses.update(consistency_loss.item())
        entropy_losses.update(entropy_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)
        torch.cuda.empty_cache()