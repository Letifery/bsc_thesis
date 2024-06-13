import time, sys, random, copy
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import torcheval.metrics
import numpy as np
from .dataloader import DataLoader

sys.path.append('./')
from tools import createdataset

def train_and_validate(model, loss_criterion, optimizer, epochs, model_name, classes, dataloader, folds, scheduler):
    stats = [[None]*epochs]*folds
    seed_init = random.randrange(1000000)
    best_fold_acc, least_fold_acc = [0]*folds, [100]*folds
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if classes == 2:
        prc_metric = torcheval.metrics.BinaryPrecision()
        rec_metric = torcheval.metrics.BinaryRecall()
    else:
        prc_metric = torcheval.metrics.MulticlassPrecision(num_classes=classes)
        rec_metric = torcheval.metrics.MulticlassRecall(num_classes=classes)
    
    for k in range(folds):
        if folds > 1:
            if k == 0 :
                model_savestate = copy.deepcopy(model)
            createdataset.create_dataset(dataloader.DATA_PATH, dataloader.TVT_PATH, fsplit = (k, folds), seed = seed_init)
            model = copy.deepcopy(model_savestate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)                          #Currently have to be declared here due to parameter pointers getting lost when making deepcopies, drastically reducing accuracy
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 3)      #Same here
        else:
            print("t")
            createdataset.create_dataset(dataloader.DATA_PATH, dataloader.TVT_PATH, tvt_split= (0.8,0.2, 0.0), seed = seed_init)
        data = dataloader.get_data(False)
        train_dataloader, valid_dataloader = dataloader.get_iterators(data, 6, False) 
        
#        tf, _ = next(iter(train_dataloader))
#        imshow(torchvision.utils.make_grid(tf))
        
        for epoch in range(epochs):
            prc_metric.reset()
            rec_metric.reset()
            epoch_start = time.time()
            model.train()
            train_loss, train_acc, valid_loss, valid_acc, avg_prc, avg_rec, avg_f1 = 0, 0, 0, 0, 0, 0, 0
            val_pred, val_labels = [],[]
            
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step(epoch+i / len(train_dataloader))
                train_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs.data, 1)
                train_acc += ((predictions == labels).sum().item())/len(inputs)
            
            with torch.no_grad():
                model.eval()
                for j, (inputs, labels) in enumerate(valid_dataloader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = loss_criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
                    _, predictions = torch.max(outputs.data, 1)
                    val_pred += predictions.tolist()
                    val_labels += labels.tolist()
                    print(predictions, labels)
                    valid_acc += ((predictions == labels).sum().item())/len(inputs)
                    avg_prc += prc_metric.update(predictions, labels).compute().item()
                    avg_rec += rec_metric.update(predictions, labels).compute().item()
                    avg_f1 += torcheval.metrics.functional.multiclass_f1_score(predictions, labels, num_classes=classes).item()
                    
            print("\nF%s: <%s/%s> Runtime:%s seconds\n" % (k+1, epoch+1, epochs, time.time()-epoch_start))


            train_loss /= len(data["train"]) 
            train_acc /= len(train_dataloader)

            valid_loss /= len(data["val"]) 
            valid_acc /= len(valid_dataloader)

            avg_prc /= len(valid_dataloader) 
            avg_rec /= len(valid_dataloader) 
            
            f1score = avg_f1/(len(valid_dataloader)) if classes > 2 else 0 if avg_rec == avg_prc == 0 else 2*((avg_prc*avg_rec)/(avg_prc+avg_rec))
            stats[k][epoch] = [train_loss, valid_loss, train_acc, valid_acc, avg_prc, avg_rec, f1score, val_pred, val_labels]
            
            if valid_acc > best_fold_acc[k]:
                best_fold_acc[k] = valid_acc
                if valid_acc >= max(best_fold_acc):
                    torch.save(model, "./models//"+model_name+'_model_best.pt')
            
            if valid_acc < least_fold_acc[k]:
                least_fold_acc[k] = valid_acc
                if valid_acc <= min(best_fold_acc):
                    torch.save(model, "./models//"+model_name+'_model_least.pt')
            
            print("[TRAINING] ACCURACY : %s\n[TRAINING] LOSS : %s" % (100*train_acc, train_loss))
            print("[VALIDATION] ACCURACY : %s\n[VALIDATION] LOSS : %s\n" % (100*valid_acc, valid_loss))
            print("[VALIDATION] PRECISION : %s\n[VALIDATION] RECALL : %s\n[VALIDATION] F1-Score : %s\n" % (100*avg_prc, 100*avg_rec, 100*f1score))
    print(best_fold_acc, least_fold_acc)
    return model, stats[best_fold_acc.index(max(best_fold_acc))], stats[least_fold_acc.index(min(least_fold_acc))]

def predict(model, test_image_name):
    #Currently not usable, has to be fixed
    dl = DataLoader()
    test_image = Image.open(test_image_name)
    test_image_tensor = dl.image_transforms['test'](test_image.convert("RGB"))
    
    test_image_tensor = test_image_tensor.view(1, 3, 128, 128).cuda() if torch.cuda.is_available() else test_image_tensor.view(1, 3, 128, 128)
    
    with torch.no_grad():
        model.eval()
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(1, dim=1)
        idx_to_class = {v: k for k, v in dl.get_data()['train'].class_to_idx.items()}
        print("Output class :  ", idx_to_class[topclass.cpu().numpy()[0][0]])
        
def imshow(inp, title=None):
    #Taken from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)