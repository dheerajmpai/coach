"""
This file consists of the main functions used to train the knowledge model.
"""

from DataLoaders import prepare_data
from KnowledgeModel import KnowledgeModel
import torch
from tqdm import tqdm
import numpy as np
import gc
import wandb
import sys
sys.path.append("../Utils")
from Utils import evaluate_metrics, get_word2vec
from future.utils import iterkeys, iteritems

def train_model(model, train_loader, criterion, optimizer):
    model.train()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 

    total_loss = 0
    total_f1 = 0

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()

        x_encoder,x_decoder,y_encoder,y_decoder,lx_encoder,lx_decoder, ly_encoder, ly_decoder = data
        x_encoder,x_decoder,y_encoder,y_decoder = x_encoder.cuda(),x_decoder.cuda(),y_encoder.cuda(),y_decoder.cuda()

        with torch.cuda.amp.autocast():     
            decoder_out = model(x_encoder,lx_encoder,y_encoder,x_decoder,lx_decoder)
            decoder_out_reshaped = decoder_out.reshape(decoder_out.shape[0] * decoder_out.shape[1], decoder_out.shape[2])
            target_reshaped = y_decoder.reshape(-1)
            loss        =  criterion(decoder_out_reshaped, target_reshaped.type(torch.LongTensor).cuda())

        total_loss += loss.item()

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr'])))

        batch_bar.update()

        scaler.scale(loss).backward() 
        scaler.step(optimizer) 
        scaler.update()

        del x_encoder,x_decoder,y_encoder,y_decoder,lx_encoder,lx_decoder, ly_encoder, ly_decoder, decoder_out, loss 
        torch.cuda.empty_cache()


    batch_bar.close()
    
    return total_loss / len(train_loader)

def validate_model(model, val_loader):
    model.eval()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    total_loss = 0
    results = []
    sigmoid = torch.nn.Sigmoid()
    for i, data in enumerate(val_loader):

        x_encoder,x_decoder,y_encoder,y_decoder,lx_encoder,lx_decoder, ly_encoder, ly_decoder = data
        x_encoder,x_decoder,y_encoder,y_decoder = x_encoder.cuda(),x_decoder.cuda(),y_encoder.cuda(),y_decoder.cuda()

        with torch.cuda.amp.autocast():   
          with torch.inference_mode():
            decoder_out = model(x_encoder,lx_encoder,y_encoder,x_decoder,lx_decoder)
            decoder_out_reshaped = decoder_out.reshape(decoder_out.shape[0] * decoder_out.shape[1], decoder_out.shape[2])
            target_reshaped = y_decoder.reshape(-1)
            loss        =  criterion(decoder_out_reshaped, target_reshaped.type(torch.LongTensor).cuda())

        i = 0
        decoder_out_sigmoid = torch.nn.functional.softmax(decoder_out, dim = -1)
        for out in decoder_out_sigmoid.detach().cpu().numpy():
          out1 = out[0:lx_decoder[i]]
          results.extend(out1)
          i+=1

        total_loss += float(loss.item())
        batch_bar.set_postfix(loss="{:.04f}".format(float(total_loss / (i + 1))))
        batch_bar.update()
    
        del x_encoder,x_decoder,y_encoder,y_decoder,lx_encoder,lx_decoder, ly_encoder, ly_decoder,decoder_out, loss
        torch.cuda.empty_cache()
        
    batch_bar.close()
    total_loss = total_loss/len(val_loader)
    return total_loss, results

if __name__ == '__main__':
    
    train_loader, val_loader, test_loader, training_labels, \
    valid_labels, test_labels,  word2Idx, pos2Idx, depLabel2Idx, wordLabel2Idx = prepare_data()
    
    model = KnowledgeModel(
        token_vocabulary = list(word2Idx.keys()),
        pos_vocab_size = len(list(pos2Idx.keys())),
        depLabelVocab_size = len(list(depLabel2Idx.keys())),
        wordLabelVocab_size = len(list(wordLabel2Idx.keys())),
        word2vec = get_word2vec(),
        encoder_hidden_size  = 128,
        decoder_hidden_size = 128,
        projection_size = 256,
        output_size = 3
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='min') # fill this out
    criterion = torch.nn.CrossEntropyLoss(reduction = 'mean' ,weight = torch.tensor([1.0, 5.0, 0]).to(device), ignore_index = 2)
    scaler = torch.cuda.amp.GradScaler()

    gc.collect()
    torch.cuda.empty_cache()

    actual = []
    for instance_id in iterkeys(test_labels):
      try:
          actual.append(test_labels[instance_id])
      except KeyError:
          print('No prediction for instance ID ' + instance_id + '!')

    wandb.login(key="4a6e96eb645ce23f4ada4b7f5106dcbaed287c63")
    run = wandb.init(
        name = "FinalKMModel - 4", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "ProjectFinalAblations" ### Project should be created in your wandb account 
    )


    best_f1 = float('-inf')
    for epoch in range(0, 50):

        print("\nEpoch: {}/{}".format(epoch+1, 50))
        
        curr_lr = optimizer.param_groups[0]['lr']

        train_loss = train_model(model, train_loader, criterion, optimizer) 
        valid_loss, val_results  = validate_model(model, val_loader)
        
        scheduler.step(valid_loss)

        print("\tTrain Loss {:.04f}\t Learning Rate {:.07f}".format(train_loss, curr_lr))
        print("Val Loss {:.04f}".format(valid_loss))
        res = np.stack(val_results, axis = 0)
        ev_Res = evaluate_metrics(actual, res[:,1])
        f1 = ev_Res[2]
        print(ev_Res)

        wandb.log({
            'train_loss': train_loss,  
            'valid_loss': valid_loss,
            'f1' : ev_Res[2],
            'auc-roc': ev_Res[1],
            'acc': ev_Res[0],
            'lr'        : curr_lr
        })
        
        if f1 >= best_f1:
            best_f1 = f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict'     : optimizer.state_dict()
                }, "/content/drive/MyDrive/KM-ModelFinal3.pt")
            print("Saved best model")
    run.finish()