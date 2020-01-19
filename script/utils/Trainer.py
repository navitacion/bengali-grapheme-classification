import copy
import pandas as pd

import torch
from torch import nn


def train_model(net, dataloader_dict, weights_dict, optimizer, device, num_epoch):
    print('Bengali Training...')
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 1000.0
    net = net.to(device)
    train_loss_list = []
    val_loss_list = []

    criterion_g = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['g']).to(device))
    criterion_v = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['v']).to(device))
    criterion_c = nn.CrossEntropyLoss(weight=torch.tensor(weights_dict['c']).to(device))

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0

            for inputs, target_g, target_v, target_c, _ in tqdm(dataloader_dict[phase]):
                inputs = inputs.to(device)
                target_g = target_g.to(device)
                target_v = target_v.to(device)
                target_c = target_c.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_g, outputs_v, outputs_c = net(inputs)
                    loss_g = criterion_g(outputs_g, target_g.long())
                    loss_v = criterion_v(outputs_v, target_v.long())
                    loss_c = criterion_c(outputs_c, target_c.long())

                    loss = loss_g + loss_v + loss_c

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)

                del inputs, target_g, target_v, target_c
                gc.collect()
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)

            print(f'Epoch {epoch + 1}/{num_epoch} {phase} Loss: {epoch_loss}')

            # Save Epoch Loss
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())

        df_loss = pd.DataFrame({
            'Train_loss': train_loss_list,
            'Val_loss': val_loss_list
        })

    time_elapsed = time.time() - since
    print('Training complete in {}'.format(str(datetime.timedelta(seconds=time_elapsed))))
    print('Best val Acc: {:4f}'.format(best_loss))

    df_loss = pd.DataFrame({
        'Epoch': np.arange(num_epoch),
        'Train_loss': train_loss_list,
        'Val_loss': val_loss_list
    })

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, best_loss, df_loss