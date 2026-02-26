import os
import time
import torch
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import root_mean_squared_error


def train_test_model(model, syreanet, decomnet, train_loader, test_loader, num_epochs, test_interval,
                     device, criterion, optimizer, scheduler, results_file, record_file, models_save_path):

    os.makedirs(models_save_path, exist_ok=True)

    model.to(device)

    for epoch in range(num_epochs):
        time_start = time.time()
        model.train()
        epoch_loss = 0.0

        for data in train_loader:
            images = data['image'].to(device)
            labels = data['label'].to(device).float()

            optimizer.zero_grad()
            a, B, T, w = syreanet(images)
            R, I = decomnet(images)
            outputs = model(images, T,B,R, I)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        f = open(record_file, 'a+')
        print("Epoch {:3d}/{:3d},  Loss: {:7.4f},  Time: {:7.4f} seconds".format(epoch+1, num_epochs, epoch_loss/len(train_loader), time.time()-time_start))
        f.write("Epoch {:3d}/{:3d},  Loss: {:7.4f},  Time: {:7.4f} seconds\n".format(epoch+1, num_epochs, epoch_loss/len(train_loader), time.time()-time_start))
        f.close()

        if (epoch + 1) % test_interval == 0:
            torch.save(model.state_dict(), f'{models_save_path}/pretrained_model_{epoch+1}.pth')
            print("Save Pretrained Model!")
            test_model(model, syreanet, decomnet, test_loader, device, results_file, record_file)


def test_model(model, syreanet, decomnet, test_loader, device, results_file, record_file):
    model.eval()
    image_paths = []
    labels = []
    predictions = []

    with torch.no_grad():
        for data in test_loader:
            images = data['image'].to(device)
            labels_batch = data['label'].numpy()
            a, B, T, w = syreanet(images)
            R, I = decomnet(images)
            outputs = model(images, T,B,R, I).cpu().numpy().squeeze(0)

            image_paths.extend(data['image_path'])
            labels.extend(labels_batch)
            predictions.extend(outputs)

    results = pd.DataFrame({
        'image': image_paths,
        'label': labels,
        'prediction': predictions
    })
    results.to_excel(results_file, index=False)
    print("Save Test Results!")

    srocc, _ = spearmanr(labels, predictions)
    krocc, _ = kendalltau(labels, predictions)
    plcc, _ = pearsonr(labels, predictions)
    rmse = root_mean_squared_error(labels, predictions)

    f = open(record_file, 'a+')
    print(f"SROCC: {srocc}, KROCC: {krocc}, PLCC: {plcc}, RMSE: {rmse}\n")
    f.write(f"SROCC: {srocc}, KROCC: {krocc}, PLCC: {plcc}, RMSE: {rmse}\n\n")
    f.close()