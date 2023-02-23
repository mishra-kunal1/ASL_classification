import torch 


def model_train(n_epochs,train_loader,val_loader,current_model,criterion,optimizer,device='cpu'):
    
    for epoch in range(n_epochs):
        # Set the model to train mode
        current_model.to(device)
        current_model.train()

        # Loop over the training data in batches
        for i, (images, labels) in enumerate(train_loader):
            # Move the batch to the GPU if available
            images = images.to(device)
            #print(images.shape)
            labels = labels.to(device)

            # Forward pass
            outputs = current_model(images)
            #print(outputs.shape)
            #print(labels.shape)
            loss = criterion(outputs, labels)

            # Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            #to optimize space del the images, labels and outputs for current epoch
            #we can empty the cache
            #del images, labels, outputs
            #torch.cuda.empty_cache()

            # Print training progress
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch+1, n_epochs, i+1, len(train_loader), loss.item()))

        # Evaluate the model on the validation set
        current_model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                # Move the batch to the GPU if available
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass and prediction
                outputs = current_model(images)
                _, predicted = torch.max(outputs.data, 1)

                # Compute accuracy
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = 100 * correct / total
            print('Validation Accuracy: {:.2f}%'.format(val_accuracy))
            #save the model's state in dictionary
            torch.save(current_model.state_dict(), '/kaggle/working/trained_model_weights')
