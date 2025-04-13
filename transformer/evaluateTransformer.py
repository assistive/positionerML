def evaluate_model(model, test_loader, device, task='accident'):
    """
    Evaluate the model and calculate relevant metrics
    
    Args:
        model: Trained IMUTransformer model
        test_loader: DataLoader for test data
        device: torch device
        task: 'accident' or 'drowsiness'
        
    Returns:
        Dictionary of evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task=task)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays for metric calculation
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_targets)
    
    if task == 'accident':
        # For binary classification
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc
        }
    else:
        # For multi-class classification (drowsiness)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(all_targets, all_preds)
        }
