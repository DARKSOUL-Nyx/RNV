import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

def calculate_metrics(model, dataloader, device, threshold=0.5):
    """
    Calculates precision, recall, and F1-score for the Siamese network.
    A threshold is used to binarize the similarity score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for pre_image, post_image, labels in dataloader:
            pre_image, post_image = pre_image.to(device), post_image.to(device)
            
            # Get the feature vectors
            f1, f2 = model.encoder(pre_image), model.encoder(post_image)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(f1, f2)
            
            # Predict based on the threshold
            preds = (similarity > threshold).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    return {"precision": precision, "recall": recall, "f1_score": f1}