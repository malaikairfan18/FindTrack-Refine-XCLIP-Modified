import torch
import torch.nn.functional as F

def compute_ssa_scores(image_features_list):
    """
    Computes the Soft Semantic Alignment (SSA) consistency score for each candidate frame.
    
    Args:
        image_features_list (list of torch.Tensor): List of Alpha-CLIP image embeddings.
            Each tensor is expected to be of shape [1, D].
            
    Returns:
        ssa_scores (torch.Tensor): A 1D tensor of shape [N] containing the consistency score 
            for each candidate. Values are typically in the range [-1, 1].
    """
    if not image_features_list:
        return torch.tensor([])

    # Stack features into shape [N, D]
    feats = torch.stack(image_features_list, dim=0).squeeze(1)
    
    # Normalize features to compute cosine similarity safely 
    # (Note: Alpha-CLIP already normalizes these, but doing it again ensures stability)
    feats = F.normalize(feats, p=2, dim=1)
    
    # Compute pairwise cosine similarity matrix [N, N]
    # sim_matrix[i, j] = cos_sim(feats[i], feats[j])
    sim_matrix = torch.matmul(feats, feats.t())
    
    # The SSA consistency score for frame i is the average similarity to all candidate frames
    # A higher score indicates the frame's semantic appearance is well-aligned with the rest of the sequence
    ssa_scores = torch.mean(sim_matrix, dim=1)
    
    return ssa_scores
