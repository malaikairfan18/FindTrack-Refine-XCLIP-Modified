import sys
import os
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(current_dir, "EVF-SAM"))
sys.path.append(os.path.join(current_dir, "AlphaCLIP"))

import alphaclip
from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
from utils import *
from ssa_module import compute_ssa_scores
import argparse
import cv2
import json
import numpy as np
from PIL import Image
import torch
import torchvision as tv
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings('ignore')


def compute_clip_similarity_and_features(clip, clip_preprocess, clip_preprocess_mask, image_np, mask_tensor, clip_text, mode="mask_crop"):
    """
    Computes CLIP similarity score and returns the raw image features for SSA consistency evaluation.
    """
    if len(mask_tensor.shape) == 2:
        mask_tensor = mask_tensor.unsqueeze(0)  # [1, H, W]
    elif len(mask_tensor.shape) == 3 and mask_tensor.shape[0] > 1:
        mask_tensor = mask_tensor.mean(dim=0, keepdim=True)

    mask_np = (mask_tensor.squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
    H, W = mask_np.shape

    # 1. Full Frame mode
    if mode == "full_frame":
        pil_img = Image.fromarray(image_np)
        img_clip = clip_preprocess(pil_img).unsqueeze(0).cuda()
        alpha = clip_preprocess_mask(mask_tensor).cuda()
        
        image_features = clip.visual(img_clip, alpha.unsqueeze(0))
        text_features = clip.encode_text(clip_text)
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = torch.matmul(image_features_norm, text_features_norm.transpose(0, 1))[0]
        return sim, image_features

    # Find bounding box
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) > 0:
        ymin, ymax = y_indices.min(), y_indices.max()
        xmin, xmax = x_indices.min(), x_indices.max()
        
        h_box, w_box = ymax - ymin, xmax - xmin
        pad_y = int(h_box * 0.1)
        pad_x = int(w_box * 0.1)
        ymin = max(0, ymin - pad_y)
        ymax = min(H - 1, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(W - 1, xmax + pad_x)
    else:
        ymin, ymax, xmin, xmax = 0, H - 1, 0, W - 1

    # 2. Object Box Crop
    if mode == "object_box_crop":
        crop_np = image_np[ymin:ymax+1, xmin:xmax+1]
        pil_crop = Image.fromarray(crop_np)
        img_clip = clip_preprocess(pil_crop).unsqueeze(0).cuda()
        
        ones_mask = torch.ones((1, ymax-ymin+1, xmax-xmin+1), dtype=torch.float32)
        alpha = clip_preprocess_mask(ones_mask).cuda()
        
        image_features = clip.visual(img_clip, alpha.unsqueeze(0))
        text_features = clip.encode_text(clip_text)
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = torch.matmul(image_features_norm, text_features_norm.transpose(0, 1))[0]
        return sim, image_features

    # 3. Mask Crop
    elif mode == "mask_crop":
        masked_img_np = image_np * mask_np[:, :, np.newaxis]
        crop_np = masked_img_np[ymin:ymax+1, xmin:xmax+1]
        
        pil_crop = Image.fromarray(crop_np)
        img_clip = clip_preprocess(pil_crop).unsqueeze(0).cuda()
        
        mask_crop_np = mask_np[ymin:ymax+1, xmin:xmax+1].astype(np.float32)
        mask_crop_tensor = torch.from_numpy(mask_crop_np).unsqueeze(0)
        alpha = clip_preprocess_mask(mask_crop_tensor).cuda()
        
        image_features = clip.visual(img_clip, alpha.unsqueeze(0))
        text_features = clip.encode_text(clip_text)
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        sim = torch.matmul(image_features_norm, text_features_norm.transpose(0, 1))[0]
        return sim, image_features
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on Ref-YouTube-VOS dataset with FindTrack-Refine-XCLIP-Modified")
    parser.add_argument("--mode", type=str, default="mask_crop", choices=["mask_crop", "object_box_crop", "full_frame"],
                        help="CLIP Reranker mode")
    parser.add_argument("--w_finder", type=float, default=0.4, help="Finder score weight (w1)")
    parser.add_argument("--w_clip", type=float, default=0.4, help="CLIP score weight (w2)")
    parser.add_argument("--w_ssa", type=float, default=0.2, help="SSA consistency score weight (w3)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--dataset_path", type=str, default="../DB/RVOS/YTVOS", help="Path to Ref-YouTube-VOS dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output predictions and force re-evaluation")
    parser.add_argument("--ref_num", type=int, default=10, help="Number of candidate reference frames to sample")
    parser.add_argument("--num_refs", type=int, default=5, help="Number of reference frames to select for tracking (e.g. 5 or 10)")
    parser.add_argument("--min_distance", type=int, default=15, help="Minimum frame distance for temporal diversity")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Entropy confidence threshold for mask refinement")
    parser.add_argument("--overlap_mode", type=str, default="argmax", choices=["argmax", "hard_discard"],
                        help="Overlap suppression mode for multi-object deconfliction")
    return parser.parse_known_args()[0]


def test(args):
    # Initialize EVF-SAM
    tokenizer, evfsam = init_models()

    # Initialize Alpha-CLIP
    clip, clip_preprocess = alphaclip.load('ViT-L/14@336px', alpha_vision_ckpt_pth='weights/clip_l14_336_grit_20m_4xe.pth', device='cuda')
    clip_preprocess_mask = transforms.Compose([transforms.Resize((336, 336)), transforms.Normalize(0.5, 0.26)])

    # Initialize Cutie
    cutie = get_default_model(config='ytvos_config')
    processor = InferenceCore(cutie, cfg=cutie.cfg)

    # Initialize ReSAM Refiner
    refiner = RvosRefiner(epsilon=args.epsilon, overlap_mode=args.overlap_mode)

    # Output directory setup
    output_dir = 'outputs'
    save_path_prefix = os.path.join(output_dir, 'Ref_YTVOS_val')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
        
    root = args.dataset_path
    
    # Auto-detect double nested valid folder for JPEGImages
    img_folder = os.path.join(root, 'valid', 'JPEGImages')
    if not os.path.exists(img_folder):
        img_folder = os.path.join(root, 'valid', 'valid', 'JPEGImages')
        
    # Auto-detect double nested meta_expressions
    meta_file = os.path.join(root, 'meta_expressions', 'valid', 'meta_expressions.json')
    if not os.path.exists(meta_file):
        meta_file = os.path.join(root, 'meta_expressions', 'meta_expressions', 'valid', 'meta_expressions.json')
        
    test_meta_file = os.path.join(root, 'meta_expressions', 'test', 'meta_expressions.json')
    if not os.path.exists(test_meta_file):
        test_meta_file = os.path.join(root, 'meta_expressions', 'meta_expressions', 'test', 'meta_expressions.json')
        
    print(f"Dataset root: {root}")
    print(f"Using img_folder: {img_folder}")
    print(f"Using meta_file: {meta_file}")
    print(f"Using test_meta_file: {test_meta_file}")

    with open(meta_file, 'r') as f:
        data = json.load(f)['videos']
    valid_test_videos = set(data.keys())
    
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in valid_videos])

    # Inference loop
    for idx_, video in enumerate(video_list):
        metas = []
        expressions = data[video]['expressions']
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        for i in range(num_expressions):
            meta = {}
            meta['video'] = video
            meta['exp'] = expressions[expression_list[i]]['exp']
            meta['exp_id'] = expression_list[i]
            meta['frames'] = data[video]['frames']
            metas.append(meta)
        meta = metas
        video_name = video
        frames = data[video]['frames']
        video_len = len(frames)

        # CHECK IF ALREADY EVALUATED (Resume capability)
        already_done = True
        if args.force:
            already_done = False
        else:
            for e in range(num_expressions):
                exp_id = meta[e]['exp_id']
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    already_done = False
                    break
                for frame in frames:
                    if not os.path.exists(os.path.join(save_path, frame + '.png')):
                        already_done = False
                        break
                if not already_done:
                    break
        
        if already_done:
            print(f"Video {idx_+1}/{len(video_list)}: {video} - Already evaluated. Skipping.")
            continue

        print(f"Video {idx_+1}/{len(video_list)}: {video}")

        # Input preprocessing
        imgs_beit = []
        imgs_sam = []
        imgs_clip = []
        imgs_cutie = []
        for i in range(video_len):
            img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
            image_np = cv2.imread(img_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            # Pre-processes
            imgs_beit.append(beit3_preprocess(Image.open(img_path), 224))
            img_sam, resize_shape = sam_preprocess(image_np)
            imgs_sam.append(img_sam)
            imgs_clip.append(clip_preprocess(Image.open(img_path)))
            imgs_cutie.append(tv.transforms.ToTensor()(Image.open(img_path)))

        # ==========================================
        # PHASE 1: GENERATE CANDIDATES FOR ALL EXPRESSIONS
        # ==========================================
        ref_num = args.ref_num
        candidate_indices = []
        for ref_idx in range(ref_num):
            i = int(ref_idx * (video_len - 1) / (ref_num - 1))
            candidate_indices.append(i)
            
        raw_logits_by_frame = {i: [] for i in candidate_indices}
        raw_scores_finder = [[] for _ in range(num_expressions)]
        
        for e in range(num_expressions):
            exp = meta[e]['exp']
            words = tokenizer(exp, return_tensors='pt')['input_ids'].cuda()
            
            for ref_idx, i in enumerate(candidate_indices):
                ref_mask, ref_score = evfsam.inference(imgs_sam[i].unsqueeze(0).cuda(), imgs_beit[i].unsqueeze(0).cuda(), words, resize_shape, original_size_list)
                raw_logits_by_frame[i].append(ref_mask)
                
                evf_val = ref_score.item() if hasattr(ref_score, 'item') else float(ref_score)
                raw_scores_finder[e].append(evf_val)

        # ==========================================
        # PHASE 2: APPLY RESAM REFINE (DENOISE & OVERLAP SUPPRESSION)
        # ==========================================
        refined_masks_by_frame = {i: [] for i in candidate_indices}
        for i in candidate_indices:
            frame_logits = raw_logits_by_frame[i]
            refined_masks = refiner.refine_candidates(frame_logits)
            refined_masks_by_frame[i] = refined_masks

        # ==========================================
        # PHASE 3: RERANK AND TRACK EACH EXPRESSION
        # ==========================================
        for e in range(num_expressions):
            video_name = meta[e]['video']
            exp = meta[e]['exp']
            exp_id = meta[e]['exp_id']
            frames = meta[e]['frames']
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            ref_masks = []
            ref_scores_clip = []
            image_features_list = []
            
            print(f"  Exp: '{exp}'")
            for ref_idx, i in enumerate(candidate_indices):
                refined_mask = refined_masks_by_frame[i][e] # shape [1, H, W]
                ref_masks.append(refined_mask)
                
                clip_text = alphaclip.tokenize([exp]).cuda()
                ref_img_path = os.path.join(img_folder, video_name, frames[i] + '.jpg')
                image_np = cv2.imread(ref_img_path)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

                # Compute Similarity and extract features for SSA
                clip_sim, image_features = compute_clip_similarity_and_features(
                    clip, clip_preprocess, clip_preprocess_mask,
                    image_np, refined_mask, clip_text, mode=args.mode
                )
                clip_val = clip_sim.item() if hasattr(clip_sim, 'item') else float(clip_sim)
                ref_scores_clip.append(clip_val)
                image_features_list.append(image_features.detach())

            # Min-Max Normalization to solve score scale dominance
            ref_scores_finder = raw_scores_finder[e]
            finder_min, finder_max = min(ref_scores_finder), max(ref_scores_finder)
            clip_min, clip_max = min(ref_scores_clip), max(ref_scores_clip)
            
            # Compute SSA consistency scores
            ssa_scores = compute_ssa_scores(image_features_list)
            ssa_scores_list = ssa_scores.cpu().tolist()
            ssa_min, ssa_max = min(ssa_scores_list), max(ssa_scores_list)
            
            finder_range = finder_max - finder_min + 1e-6
            clip_range = clip_max - clip_min + 1e-6
            ssa_range = ssa_max - ssa_min + 1e-6
            
            normalized_finder = [(s - finder_min) / finder_range for s in ref_scores_finder]
            normalized_clip = [(s - clip_min) / clip_range for s in ref_scores_clip]
            normalized_ssa = [(s - ssa_min) / ssa_range for s in ssa_scores_list]
            
            # Combine scores (EVF-SAM: 40%, Alpha-CLIP similarity: 40%, SSA: 20%)
            w1, w2, w3 = args.w_finder, args.w_clip, args.w_ssa
            combined_scores = []
            for i_cand in range(ref_num):
                score = w1 * normalized_finder[i_cand] + w2 * normalized_clip[i_cand] + w3 * normalized_ssa[i_cand]
                combined_scores.append(score)
                print(f"    Frame {frames[candidate_indices[i_cand]]} (idx {candidate_indices[i_cand]:02d}): "
                      f"EVF-SAM={ref_scores_finder[i_cand]:.4f} (Norm={normalized_finder[i_cand]:.4f}), "
                      f"CLIP={ref_scores_clip[i_cand]:.4f} (Norm={normalized_clip[i_cand]:.4f}), "
                      f"SSA={ssa_scores_list[i_cand]:.4f} (Norm={normalized_ssa[i_cand]:.4f}), "
                      f"Combined={score:.4f}")

            # Dynamic min_distance scaling for 5 or 10 reference selection
            optimal_min_dist = max(1, video_len // (args.num_refs + 1))
            min_dist = min(args.min_distance, optimal_min_dist)

            # Temporal Diversity Filter for Top-K Reference Selection
            sorted_indices = np.argsort(combined_scores)[::-1]
            selected_candidate_indices = []
            
            for idx in sorted_indices:
                if len(selected_candidate_indices) >= args.num_refs:
                    break
                current_frame_pos = candidate_indices[idx]
                diverse = True
                for sel_idx in selected_candidate_indices:
                    sel_frame_pos = candidate_indices[sel_idx]
                    if abs(current_frame_pos - sel_frame_pos) < min_dist:
                        diverse = False
                        break
                if diverse:
                    selected_candidate_indices.append(idx)
            
            # Fallback if we couldn't find enough diverse references
            if len(selected_candidate_indices) < args.num_refs:
                for idx in sorted_indices:
                    if len(selected_candidate_indices) >= args.num_refs:
                        break
                    if idx not in selected_candidate_indices:
                        selected_candidate_indices.append(idx)

            # Sort selected references chronologically
            selected_candidate_indices.sort()
            selected_refs = [candidate_indices[idx] for idx in selected_candidate_indices]
            earliest_ref_idx = selected_refs[0]
            earliest_candidate_idx = selected_candidate_indices[0]
            
            print("  => Selected Reference Frames:")
            for idx in selected_candidate_indices:
                f_idx = candidate_indices[idx]
                print(f"     Frame {frames[f_idx]} (idx {f_idx:02d}) with Combined Score: {combined_scores[idx]:.4f}")

            # Forward pass tracking
            for i in range(earliest_ref_idx, video_len):
                if i in selected_refs:
                    ref_list_idx = selected_refs.index(i)
                    cand_idx = selected_candidate_indices[ref_list_idx]
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                if i == video_len - 1:
                    processor.clear_memory()

                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

            # Backward pass tracking
            for i in range(earliest_ref_idx, -1, -1):
                if i == earliest_ref_idx:
                    cand_idx = earliest_candidate_idx
                    mask_prob = processor.step(imgs_cutie[i].cuda(), ref_masks[cand_idx].squeeze(0), objects=[1])
                else:
                    mask_prob = processor.step(imgs_cutie[i].cuda())
                mask = processor.output_prob_to_mask(mask_prob).float()

                if i == 0:
                    processor.clear_memory()

                mask = mask.detach().cpu().numpy().astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frames[i] + '.png')
                mask.save(save_file)

        # Free GPU memory
        processor.clear_memory()
        del imgs_beit, imgs_sam, imgs_clip, imgs_cutie
        del raw_logits_by_frame, refined_masks_by_frame, raw_scores_finder
        import gc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.gpu)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
        test(args)
