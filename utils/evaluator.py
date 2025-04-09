import torch
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
import logging

# Download required NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

class Evaluator:
    """
    Evaluator class for image captioning models.
    Supports metrics: BLEU-1,2,3,4, METEOR, ROUGE-L
    """
    
    def __init__(self, save_dir: str = "eval_results"):
        """
        Initialize the evaluator.
        
        Args:
            save_dir: Directory to save evaluation results
        """
        self.save_dir = save_dir
        self.rouge = Rouge()
        self.smooth = SmoothingFunction()
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "visualize_image_caption"), exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def _tokenize(self, caption: str) -> List[str]:
        """
        Tokenize a caption string.
        
        Args:
            caption: Caption string
            
        Returns:
            List of tokens
        """
        return nltk.word_tokenize(caption.lower())
    
    def calculate_bleu(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        """
        Calculate BLEU-1,2,3,4 scores.
        
        Args:
            references: List of lists of reference captions (tokenized) for each image
            hypotheses: List of generated captions (tokenized)
            
        Returns:
            Dictionary with BLEU scores
        """
        bleu_scores = {}
        
        # Calculate BLEU-1,2,3,4
        for i in range(1, 5):
            weights = tuple([1.0 / i] * i + [0.0] * (4 - i))
            score = corpus_bleu(
                references, 
                hypotheses, 
                weights=weights,
                smoothing_function=self.smooth.method1
            )
            bleu_scores[f'BLEU-{i}'] = score
            
        return bleu_scores
    
    def calculate_meteor(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> float:
        """
        Calculate METEOR score.
        
        Args:
            references: List of lists of reference captions (tokenized)
            hypotheses: List of generated captions (tokenized)
            
        Returns:
            METEOR score
        """
        scores = []
        
        for hyp, refs in zip(hypotheses, references):
            # METEOR expects tokenized inputs (lists of strings)
            # Calculate METEOR for each hypothesis against all its references
            # and take the maximum score
            meteor_scores = [
                meteor_score([ref], hyp) for ref in refs
            ]
            scores.append(max(meteor_scores) if meteor_scores else 0)
            
        return np.mean(scores)
    
    def calculate_rouge(self, references: List[List[List[str]]], hypotheses: List[List[str]]) -> Dict[str, float]:
        """
        Calculate ROUGE-L score.
        
        Args:
            references: List of lists of reference captions (tokenized)
            hypotheses: List of generated captions (tokenized)
            
        Returns:
            Dictionary with ROUGE scores
        """
        # Convert tokens back to strings for Rouge
        hyp_str = [' '.join(hyp) for hyp in hypotheses]
        
        # For each hypothesis, use the reference that gives the highest score
        rouge_scores = {'rouge-l-f': 0.0, 'rouge-l-p': 0.0, 'rouge-l-r': 0.0}
        count = 0
        
        for i, hyp in enumerate(hyp_str):
            refs_str = [' '.join(ref) for ref in references[i]]
            best_score = None
            
            for ref in refs_str:
                try:
                    scores = self.rouge.get_scores(hyp, ref)[0]['rouge-l']
                    if best_score is None or scores['f'] > best_score['f']:
                        best_score = scores
                except Exception as e:
                    self.logger.warning(f"Error calculating ROUGE for hypothesis {i}: {e}")
                    continue
            
            if best_score:
                rouge_scores['rouge-l-f'] += best_score['f']
                rouge_scores['rouge-l-p'] += best_score['p']
                rouge_scores['rouge-l-r'] += best_score['r']
                count += 1
        
        # Calculate average
        if count > 0:
            rouge_scores['rouge-l-f'] /= count
            rouge_scores['rouge-l-p'] /= count
            rouge_scores['rouge-l-r'] /= count
        
        return rouge_scores
    
    def evaluate(self, references_list: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
        """
        Evaluate generated captions against references using multiple metrics.
        
        Args:
            references_list: List of lists of reference captions
            hypotheses: List of generated captions
            
        Returns:
            Dictionary with evaluation scores
        """
        # Tokenize hypotheses
        tokenized_hyps = [self._tokenize(hyp) for hyp in hypotheses]
        
        # Tokenize references - properly format for BLEU calculation
        # BLEU expects list of list of references for each hypothesis
        tokenized_refs = [[self._tokenize(ref) for ref in refs] for refs in references_list]
        
        # Calculate metrics
        bleu_scores = self.calculate_bleu(tokenized_refs, tokenized_hyps)
        meteor_score = self.calculate_meteor(tokenized_refs, tokenized_hyps)
        rouge_scores = self.calculate_rouge(tokenized_refs, tokenized_hyps)
        
        # Combine all scores
        results = {
            **bleu_scores,
            'METEOR': meteor_score,
            **rouge_scores
        }
        
        return results
    
    def save_results(self, results: Dict[str, float], filename: str = "evaluation_results.json") -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Dictionary with evaluation scores
            filename: Name of the output file
        """
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f"Evaluation results saved to {filepath}")
        
        # Also print results
        self.logger.info("Evaluation Results:")
        for metric, score in results.items():
            self.logger.info(f"{metric}: {score:.4f}")
    
    def save_sample_captions(self, image_paths: List[str], references: List[List[str]], 
                            hypotheses: List[str], sample_indices: List[int], 
                            attention_maps: Optional[List[np.ndarray]] = None) -> None:
        """
        Save sample captions and optionally attention maps for visualization.
        
        Args:
            image_paths: List of image paths
            references: List of lists of reference captions
            hypotheses: List of generated captions
            sample_indices: Indices of samples to save
            attention_maps: Optional list of attention maps
        """
        samples = []
        
        for idx in sample_indices:
            if idx >= len(image_paths):
                continue
                
            sample = {
                "image_path": image_paths[idx],
                "references": references[idx],
                "hypothesis": hypotheses[idx]
            }
            samples.append(sample)
        
        # Save samples
        filepath = os.path.join(self.save_dir, "visualize_image_caption", "samples.json")
        with open(filepath, 'w') as f:
            json.dump(samples, f, indent=4)
        
        # If attention maps are provided, save them
        if attention_maps:
            for i, idx in enumerate(sample_indices):
                if idx < len(attention_maps):
                    attn_map = attention_maps[idx]
                    np.save(
                        os.path.join(self.save_dir, "visualize_image_caption", f"attention_map_{i}.npy"),
                        attn_map
                    )
