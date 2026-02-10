"""
Model Evaluation Module for Text Summarization

This module evaluates pretrained transformer models for text summarization
using quality metrics (ROUGE, BLEU) and efficiency metrics (inference time,
model size, memory usage).

Author: ML Research Assistant
Date: February 10, 2026
"""

import json
import time
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evaluates pretrained text summarization models on multiple criteria.
    
    Attributes:
        models_config (dict): Configuration of models to evaluate
        dataset (list): List of text samples with reference summaries
        results (pd.DataFrame): Evaluation results
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the evaluator with a dataset.
        
        Args:
            dataset_path (str): Path to JSON file containing evaluation data
        """
        self.models_config = {
            'BART-CNN': 'facebook/bart-large-cnn',
            'DistilBART': 'sshleifer/distilbart-cnn-12-6',
            'T5-Small': 't5-small'
        }
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
        
        self.results = []
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
    
    def load_model(self, model_name: str) -> Tuple:
        """
        Load a pretrained model and tokenizer.
        
        Args:
            model_name (str): HuggingFace model identifier
            
        Returns:
            tuple: (model, tokenizer)
        """
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        return model, tokenizer, device
    
    def generate_summary(
        self, 
        model, 
        tokenizer, 
        text: str, 
        device: str,
        max_length: int = 130
    ) -> str:
        """
        Generate summary for a given text.
        
        Args:
            model: Pretrained model
            tokenizer: Model tokenizer
            text (str): Input text to summarize
            device (str): Device to run inference on
            max_length (int): Maximum summary length
            
        Returns:
            str: Generated summary
        """
        # Tokenize input
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=30,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
    
    def compute_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Compute ROUGE scores.
        
        Args:
            reference (str): Reference summary
            candidate (str): Generated summary
            
        Returns:
            dict: ROUGE-1, ROUGE-2, ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(reference, candidate)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bleu(self, reference: str, candidate: str) -> float:
        """
        Compute BLEU score.
        
        Args:
            reference (str): Reference summary
            candidate (str): Generated summary
            
        Returns:
            float: BLEU score
        """
        # Tokenize by whitespace
        reference_tokens = [reference.split()]
        candidate_tokens = candidate.split()
        
        # Use smoothing to avoid zero scores
        smoothing = SmoothingFunction()
        bleu = sentence_bleu(
            reference_tokens,
            candidate_tokens,
            smoothing_function=smoothing.method1
        )
        
        return bleu
    
    def measure_latency(
        self,
        model,
        tokenizer,
        device: str,
        num_runs: int = 5
    ) -> float:
        """
        Measure average inference latency.
        
        Args:
            model: Pretrained model
            tokenizer: Model tokenizer
            device (str): Device to run inference on
            num_runs (int): Number of runs for averaging
            
        Returns:
            float: Average latency in milliseconds
        """
        latencies = []
        
        for sample in self.dataset[:num_runs]:
            start_time = time.time()
            _ = self.generate_summary(
                model,
                tokenizer,
                sample['text'],
                device,
                max_length=130
            )
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        return np.mean(latencies)
    
    def get_model_size(self, model) -> float:
        """
        Get model size in millions of parameters.
        
        Args:
            model: Pretrained model
            
        Returns:
            float: Number of parameters in millions
        """
        total_params = sum(p.numel() for p in model.parameters())
        return total_params / 1e6
    
    def get_memory_usage(self, model, device: str) -> float:
        """
        Estimate memory usage in MB.
        
        Args:
            model: Pretrained model
            device (str): Device model is on
            
        Returns:
            float: Estimated memory usage in MB
        """
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Run a dummy forward pass
            dummy_input = torch.randint(0, 1000, (1, 512)).to(device)
            with torch.no_grad():
                _ = model.generate(dummy_input, max_length=50)
            
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
            return memory_mb
        else:
            # Estimate based on parameter count for CPU
            param_size_mb = sum(
                p.numel() * p.element_size() for p in model.parameters()
            ) / 1024 / 1024
            return param_size_mb * 1.5  # Add overhead
    
    def evaluate_model(self, model_key: str, model_name: str) -> Dict:
        """
        Evaluate a single model on all metrics.
        
        Args:
            model_key (str): Model display name
            model_name (str): HuggingFace model identifier
            
        Returns:
            dict: Evaluation results
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}")
        print(f"{'='*60}")
        
        # Load model
        model, tokenizer, device = self.load_model(model_name)
        
        # Collect quality metrics
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        bleu_scores = []
        
        print(f"Generating summaries for {len(self.dataset)} samples...")
        for i, sample in enumerate(self.dataset, 1):
            # Generate summary
            summary = self.generate_summary(
                model,
                tokenizer,
                sample['text'],
                device
            )
            
            # Compute metrics
            rouge = self.compute_rouge(sample['summary'], summary)
            bleu = self.compute_bleu(sample['summary'], summary)
            
            rouge_scores['rouge1'].append(rouge['rouge1'])
            rouge_scores['rouge2'].append(rouge['rouge2'])
            rouge_scores['rougeL'].append(rouge['rougeL'])
            bleu_scores.append(bleu)
            
            if i % 3 == 0:
                print(f"  Processed {i}/{len(self.dataset)} samples...")
        
        # Compute averages
        avg_rouge1 = np.mean(rouge_scores['rouge1'])
        avg_rouge2 = np.mean(rouge_scores['rouge2'])
        avg_rougeL = np.mean(rouge_scores['rougeL'])
        avg_bleu = np.mean(bleu_scores)
        
        # Measure efficiency metrics
        print("Measuring latency...")
        latency = self.measure_latency(model, tokenizer, device)
        
        print("Computing model size...")
        model_size = self.get_model_size(model)
        
        print("Estimating memory usage...")
        memory = self.get_memory_usage(model, device)
        
        # Compile results
        results = {
            'Model': model_key,
            'ROUGE-1': round(avg_rouge1, 4),
            'ROUGE-2': round(avg_rouge2, 4),
            'ROUGE-L': round(avg_rougeL, 4),
            'BLEU': round(avg_bleu, 4),
            'Latency (ms)': round(latency, 2),
            'Size (M params)': round(model_size, 1),
            'Memory (MB)': round(memory, 1)
        }
        
        print(f"\nResults for {model_key}:")
        for key, value in results.items():
            if key != 'Model':
                print(f"  {key}: {value}")
        
        # Clean up
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return results
    
    def evaluate_all_models(self) -> pd.DataFrame:
        """
        Evaluate all configured models.
        
        Returns:
            pd.DataFrame: Results for all models
        """
        print("\n" + "="*60)
        print("STARTING MODEL EVALUATION PIPELINE")
        print("="*60)
        
        for model_key, model_name in self.models_config.items():
            try:
                result = self.evaluate_model(model_key, model_name)
                self.results.append(result)
            except Exception as e:
                print(f"Error evaluating {model_key}: {str(e)}")
                continue
        
        # Create DataFrame
        results_df = pd.DataFrame(self.results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print("\nAll Models Performance:")
        print(results_df.to_string(index=False))
        
        return results_df
    
    def save_results(self, output_path: str):
        """
        Save evaluation results to CSV.
        
        Args:
            output_path (str): Path to save CSV file
        """
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function for standalone testing."""
    import os
    
    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_path = os.path.join(project_root, 'dataset', 'sample_texts.json')
    output_path = os.path.join(project_root, 'results', 'metrics.csv')
    
    # Run evaluation
    evaluator = ModelEvaluator(dataset_path)
    results = evaluator.evaluate_all_models()
    evaluator.save_results(output_path)


if __name__ == "__main__":
    main()
