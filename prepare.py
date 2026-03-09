#!/usr/bin/env python3
"""
Dataset Preparation Script for CosmicText
Processes datasets ONE AT A TIME sequentially with HuggingFace authentication.
Tokenizes on-the-fly and creates train.bin/val.bin files directly.

Dataset Plan:
- Wikipedia (250M) + OpenWebText (250M) = 500M text tokens
- Conceptual Captions (100M) + LAION COCO (100M) = 200M image caption tokens
- Total: ~700M tokens across 4 datasets
"""

import os
import sys
import time
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import tiktoken
from termcolor import colored
from dataclasses import dataclass
from huggingface_hub import login

HF_TOKEN = "HF_token_here"

tokenizer = tiktoken.get_encoding("gpt2")
EOT_TOKEN = tokenizer.eot_token

@dataclass
class DatasetConfig:
    name: str
    dataset_id: str
    config: Optional[str] = None
    revision: Optional[str] = None
    target_tokens: int = 0
    use_streaming: bool = False
    columns: List[str] = None
    min_tokens: int = 50
    test_size: float = 0.0005
    is_caption: bool = False

def print_status(message: str, status: str = "info") -> None:
    colors = {"info": "cyan", "success": "green", "error": "red", "warning": "yellow", "progress": "magenta"}
    print(colored(f"[{status.upper()}] {message}", colors.get(status, "white")))

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("<|endoftext|>", "")
    text = text.replace(chr(50256), "")
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def chunk_and_tokenize_text(text: str, max_tokens: int = 512, min_chunk_tokens: int = 50) -> List[List[int]]:
    if not text.strip():
        return []
    try:
        tokens = tokenizer.encode(text)
    except:
        return []
    if len(tokens) <= max_tokens:
        return [tokens] if len(tokens) >= min_chunk_tokens else []
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        if len(chunk_tokens) >= min_chunk_tokens:
            chunks.append(chunk_tokens)
    return chunks


class DatasetProcessor:

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.authenticate_huggingface()

        self.dataset_configs = [
            DatasetConfig(
                name="wikipedia",
                dataset_id="wikimedia/wikipedia",
                config="20231101.en",
                target_tokens=250_000_000,
                use_streaming=False,
                columns=["title", "text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="openwebtext",
                dataset_id="vietgpt/openwebtext_en",
                target_tokens=250_000_000,
                use_streaming=True,
                columns=["text"],
                min_tokens=50,
                test_size=0.0005
            ),
            DatasetConfig(
                name="conceptual_captions",
                dataset_id="google-research-datasets/conceptual_captions",
                target_tokens=100_000_000,
                use_streaming=True,
                columns=["caption"],
                min_tokens=5,
                test_size=0.0005,
                is_caption=True
            ),
            DatasetConfig(
                name="laion_coco",
                dataset_id="laion/laion-coco",
                target_tokens=100_000_000,
                use_streaming=True,
                columns=["TEXT"],
                min_tokens=5,
                test_size=0.0005,
                is_caption=True
            ),
        ]

        self.total_tokens_processed = 0
        self.dataset_stats = {}

    def authenticate_huggingface(self):
        try:
            login(token=HF_TOKEN)
            print_status("HuggingFace authentication successful", "success")
        except Exception as e:
            print_status(f"HuggingFace authentication failed: {e}", "warning")
            print_status("Continuing without authentication - may hit rate limits", "warning")

    def extract_text_from_item(self, item: Dict, config: DatasetConfig) -> Optional[str]:
        try:
            texts = []
            for column in config.columns:
                if column in item and item[column]:
                    if isinstance(item[column], str):
                        texts.append(item[column])
                    else:
                        texts.append(str(item[column]))
            if not texts:
                return None
            combined_text = "\n".join(texts)
            return clean_text(combined_text)
        except:
            return None

    def flush_tokens_to_temp_file(self, tokens: List[int], temp_dir: Path, file_counter: int) -> str:
        temp_file = temp_dir / f"temp_{file_counter:06d}.bin"
        np.array(tokens, dtype=np.uint16).tofile(temp_file)
        return str(temp_file)

    def combine_temp_files_to_final(self, temp_files: List[str], output_file: Path):
        print_status(f"Combining {len(temp_files)} temp files into {output_file.name}...", "info")
        with open(output_file, 'wb') as outf:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, 'rb') as inf:
                        outf.write(inf.read())
                    os.remove(temp_file)

    def process_single_dataset(self, config: DatasetConfig, dataset_number: int, total_datasets: int) -> Dict:
        print_status("="*80, "info")
        print_status(f"DATASET {dataset_number}/{total_datasets}: {config.name.upper()} {'[CAPTIONS]' if config.is_caption else '[TEXT]'}", "info")
        print_status(f"Target: {config.target_tokens/1e6:.0f}M tokens", "info")
        print_status(f"Streaming: {'Yes' if config.use_streaming else 'No'}", "info")
        print_status(f"Min Tokens: {config.min_tokens}", "info")
        print_status(f"Val Split: {config.test_size*100:.2f}%", "info")
        print_status("="*80, "info")

        dataset_dir = self.output_dir / config.name
        dataset_dir.mkdir(exist_ok=True)

        temp_dir = dataset_dir / "temp"
        temp_dir.mkdir(exist_ok=True)

        print_status(f"Loading {config.name} from {config.dataset_id}...", "info")
        try:
            load_args = {
                "path": config.dataset_id,
                "split": "train"
            }
            if config.config:
                load_args["name"] = config.config
            if config.revision:
                load_args["revision"] = config.revision
            if config.use_streaming:
                load_args["streaming"] = True

            dataset = load_dataset(**load_args)
            print_status(f"Successfully loaded {config.name}", "success")
        except Exception as e:
            print_status(f"Failed to load {config.name}: {e}", "error")
            return {"tokens": 0, "chunks": 0, "examples": [], "error": str(e)}

        tokens_written = 0
        chunks_written = 0
        items_processed = 0
        examples_collected = []
        accumulated_tokens = []
        temp_files = []
        file_counter = 0
        max_tokens_in_memory = 50_000_000

        start_time = time.time()
        print_status(f"Processing {config.name}...", "info")

        if config.use_streaming:
            pbar = tqdm(desc=f"Processing {config.name}", unit="items",
                       bar_format='{l_bar}{bar}| {n_fmt} items [{elapsed}, {rate_fmt}] Tokens: {postfix}')

            for item in dataset:
                if tokens_written >= config.target_tokens:
                    break

                text = self.extract_text_from_item(item, config)
                if not text:
                    continue

                token_chunks = chunk_and_tokenize_text(text, min_chunk_tokens=config.min_tokens)

                for chunk_tokens in token_chunks:
                    if len(chunk_tokens) > 0:
                        chunk_tokens_with_eot = chunk_tokens + [EOT_TOKEN]
                        accumulated_tokens.extend(chunk_tokens_with_eot)
                        tokens_written += len(chunk_tokens_with_eot)
                        chunks_written += 1

                        if len(examples_collected) < 3:
                            try:
                                example_text = tokenizer.decode(chunk_tokens[:100])
                                examples_collected.append(example_text + "..." if len(chunk_tokens) > 100 else example_text)
                            except:
                                pass

                        if len(accumulated_tokens) >= max_tokens_in_memory:
                            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
                            temp_files.append(temp_file)
                            file_counter += 1
                            accumulated_tokens = []

                        if tokens_written >= config.target_tokens:
                            break

                items_processed += 1
                if items_processed % 100 == 0:
                    pbar.update(100)
                    pbar.set_postfix_str(f"{tokens_written/1e6:.1f}M/{config.target_tokens/1e6:.0f}M")

            pbar.close()

        else:
            total_items = len(dataset)
            pbar = tqdm(dataset, desc=f"Processing {config.name}", total=total_items,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] Tokens: {postfix}')

            for item in pbar:
                if config.target_tokens > 0 and tokens_written >= config.target_tokens:
                    break

                text = self.extract_text_from_item(item, config)
                if not text:
                    continue

                token_chunks = chunk_and_tokenize_text(text, min_chunk_tokens=config.min_tokens)

                for chunk_tokens in token_chunks:
                    if len(chunk_tokens) > 0:
                        chunk_tokens_with_eot = chunk_tokens + [EOT_TOKEN]
                        accumulated_tokens.extend(chunk_tokens_with_eot)
                        tokens_written += len(chunk_tokens_with_eot)
                        chunks_written += 1

                        if len(examples_collected) < 3:
                            try:
                                example_text = tokenizer.decode(chunk_tokens[:100])
                                examples_collected.append(example_text + "..." if len(chunk_tokens) > 100 else example_text)
                            except:
                                pass

                        if len(accumulated_tokens) >= max_tokens_in_memory:
                            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
                            temp_files.append(temp_file)
                            file_counter += 1
                            accumulated_tokens = []

                        if config.target_tokens > 0 and tokens_written >= config.target_tokens:
                            break

                items_processed += 1
                pbar.set_postfix_str(f"{tokens_written/1e6:.1f}M")

            pbar.close()

        if accumulated_tokens:
            temp_file = self.flush_tokens_to_temp_file(accumulated_tokens, temp_dir, file_counter)
            temp_files.append(temp_file)

        processing_time = time.time() - start_time

        print_status(f"Creating train/val split for {config.name}...", "info")

        val_size = int(tokens_written * config.test_size)
        train_size = tokens_written - val_size

        val_tokens_needed = val_size
        train_temp_files = []
        val_temp_files = []

        current_val_tokens = 0
        for temp_file in temp_files:
            if current_val_tokens < val_tokens_needed:
                val_temp_files.append(temp_file)
                file_size = os.path.getsize(temp_file)
                estimated_tokens = file_size // 2
                current_val_tokens += estimated_tokens
            else:
                train_temp_files.append(temp_file)

        train_path = dataset_dir / "train.bin"
        val_path = dataset_dir / "val.bin"

        if val_temp_files:
            self.combine_temp_files_to_final(val_temp_files, val_path)
        if train_temp_files:
            self.combine_temp_files_to_final(train_temp_files, train_path)

        try:
            temp_dir.rmdir()
        except:
            pass

        metadata = {
            "dataset_name": config.name,
            "dataset_id": config.dataset_id,
            "config": config.config,
            "revision": config.revision,
            "is_caption": config.is_caption,
            "target_tokens": config.target_tokens,
            "actual_tokens": tokens_written,
            "train_tokens": train_size,
            "val_tokens": val_size,
            "chunks_written": chunks_written,
            "items_processed": items_processed,
            "processing_time_seconds": processing_time,
            "processing_time_human": f"{processing_time/60:.1f} minutes",
            "tokens_per_second": tokens_written / processing_time if processing_time > 0 else 0,
            "completed_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "use_streaming": config.use_streaming,
            "min_tokens": config.min_tokens,
            "test_size": config.test_size,
            "files_created": ["train.bin", "val.bin"]
        }

        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print_status(f"{config.name} COMPLETED", "success")
        print_status(f"  Tokens : {tokens_written/1e6:.1f}M / {config.target_tokens/1e6:.0f}M ({(tokens_written/config.target_tokens)*100:.1f}%)", "success")
        print_status(f"  Chunks : {chunks_written:,}", "success")
        print_status(f"  Train  : {train_size/1e6:.1f}M tokens -> train.bin", "success")
        print_status(f"  Val    : {val_size/1e6:.1f}M tokens -> val.bin", "success")
        print_status(f"  Time   : {processing_time/60:.1f} minutes", "success")
        print_status(f"  Speed  : {(tokens_written/1e6)/(processing_time/60):.1f}M tokens/min", "success")

        return {
            "tokens": tokens_written,
            "train_tokens": train_size,
            "val_tokens": val_size,
            "chunks": chunks_written,
            "examples": examples_collected,
            "metadata": metadata,
            "items_processed": items_processed,
            "processing_time": processing_time
        }

    def wait_between_datasets(self, seconds: int = 10):
        print_status(f"Waiting {seconds} seconds before next dataset...", "warning")
        for i in range(seconds, 0, -1):
            print(f"\rCountdown: {i} seconds", end="", flush=True)
            time.sleep(1)
        print("\r" + " " * 20 + "\r", end="")

    def generate_examples_file(self):
        examples_file = self.output_dir / "examples.txt"

        with open(examples_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EXAMPLES FROM COSMICTEXT DATASET\n")
            f.write("="*80 + "\n\n")

            for dataset_name, stats in self.dataset_stats.items():
                f.write(f"=== {dataset_name.upper()} ===\n")
                f.write(f"Tokens: {stats['tokens']/1e6:.1f}M | Chunks: {stats['chunks']:,} | Time: {stats.get('processing_time', 0)/60:.1f}min\n")
                f.write(f"Train: {stats.get('train_tokens', 0)/1e6:.1f}M | Val: {stats.get('val_tokens', 0)/1e6:.1f}M\n")
                f.write("-" * 60 + "\n")

                examples = stats.get('examples', [])
                for i, example in enumerate(examples[:3], 1):
                    f.write(f"Example {i}:\n{example}\n\n")

                f.write("="*80 + "\n\n")

        print_status(f"Generated examples.txt with samples from all {len(self.dataset_stats)} datasets", "success")

    def generate_summary_report(self):
        total_tokens = sum(stats['tokens'] for stats in self.dataset_stats.values())
        total_train_tokens = sum(stats.get('train_tokens', 0) for stats in self.dataset_stats.values())
        total_val_tokens = sum(stats.get('val_tokens', 0) for stats in self.dataset_stats.values())
        total_chunks = sum(stats['chunks'] for stats in self.dataset_stats.values())
        total_time = sum(stats.get('processing_time', 0) for stats in self.dataset_stats.values())
        target_total = 700_000_000

        report_file = self.output_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COSMICTEXT DATASET PREPARATION SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"TOTAL TOKENS PROCESSED: {total_tokens/1e6:.1f}M\n")
            f.write(f"TRAIN TOKENS: {total_train_tokens/1e6:.1f}M\n")
            f.write(f"VALIDATION TOKENS: {total_val_tokens/1e6:.1f}M\n")
            f.write(f"TOTAL CHUNKS CREATED: {total_chunks:,}\n")
            f.write(f"TARGET ACHIEVEMENT: {(total_tokens/target_total)*100:.1f}%\n")
            f.write(f"TOTAL PROCESSING TIME: {total_time/3600:.1f} hours\n")
            f.write(f"AVERAGE SPEED: {(total_tokens/1e6)/(total_time/60):.1f}M tokens/minute\n\n")

            f.write("DATASET RESULTS:\n")
            f.write("-" * 70 + "\n")

            for i, (dataset_name, stats) in enumerate(self.dataset_stats.items(), 1):
                tokens = stats['tokens']
                train_tokens = stats.get('train_tokens', 0)
                val_tokens = stats.get('val_tokens', 0)
                proc_time = stats.get('processing_time', 0)
                is_caption = stats.get('metadata', {}).get('is_caption', False)
                tag = "[CAPTIONS]" if is_caption else "[TEXT]    "
                f.write(f"{i:2d}. {dataset_name:25s} {tag}: {tokens/1e6:6.1f}M total ({train_tokens/1e6:6.1f}M train, {val_tokens/1e3:6.1f}K val) [{proc_time/60:5.1f}min]\n")

            f.write(f"\nCATEGORY BREAKDOWN:\n")
            f.write("-" * 30 + "\n")

            categories = {
                "Core Text (target: 500M)": ["wikipedia", "openwebtext"],
                "Image Captions (target: 200M)": ["conceptual_captions", "laion_coco"]
            }

            for category, datasets in categories.items():
                f.write(f"\n{category}:\n")
                category_tokens = 0
                for ds in datasets:
                    if ds in self.dataset_stats:
                        tokens = self.dataset_stats[ds]['tokens']
                        chunks = self.dataset_stats[ds]['chunks']
                        category_tokens += tokens
                        f.write(f"  {ds}: {tokens/1e6:.1f}M tokens ({chunks:,} chunks)\n")
                f.write(f"  Category Total: {category_tokens/1e6:.1f}M tokens\n")

            f.write(f"\nFORMAT: Tokenized binary files (train.bin + val.bin per dataset)\n")
            f.write(f"TOKENIZER: GPT-2 encoding (vocab_size: {tokenizer.n_vocab})\n")
            f.write(f"Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model target: CosmicText\n")

        print_status(f"Generated summary report: {total_tokens/1e6:.1f}M tokens total", "success")

        print_status("="*80, "info")
        print_status("ALL DATASETS PROCESSED", "success")
        print_status(f"Total tokens : {total_tokens/1e6:.1f}M / 700M ({(total_tokens/target_total)*100:.1f}%)", "info")
        print_status(f"Train        : {total_train_tokens/1e6:.1f}M tokens", "info")
        print_status(f"Val          : {total_val_tokens/1e6:.1f}M tokens", "info")
        print_status(f"Total time   : {total_time/3600:.1f} hours", "info")
        print_status(f"Output dir   : {self.output_dir}", "info")
        print_status("="*80, "info")

    def run(self):
        print_status("Starting CosmicText Dataset Preparation", "info")
        print_status(f"Output directory: {self.output_dir}", "info")
        print_status(f"Target: ~700M tokens across {len(self.dataset_configs)} datasets", "info")
        print_status(f"  Text    : Wikipedia (250M) + OpenWebText (250M) = 500M", "info")
        print_status(f"  Captions: Conceptual Captions (100M) + LAION COCO (100M) = 200M", "info")
        print_status(f"Processing: One dataset at a time (sequential)", "info")
        print_status(f"Output format: train.bin + val.bin per dataset", "info")
        print_status("="*80, "info")

        overall_start_time = time.time()

        for i, config in enumerate(self.dataset_configs, 1):
            try:
                if i > 1:
                    self.wait_between_datasets(10)

                stats = self.process_single_dataset(config, i, len(self.dataset_configs))

                if stats['tokens'] > 0:
                    self.dataset_stats[config.name] = stats
                    self.total_tokens_processed += stats['tokens']
                    print_status(f"Running total: {self.total_tokens_processed/1e6:.1f}M tokens ({(self.total_tokens_processed/700e6)*100:.1f}% of 700M)", "progress")
                else:
                    print_status(f"{config.name} produced 0 tokens - skipping", "warning")

                print()

            except Exception as e:
                print_status(f"Error processing {config.name}: {e}", "error")
                print_status("Continuing with next dataset...", "warning")
                continue

        print_status("Generating final reports...", "info")
        self.generate_examples_file()
        self.generate_summary_report()

        overall_elapsed_time = time.time() - overall_start_time
        print_status(f"Total processing time: {overall_elapsed_time/3600:.1f} hours", "info")
        print_status(f"Average speed: {(self.total_tokens_processed/1e6)/(overall_elapsed_time/60):.1f}M tokens/minute", "info")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare ~700M token dataset for CosmicText training")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--test_mode", action="store_true")
    parser.add_argument("--start_from", type=str, default=None)

    args = parser.parse_args()

    processor = DatasetProcessor(args.output_dir)

    if args.test_mode:
        print_status("Running in TEST MODE - reduced targets", "warning")
        for config in processor.dataset_configs:
            config.target_tokens = max(100_000, config.target_tokens // 1000)

    if args.start_from:
        dataset_names = [config.name for config in processor.dataset_configs]
        if args.start_from in dataset_names:
            start_index = dataset_names.index(args.start_from)
            processor.dataset_configs = processor.dataset_configs[start_index:]
            print_status(f"Resuming from dataset: {args.start_from}", "warning")
        else:
            print_status(f"Dataset '{args.start_from}' not found. Available: {dataset_names}", "error")
            sys.exit(1)

    processor.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_status("\nProcessing interrupted by user", "warning")
        print_status("Resume with: --start_from DATASET_NAME", "info")
        sys.exit(1)
    except Exception as e:
        print_status(f"\nFatal error: {e}", "error")
        import traceback
        traceback.print_exc()
        sys.exit(1)