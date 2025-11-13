"""
A simple example script demonstrating how to perform supervised fine‑tuning (SFT)
using the Hugging Face TRL library.  This script assumes you want to teach a
language model to generate Rust code from a specification (or any other
instruction→completion task).  The key steps are:

1.  Define a small dataset of instruction–response pairs.  Here we use the
    "prompt"/"completion" format supported by `SFTTrainer`【723152890015460†L126-L146】.  Each
    example contains the user’s specification (prompt) and the desired code
    (completion).  In a real project you would replace the examples below with
    your own curated data.

2.  Load a pre‑trained model and tokenizer.  The example uses the Qwen3‑0.6B
    model, but you can substitute any causal language model compatible with
    Hugging Face transformers.  If you wish to use LoRA/QLoRA for parameter
    efficient fine‑tuning, you can provide a `LoraConfig` (see below)【776457794560684†L111-L141】.

3.  Construct an `SFTTrainer` with appropriate training arguments.  You can
    configure batch size, number of epochs, sequence length, learning rate,
    gradient accumulation, and more via `SFTConfig`.  For LoRA fine‑tuning,
    supply a `peft_config` argument.  When using LoRA, only a small set of
    adapter weights are updated, reducing memory usage and preserving the base
    model【776457794560684†L111-L141】.

4.  Call `trainer.train()` to perform the fine‑tuning and then save the
    resulting model and tokenizer.  After training, you can load the model
    adapter for inference.
"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Optional: LoRA configuration for parameter‑efficient fine‑tuning
try:
    from peft import LoraConfig
except ImportError:
    LoraConfig = None  # If peft is not installed, set this to None


def build_dataset() -> Dataset:
    """
    Build a prompt/completion dataset with diverse Verus examples.
    Each entry contains a "text" field that combines the prompt and completion.
    
    Returns:
        A Hugging Face ``Dataset`` containing the examples.
    """
    examples = [
        {
            "text": "Add Verus specs to this function:\n```rust\nfn abs(x: i32) -> i32 {\n    if x < 0 { -x } else { x }\n}\n```\n```verus\nfn abs(x: i32) -> i32 {\n    requires true;\n    ensures |result| == |x| && (result == x || result == -x);\n    if x < 0 { -x } else { x }\n}\n```",
        },
        {
            "text": "Generate Rust code that satisfies: requires x > 0, ensures return == x + 1.\nfn add_one(x: i32) -> i32 {\n    requires x > 0;\n    ensures result == x + 1;\n    x + 1\n}",
        },
        {
            "text": "Add Verus specs to this max function:\n```rust\nfn max(a: i32, b: i32) -> i32 {\n    if a > b { a } else { b }\n}\n```\n```verus\nfn max(a: i32, b: i32) -> i32 {\n    ensures result >= a && result >= b;\n    ensures result == a || result == b;\n    if a > b { a } else { b }\n}\n```",
        },
        {
            "text": "Write a Verus function that doubles a number:\n```verus\nfn double(x: i32) -> i32 {\n    requires x < i32::MAX / 2;\n    ensures result == 2 * x;\n    x * 2\n}\n```",
        },
        {
            "text": "Add Verus specification for array bounds checking:\n```rust\nfn get_first(arr: &[i32]) -> Option<i32> {\n    if arr.len() > 0 {\n        Some(arr[0])\n    } else {\n        None\n    }\n}\n```\n```verus\nfn get_first(arr: &[i32]) -> Option<i32> {\n    requires true;\n    ensures arr.len() > 0 ==> result == Some(arr[0]);\n    ensures arr.len() == 0 ==> result == None;\n    if arr.len() > 0 {\n        Some(arr[0])\n    } else {\n        None\n    }\n}\n```",
        },
        {
            "text": "Create a Verus function for subtraction:\n```verus\nfn subtract(a: i32, b: i32) -> i32 {\n    requires a >= b;\n    ensures result == a - b;\n    ensures result >= 0;\n    a - b\n}\n```",
        },
        {
            "text": "Add Verus specs for division:\n```rust\nfn divide(a: u32, b: u32) -> u32 {\n    a / b\n}\n```\n```verus\nfn divide(a: u32, b: u32) -> u32 {\n    requires b > 0;\n    ensures result == a / b;\n    ensures result <= a;\n    a / b\n}\n```",
        },
        {
            "text": "Write a Verus function that checks if a number is positive:\n```verus\nfn is_positive(x: i32) -> bool {\n    ensures result == (x > 0);\n    x > 0\n}\n```",
        },
        {
            "text": "Add Verus specs for minimum function:\n```rust\nfn min(a: i32, b: i32) -> i32 {\n    if a < b { a } else { b }\n}\n```\n```verus\nfn min(a: i32, b: i32) -> i32 {\n    ensures result <= a && result <= b;\n    ensures result == a || result == b;\n    if a < b { a } else { b }\n}\n```",
        },
        {
            "text": "Create a Verus function for squaring:\n```verus\nfn square(x: i32) -> i32 {\n    requires x.abs() < 46341;  // sqrt(i32::MAX)\n    ensures result == x * x;\n    ensures result >= 0;\n    x * x\n}\n```",
        },
    ]
    return Dataset.from_list(examples)


def main():
    # Load your dataset
    train_dataset = build_dataset()

    # Choose a base model.  gpt2 is used here for demonstration.
    model_name = "gpt2"

    # Load tokenizer and model.  Use ``tokenizer`` to map strings to token
    # sequences and ``model`` to initialize the pre‑trained weights.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token for GPT-2 which doesn't have one by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Define training configuration.  Adjust hyperparameters according to your
    # compute resources.  ``max_seq_length`` should be large enough to fit
    # both prompt and completion; ``per_device_train_batch_size`` and
    # ``gradient_accumulation_steps`` should be set to reach your desired
    # effective batch size.
    config = SFTConfig(
        output_dir="./sft_output",
        per_device_train_batch_size=2,  # Increased batch size
        num_train_epochs=10,  # More epochs for better learning
        max_seq_length=1024,
        learning_rate=3e-4,  # Slightly higher learning rate
        gradient_accumulation_steps=2,
        logging_steps=5,  # Log every 5 steps
        save_strategy="epoch",  # Save checkpoint each epoch
        save_total_limit=2,  # Keep only 2 best checkpoints
    )

    # Optionally configure LoRA for parameter‑efficient fine‑tuning.  If you
    # provide a LoraConfig, only the adapter weights will be updated.  This
    # reduces memory usage and speeds up training on consumer GPUs【776457794560684†L111-L141】.
    lora_config = None
    if LoraConfig is not None:
        lora_config = LoraConfig(
            r=16,  # Increased rank for more capacity
            lora_alpha=32,  # Scaled with r
            target_modules=["c_attn", "c_proj"],  # Target both attention and projection layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # Initialize the SFTTrainer.  ``train_dataset`` must be a Hugging Face
    # ``Dataset`` with a "text" field containing the training examples.
    # The trainer will tokenize these fields for training.
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=config,
    )

    # Fine‑tune the model on your dataset.  This call will perform the
    # training loop and update either the full model or just the LoRA adapter.
    trainer.train()

    # Save the fine‑tuned model (and LoRA adapter, if applicable).  After
    # training completes, you can load the weights from ``config.output_dir`` to
    # generate new code/specifications from your trained model.
    trainer.model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()