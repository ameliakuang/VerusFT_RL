"""
Simple inference script to test the fine-tuned Verus code generation model.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_model(model_path="./sft_output"):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"Loading model from {model_path}...")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("Model loaded successfully!")
    return model, tokenizer


def generate_code(model, tokenizer, prompt, max_length=300):
    """Generate Verus code from a prompt."""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    # Load model
    model, tokenizer = load_model()
    
    # Test prompts
    test_prompts = [
        "Add Verus specs to this function:\n```rust\nfn clamp(x: i32, min: i32, max: i32) -> i32 {\n    if x < min { min } else if x > max { max } else { x }\n}\n```\n",
        "Write a Verus function that multiplies two numbers:\n",
        "Add Verus specs for minimum function:\n```rust\nfn min(a: i32, b: i32) -> i32 {\n    if a < b { a } else { b }\n}\n```\n",
    ]
    
    print("\n" + "="*80)
    print("Testing Fine-Tuned Verus Code Generator")
    print("="*80 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}:")
        print(f"{'='*80}")
        print(f"Prompt:\n{prompt}")
        print(f"\n{'-'*80}")
        print("Generated Code:")
        print(f"{'-'*80}")
        
        generated = generate_code(model, tokenizer, prompt)
        # Only show the generated part (after the prompt)
        generated_only = generated[len(prompt):]
        print(generated_only)
        print()


if __name__ == "__main__":
    main()



