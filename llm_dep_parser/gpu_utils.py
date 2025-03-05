import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from llm_dep_parser.text_utils import (
    format_prompt,
    extract_json,
    format_prompt_icl,
)


def load_llm_on_gpu(gpu_id, model_name):
    """Load the LLM on the assigned GPU with efficient settings."""
    print(f"Loading LLM on GPU {gpu_id}...")

    torch.cuda.set_device(gpu_id)  # Ensure correct GPU is used

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Enable 8-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model on the assigned GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )


def process_on_gpu(
    model_name,
    gpu_id,
    texts,
    batch_size,
    is_icl=False,
    labels_df_fp="",
    matrix_fp="",
    vectorizer_fp="",
):
    """Process text in parallel on the assigned GPU"""
    torch.cuda.set_device(gpu_id)  # Ensure correct GPU usage

    nlp_model = load_llm_on_gpu(gpu_id, model_name)  # Load once per process

    # Generate prompts in bulk (vectorized processing)
    if is_icl:
        prompts = [
            format_prompt_icl(text, vectorizer_fp=vectorizer_fp, matrix_fp=matrix_fp, path_to_labels_df=labels_df_fp)
            for text in texts
        ]
    else:
        prompts = [format_prompt(text) for text in texts]

    # Process all prompts at once in parallel
    responses = nlp_model(
        prompts, max_length=1600, do_sample=False, batch_size=batch_size
    )

    results = []
    for prompt, response in zip(prompts, responses):
        try:
            full_response = response[0]["generated_text"]
        except (IndexError, KeyError):
            full_response = "ERROR: LLM response missing"

        results.append((prompt, full_response,))

    return results
