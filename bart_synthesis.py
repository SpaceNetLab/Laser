import re
import torch
from transformers import BartTokenizer, BartForConditionalGeneration


def remove_chinese_lines(text: str) -> str:
    """
    Remove lines containing Chinese characters.
    """
    pattern_chinese = re.compile(r'[\u4e00-\u9fff]')
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if not pattern_chinese.search(line):
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def bart_single_sentence(text: str,
                         model_name="facebook/bart-large-cnn",
                         min_length=20,
                         max_length=60,
                         temperature=0.7,
                         top_p=0.8,
                         num_beams=4):
    """
    Attempt to generate a single "logically better" sentence:
    1) Use sampling (do_sample=True) + top_p + temperature
    2) Limit min_length and max_length
    3) Extract the first sentence afterward

    Returns: (str) The final single sentence.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenization
    inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,  # Enable sampling
            top_p=top_p,  # Nucleus sampling
            temperature=temperature,  # Temperature
            num_beams=num_beams,  # Beam search + sampling
            min_length=min_length,
            max_length=max_length,
            no_repeat_ngram_size=3,  # Avoid simple repetition
            early_stopping=True
        )

    generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Post-processing: Extract only the first sentence
    # Use simple regex or find '.' to truncate
    # Here, split into an array of sentences and take the first one
    sentences = re.split(r'[.!?]', generated_text)
    first_sentence = sentences[0].strip()
    if first_sentence:
        # Add a period at the end to ensure semantic completeness
        return first_sentence + "."
    else:
        # If unexpectedly not truncated, return the full text
        return generated_text


if __name__ == "__main__":
    # Read your final_25_paras.txt
    input_file = "final_25_paras.txt"
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Optional: Remove lines containing Chinese characters, keep pure English
    text_no_cn = remove_chinese_lines(raw_text)

    # Call bart_single_sentence
    # These parameters can be further adjusted
    one_sentence = bart_single_sentence(
        text_no_cn,
        model_name="facebook/bart-large-cnn",
        min_length=20,
        max_length=60,
        temperature=0.7,
        top_p=0.8,
        num_beams=4
    )

    print("[One-sentence Logical Summary]")
    print(one_sentence)