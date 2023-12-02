from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline

def pegasus_summarizer():
    # Pick model
    model_name = "google/pegasus-xsum"

    # Load pretrained tokenizer
    pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)

    with open('/Users/rstridib/Desktop/Integrated/output.txt') as f:
        example_text = f.read()

    # Define PEGASUS model
    pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Create tokens
    tokens = pegasus_tokenizer(example_text, truncation=True, max_length=512, padding="max_length")

    # Define summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=pegasus_tokenizer,
        framework="pt"
    )

    summary = summarizer(example_text, min_length=100, max_length=150)
    result = summary[0]["summary_text"]

    # Write the result to the output file
    with open("result.txt", "w") as result_file:
        result_file.write(result)

    print("Result has been saved to result.txt")
    return result

pegasus_summarizer()

