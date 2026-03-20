from transformers import AutoTokenizer, AutoModelForSequenceClassification
try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

def load_transformer(model_name, num_labels, peft_type=None):
    """
    Loads tokenizer and transformer model.
    Optionally applies LoRA if peft_type == "lora".
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # Optional 
    if peft_type == "lora" and get_peft_model is not None:
        print("Using LoRA adaptation")

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],  # attention projection layers
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )

        model = get_peft_model(model, lora_config)

        # print trainable params
        model.print_trainable_parameters()

    return tokenizer, model