from transformers import  BertTokenizer

def preprocess(review):
    maxlen=512
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    # Use BERT tokenizer since it needs to be able to match the tokens to the pre trained words.
    encoded = tokenizer.encode_plus(
    text=review,  # the sentence to be encoded
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length = maxlen,  # maximum length of a sentence
    truncation=True,
    pad_to_max_length=True,  # Add [PAD]s
    return_attention_mask = True,  # Generate the attention mask
    return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )

    tokens_ids_tensor = encoded['input_ids']
    attn_mask = encoded['attention_mask']
    return tokens_ids_tensor.unsqueeze(dim=0),attn_mask.unsqueeze(dim=0)