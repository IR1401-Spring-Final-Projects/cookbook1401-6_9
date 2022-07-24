from transformers import BertConfig, BertTokenizer, BertModel

model_name = None
config = None
tokenizer = None
model = None


def load_pars_bert():
    global model, tokenizer, config, model_name
    device = "cpu"
    # v3.0
    model_name = "HooshvareLab/bert-fa-base-uncased"  # "HooshvareLab/bert-fa-zwnj-base"
    config = BertConfig.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)
    model.eval()

def get_model():
    return model

def get_tokenizer():
    return tokenizer
