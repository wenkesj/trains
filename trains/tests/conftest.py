import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def processor():
    raw = Tokenizer(
        WordLevel({"[UNK]": 0, "user": 1, "go": 2, "assistant": 3, "north": 4}, unk_token="[UNK]")
    )
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{m.role + ' ' + m.content + ' '}}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    return tokenizer
