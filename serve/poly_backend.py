"""Optional PolyStrata architectures; the original server remains the protocol owner."""
from pathlib import Path
from serve import frontend

IMAGE_PAD = "<|image_pad|>"
VISION_START = "<|vision_start|>"
TURN_OPENERS = {"function": ("<|im_start|>",), "gemma": ("<|turn>",),
                "glm": ("<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"),
                "dsml": ("<｜User｜>", "<｜Assistant｜>"), "lfm2": ("<|im_start|>",)}


def is_poly(cfg):
    return Path(str(cfg.get("exe") or "")).stem.lower() in ("strata-poly", "strata-glm")


def configure(service, tokenizer):
    pre = getattr(tokenizer, "pre", "")
    service.call_format = "function"
    service.think_open_ids = []
    service.template_tokens = {}
    service.image_marks = (IMAGE_PAD, VISION_START if VISION_START in getattr(service.template, "source", VISION_START) else None, None)
    if pre not in ("glm4", "glm5", "gemma4", "joyai-llm", "deepseek-v3", "lfm2"):
        return
    service.literals = frontend.literal_tags(getattr(tokenizer, "control_tokens", ()))
    im_end = tokenizer.encode("<|im_end|>", parse_special=True)
    if len(im_end) == 1:
        service.stop_ids = set(im_end + tokenizer.encode("<|endoftext|>", parse_special=True))
    else:   # not a ChatML model (GLM): the ids its GGUF names for the end of the text, the turn and the message
        special = getattr(tokenizer, "special_ids", None) or {}
        service.stop_ids = {special[k] for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                              "tokenizer.ggml.eom_token_id") if k in special}
    # the tokenizer's own tokens under the names a template knows them by (Gemma's starts with bos_token)
    special = getattr(tokenizer, "special_ids", None) or {}
    service.template_tokens = {name: tokenizer.decode([special[key]]) for name, key in (
        ("bos_token", "tokenizer.ggml.bos_token_id"), ("eos_token", "tokenizer.ggml.eos_token_id"))
        if key in special}
    # how the model writes a tool call (serve/frontend.py's OutputParser)
    pre = getattr(tokenizer, "pre", "")
    service.call_format = "glm" if pre in ("glm4", "glm5") else "gemma" if pre == "gemma4" else \
        "dsml" if pre in ("joyai-llm", "deepseek-v3") else "lfm2" if pre == "lfm2" else "function"
    service.turn_ids = frozenset(t[0] for t in (tokenizer.encode(s, parse_special=True)
                                             for s in TURN_OPENERS[service.call_format]) if len(t) == 1)
    service.engine.turn_ids = service.turn_ids            # the GEN line's turn_at (StrataEngine.turn_key)
    service.think_open_ids = []
    if service.call_format == "gemma":
        # after its calls Gemma opens the tools' answer, which is the client's to give: the reply ends there.
        # Its template opens the reasoning for it after a tool's answer; run() looks for that at the prompt's end
        answer = tokenizer.encode("<|tool_response>", parse_special=True)
        service.stop_ids |= set(answer if len(answer) == 1 else [])
        # the end of its turn: a finetune's GGUF can name only <eos> (HauhauCS 1M: eos 1, no eot), and the reply
        # then runs on past <turn|> until max_tokens (llama.cpp ends it by the token's text, like this)
        turn_end = tokenizer.encode("<turn|>", parse_special=True)
        service.stop_ids |= set(turn_end if len(turn_end) == 1 else [])
        service.think_open_ids = tokenizer.encode("<|channel>thought\n", parse_special=True)
    # an image in a prompt (prepare()): the token the template writes for it, which stands for each of its rows;
    # the token the template writes before that one, if it writes one; the tokens to put around the rows, if
    # the template leaves that to the program
    # A template that writes the marker alone (MiniCPM-V 4.7's, like Gemma 4's) has no token before it to tell
    # an image's marker from the same text in a message: the first ones are the images (prepare()).
    service.image_marks = ("<|image|>", None, ("<|image>", "<image|>")) if service.call_format == "gemma" else \
        (IMAGE_PAD, VISION_START if VISION_START in getattr(service.template, "source", VISION_START) else None, None)


def render(service, messages, tools, kwargs):
    if not messages:
        raise ValueError("No messages provided.")
    off = kwargs.get("enable_thinking") is False
    kwargs = {**service.template_tokens, **kwargs, "enable_thinking": not off}
    if service.call_format in ("gemma", "dsml") and tools:
        tools = [{"type": "function", "function": t} for t in tools]
    prompt = service.template.render(messages, tools=tools, **kwargs)
    if service.call_format == "gemma":
        return frontend.gemma_model_turn(prompt, not off)
    return prompt + "</think>" if off and prompt.endswith("<think>") else prompt
