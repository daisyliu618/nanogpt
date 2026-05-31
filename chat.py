"""
Interactive chat and rewrite helper for a fine-tuned nanoGPT checkpoint.

python3 chat.py --out_dir=out-shakespeare --device=mps --mode=chat

python3 chat.py --out_dir=out-shakespeare --device=mps --mode=rewrite
"""
import os
import pickle
import re
from contextlib import nullcontext

import torch
import tiktoken

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
out_dir = 'out-shakespeare'
mode = 'chat' # 'chat' or 'rewrite'
speaker = 'ROMEO'
history_turns = 6
max_new_tokens = 120
temperature = 0.7
top_k = 50
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------


def load_model():
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model)
    return model, checkpoint


def build_codec(checkpoint):
    load_meta = False
    if 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        return lambda s: [stoi[c] for c in s], lambda l: ''.join([itos[i] for i in l])

    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    return lambda s: enc.encode(s, allowed_special={"<|endoftext|>"}), lambda l: enc.decode(l)


def make_context():
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    return nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def generate(model, encode, decode, ctx, prompt):
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    text = decode(y[0].tolist())
    return text[len(prompt):] if text.startswith(prompt) else text


def first_nonempty_line(text):
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            return clean
    return text.strip()


def clean_chat_reply(text):
    match = re.search(r"\n[A-Z][A-Z '\-]{1,32}:", text)
    if match:
        text = text[:match.start()]
    for marker in ("\nHUMAN:", "\nModern:", "\nShakespearean:"):
        marker_index = text.find(marker)
        if marker_index != -1:
            text = text[:marker_index]
    return text.strip().strip('"')


def clean_rewrite(text):
    for marker in ("\nModern:", "\nShakespearean:", "\nHUMAN:", "\nROMEO:", "\nOriginal:", "\nRewrite:"):
        marker_index = text.find(marker)
        if marker_index != -1:
            text = text[:marker_index]
    rewritten = first_nonempty_line(text).strip().strip('"').strip("'")
    return rewritten.rstrip()


def chat_prompt(history):
    preamble = "The following is a conversation in the style of Shakespeare.\n\n"
    return preamble + "\n".join(history) + f"\n{speaker}:"


def rewrite_prompt(user_text):
    return (
        "Rewrite each modern sentence as one clear line in the style of Shakespeare. "
        "Keep the original meaning. Do not continue the scene.\n\n"
        "Modern: I miss you and cannot sleep tonight.\n"
        "Shakespearean: Mine heart doth ache for thee, and sleep forsakes mine eyes this night.\n\n"
        "Modern: Please forgive me for what I said.\n"
        "Shakespearean: I pray thee, pardon the words my hasty tongue hath spoken.\n\n"
        "Modern: The storm is coming soon.\n"
        "Shakespearean: The tempest draws near with swift and troubled breath.\n\n"
        "Modern: I am happy to see you again.\n"
        "Shakespearean: My heart is glad to behold thy face once more.\n\n"
        "Modern: Do not leave me alone.\n"
        "Shakespearean: Leave me not to stand alone, bereft of thy dear company.\n\n"
        "Modern: We should go before it gets dark.\n"
        "Shakespearean: Let us away ere night doth spread her sable cloak.\n\n"
        f"Modern: {user_text}\n"
        "Shakespearean:"
    )


def run_chat(model, encode, decode, ctx):
    history = []
    print(f"Chat mode. You are talking to {speaker}. Type 'quit' or 'exit' to stop.")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        history.append(f"HUMAN: {user_text}")
        prompt = chat_prompt(history[-history_turns * 2:])
        reply = clean_chat_reply(generate(model, encode, decode, ctx, prompt))
        if not reply:
            reply = "(no reply generated)"
        print(f"{speaker}: {reply}\n")
        history.append(f"{speaker}: {reply}")


def run_rewrite(model, encode, decode, ctx):
    print("Rewrite mode. Type a modern English sentence; use 'quit' or 'exit' to stop.")
    while True:
        user_text = input("Modern: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        rewritten = clean_rewrite(generate(model, encode, decode, ctx, rewrite_prompt(user_text)))
        if not rewritten:
            rewritten = "(no rewrite generated)"
        print(f"Shakespearean: {rewritten}\n")


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model, checkpoint = load_model()
encode, decode = build_codec(checkpoint)
ctx = make_context()

if mode == 'chat':
    run_chat(model, encode, decode, ctx)
elif mode == 'rewrite':
    run_rewrite(model, encode, decode, ctx)
else:
    raise ValueError("mode must be 'chat' or 'rewrite'")
