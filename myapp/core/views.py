from django.shortcuts import render
from transformers import AutoModelForCausalLM, AutoTokenizer
from django.views.decorators.csrf import csrf_exempt
import torch, json, os, re, difflib
from django.http import JsonResponse

def index(request):
    return render(request, template_name='index.html')

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

specific_responses_file = os.path.join(os.path.dirname(__file__), 'faq.json')

with open(specific_responses_file, 'r', encoding='utf-8') as f:
    specific_responses_data = json.load(f)

# ---------- Normalization ----------
_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())

def tokens(text: str) -> list:
    return normalize(text).split()

# ---------- Build base maps ----------
specific_responses = {item["input"]: item["response"] for item in specific_responses_data}
norm_q_to_q = {}
alias_to_q = {}

for item in specific_responses_data:
    q = item["input"]
    norm_q_to_q[normalize(q)] = q
    for alias in item.get("aliases", []):
        alias_to_q[normalize(alias)] = q

# ---------- Auto-alias generator (no JSON edit needed) ----------
# Extracts keywords/phrases from patterns like "what is X", "who is X", "define X", "meaning of X"
WHAT_IS_PAT = re.compile(r"^(what is|who is|define|meaning of)\s+(.+?)(\?|$)", re.IGNORECASE)
STOPWORDS = set("""
a an the what who is of for to and or in on at with from by as be are am do does did how why when where
""".split())

keyword_index = {}  # single keyword -> canonical question
phrase_index = {}   # full phrase (e.g., "cloud computing") -> canonical question

def add_keyword(k: str, q: str):
    k_norm = normalize(k)
    if not k_norm or k_norm in STOPWORDS:
        return
    # prefer first seen (stable); you can change to last-wins if needed
    keyword_index.setdefault(k_norm, q)

def index_question(q: str):
    # match "what is X" style
    m = WHAT_IS_PAT.match(q.strip())
    if m:
        phrase = m.group(2).strip()
        if phrase:
            # map full phrase and its main tokens
            phrase_index.setdefault(normalize(phrase), q)
            for t in tokens(phrase):
                if t not in STOPWORDS and len(t) >= 3:
                    add_keyword(t, q)
            return
    # generic indexing: take informative tokens from the whole question
    for t in tokens(q):
        if t not in STOPWORDS and len(t) >= 4:
            add_keyword(t, q)

for q in specific_responses.keys():
    index_question(q)

# Also fold explicit aliases into the indexes
for alias_norm, q in alias_to_q.items():
    if " " in alias_norm:
        phrase_index.setdefault(alias_norm, q)
    else:
        add_keyword(alias_norm, q)

# ---------- Small talk (short replies, handled first) ----------
SMALLTALK_PATTERNS = {
    r"^(hi|hello|hey|yo|hola|assalamu ?alaikum|salam)$": "Hi! How can I help you today?",
    r"^(good (morning|afternoon|evening))$": "Hello! How can I assist you?",
    r"^(thanks|thank you|ty|shukriya|dhonnobad)$": "You’re welcome! Anything else I can help with?",
    r"^(bye|goodbye|see ya|see you|tata)$": "Bye! If you need help later, just message me.",
    r"^(how are you|how’s it going)$": "I’m doing great! How can I assist you today?",
    r"^(who (are|r) you|what can you do)$": "I’m your support assistant—ask me a question or describe your issue.",
}

def detect_smalltalk(user_message: str):
    um = normalize(user_message)
    for pat, reply in SMALLTALK_PATTERNS.items():
        if re.fullmatch(pat, um):
            return reply
    return None

# ---------- Matchers ----------
def fuzzy_match(user_message: str, candidate_questions, cutoff=0.6):
    um = normalize(user_message)
    norm_to_orig = {normalize(q): q for q in candidate_questions}
    norm_candidates = list(norm_to_orig.keys())
    tok_count = len(tokens(user_message))
    dyn_cutoff = max(cutoff, 0.8 if tok_count <= 2 else 0.6)
    matches = difflib.get_close_matches(um, norm_candidates, n=1, cutoff=dyn_cutoff)
    return norm_to_orig[matches[0]] if matches else None

def substring_match(user_message: str, candidate_questions):
    um = normalize(user_message)
    # Allow single-word substring matches if the word is not a greeting/stopword
    if len(tokens(user_message)) <= 1 and um in STOPWORDS:
        return None
    best, best_len = None, 0
    for q in candidate_questions:
        nq = normalize(q)
        if um and (um in nq or nq in um):
            if len(nq) > best_len:
                best, best_len = q, len(nq)
    return best

def keyword_lookup(user_message: str):
    um = normalize(user_message)
    # 1) exact phrase hit (e.g., "cloud computing")
    if um in phrase_index:
        return phrase_index[um]
    # 2) single keyword hit (e.g., "python")
    if um in keyword_index:
        return keyword_index[um]
    # 3) any token hit (take first best)
    for t in tokens(user_message):
        if t in keyword_index:
            return keyword_index[t]
    return None

# ---------- Torch device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

@csrf_exempt
def chatbot_response(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get('message', '')

        # 0) Small talk
        smalltalk_reply = detect_smalltalk(user_message)
        if smalltalk_reply:
            return JsonResponse({'response': smalltalk_reply, 'correct_question': None, 'type': 'smalltalk'})

        norm_um = normalize(user_message)

        # 1) Exact normalized question
        if norm_um in norm_q_to_q:
            q = norm_q_to_q[norm_um]
            return JsonResponse({'response': specific_responses[q], 'correct_question': q})

        # 2) Explicit alias (from JSON)
        if norm_um in alias_to_q:
            q = alias_to_q[norm_um]
            return JsonResponse({'response': specific_responses[q], 'correct_question': q})

        possible_inputs = list(specific_responses.keys())

        # 3) Keyword/phrase lookup (works for single words like "python")
        q = keyword_lookup(user_message)
        if q:
            return JsonResponse({'response': specific_responses[q], 'correct_question': q})

        # 4) Substring (now allowed for non-greeting single words)
        q = substring_match(user_message, possible_inputs)
        if q:
            return JsonResponse({'response': specific_responses[q], 'correct_question': q})

        # 5) Fuzzy
        q = fuzzy_match(user_message, possible_inputs, cutoff=0.6)
        if q:
            return JsonResponse({'response': specific_responses[q], 'correct_question': q})

        # 6) Final fallback
        if len(tokens(user_message)) <= 1:
            return JsonResponse({'response': "Could you share a bit more about the issue?"})

        inputs = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
        attention_mask = torch.ones(inputs.shape, device=device)
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_length=300,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        return JsonResponse({'response': response})
