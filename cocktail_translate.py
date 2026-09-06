"""Cocktail — 번역 계층.

번역 모델(opus-mt / m2m100 하이브리드), 메모리 LRU + DPAPI 영구 캐시,
언어 스크립트 판정, 표시 직전 게이트(DG-1), 배치 번역.
OCR·UI를 import 하지 않는다.

라이선스: opus-mt-tc-big-en-ko는 **CC-BY-4.0**(출처 표시 의무).
LICENSE-3RDPARTY.md의 저작자 표기를 배포물에 동봉해야 한다.
"""
import os
import re
import sys
import threading
import time
import unicodedata
from collections import OrderedDict

from cocktail_platform import dpapi_protect, dpapi_unprotect


# --- PV-2 튜닝 레버: 로그에 원문을 남기지 않는다 ------------------------------
# 진단 로그(DG-1 숨김 / RP-2 재번역)가 **원문 머리 60자를 그대로** 찍고 있었다.
# 은행·로그인 화면이 지나갈 수 있는 앱에서 화면 글자가 콘솔이나 리다이렉트된 로그
# 파일(`debug_log.txt`)에 남으면 안 된다.
# 무음 실패로 돌아가지는 않는다(CO-2 교훈) — **무엇이 왜 걸렸는가**(사유·길이)는 그대로
# 남기고, "어떤 줄인가"만 해시로 바꾼다. 같은 줄은 같은 해시라 반복 억제도 그대로 된다.
# 원문이 필요한 개발자는 `COCKTAIL_LOG_TEXT=1` 로 켠다(그 환경에서만 평문이 찍힌다).
LOG_PLAINTEXT = os.environ.get("COCKTAIL_LOG_TEXT", "").strip() == "1"


def log_text(text) -> str:
    """로그에 넣어도 되는 텍스트 식별자. 진단 모드에서만 원문을 돌려준다."""
    s = _cache_key_text(text or "")
    if LOG_PLAINTEXT:
        return s[:60]
    import hashlib
    return f"{len(s)}자 #{hashlib.sha1(s.encode('utf-8')).hexdigest()[:8]}"


# --- RP-1 튜닝 레버: 신경망 번역 반복 붕괴 방지 ------------------------------
# 잡음 입력(OCR 쓰레기)에서 seq2seq가 같은 구를 무한 반복하는 고전적 붕괴를 막는다.
# 3은 한국어 정상 문장(어미·조사 반복)에도 걸릴 수 있어 4를 기본으로 둔다.
NO_REPEAT_NGRAM_SIZE = 4          # ↓3이면 더 강하게 억제(정상 문장 왜곡 위험), ↑5면 관대

# --- RP-2 튜닝 레버: 붕괴한 조각만 빔 서치로 다시 뽑기 ------------------------
# RP-1(no_repeat_ngram)은 **무한 반복**만 막는다. 변주를 섞은 반복은 그대로 통과한다:
#     "The kettle on the far shelf had been whistling for a while before anyone noticed it."
#   → "그 때, 그 때, 한 때의 그 시절, 그 시절, 그때의 그 시절을 떠올리는 사람이 있었다."
#   (OCR CER 0.00% = 원문은 완벽했다. greedy 디코딩이 혼자 무너진 것이다.)
# 실측(2026-09-07, 영어 1,940문장 · en→ko):
#   - 원인은 **greedy**다. 같은 문장을 `num_beams=4`로 뽑으면 정상 번역이 나온다.
#     `no_repeat_ngram_size=0`으로 두면 오히려 "그 때," 무한 루프가 되므로 RP-1은 유지한다.
#     한 글자만 달라도(`for a while`→`for 2 while`) 붕괴가 사라진다 = 입력 특성이 아니라 탐색 실패.
#   - 빈도 8/1940 = **0.41%**. OCR 오독을 주입한 600문장에서는 0/600 — 붕괴는
#     "쓰레기 입력" 현상이 **아니라** 깨끗한 입력에서 나는 탐색 실패다.
#   - 전면 `num_beams=4`는 배치 12문장 p50 311ms → 553ms(**+78%**)라 기준 4를 깬다
#     (현재 p50 1.46s / 한도 1.5s). 그래서 **붕괴 징후가 있는 조각만** 다시 뽑는다.
# 판정값: 글자만 남긴 문자 3-gram 중복률의 "원문 대비 초과분"(`repetition_excess`).
#   원문 자신이 반복이면(`----------`, `:py:data:` 나열) 번역문도 반복이라 그건 빼야 한다.
#   실측 분포: 정상 p99 = +0.064, 붕괴 = +0.10 ~ +0.33.
RETRY_REPETITION_EXCESS = 0.10    # ↓면 더 많이 재시도(느려짐), ↑면 붕괴를 놓친다
RETRY_NUM_BEAMS = 4               # 재시도 빔 수. 실측에서 4가 관측된 붕괴를 전부 고쳤다
RETRY_MAX_PIECES = 8              # 한 프레임에서 재시도할 조각 수 상한(최악 지연 방어).
                                  # 실측 발생률 0.4%라 8이면 사실상 항상 충분하다.


# --- DG-1 튜닝 레버: 표시 직전 최종 게이트 (7차) ------------------------------
# 원칙: "확신 없으면 숨긴다". 깨진 자막을 보여주는 것보다 안 보여주는 게 낫다.
# NZ-1(원문 문자 구성)은 OCR이 "말이 되는 오독"을 하면 뚫린다(초소형 캡션 →
# "마우 냄비 thang에"류 그럴싸한 쓰레기). 번역까지 끝난 뒤 마지막으로 한 번 더 거른다.
DISPLAY_GATE_ENABLED = True            # False면 게이트 전체 무효(디버그용)
# DG-1b(2026-09-05): 70 → 60. 70은 근거 없이 잡은 값이었고, 채점판에서 **정상 줄만** 잡았다.
# DG-1c(2026-09-07): "극소 라인 **이면서** 저 conf" 였던 조건을 **conf 단독**으로 바꿨다.
#   라인 높이는 1차 OCR이 만든 값이라 1차가 무너지면 같이 오염된다 — 실측: 10px Calibri 를
#   28px 로 보고해서(OU-2와 같은 뿌리) 이 규칙이 영원히 안 걸렸고, conf 52.0 짜리
#   **틀린 자막**이 그대로 떴다. 게이트가 원문을 못 보고 번역문만 보는 정책 위반이었다.
#   높이는 못 믿어도 conf 는 이미지에서 나온 값이라 믿을 수 있다.
# 임계 실측(bench 시료 전체를 앱 경로로 태워 정답 유사도 0.80 으로 정상/깨짐 분류):
#     정상 줄 78개  min 64.0(다크 UI "4 spaces") · p05 86.1 · 중앙값 95.9
#     깨진 줄       43.7 / 52.0 / 58.4 / 62.6 / 63.9 (Calibri·Impact·Georgia·Times 10px)
#   conf<60 → 정상 숨김 0/78, conf<65 → 1/78(=1.3%, 기준 2의 오탐 한도 1% 초과).
#   그래서 60이 상한이다. 이 위로 올리지 마라 — 오탐(정상 줄 숨김)은 무음 실패라 더 비싸다.
# ↓면 깨진 자막이 새어나온다. Georgia/Times 급(62~64)은 conf 로는 못 막는다 —
# 그건 OU-2(확대 트리거)가 애초에 제대로 읽게 해서 막는다.
DISPLAY_SRC_MIN_CONF = 60.0            # 원문 OCR 평균 conf 가 이 미만이면 그 줄은 안 그린다
DISPLAY_MIN_TGT_SCRIPT_RATIO = 0.3     # 번역문 글자 중 목표 언어 스크립트 비율 하한 (한국어 목표인데 한글 극소 = 실패)
DISPLAY_LEN_COLLAPSE_MIN_SRC = 20      # 원문이 이 길이 이상일 때만 길이 붕괴 검사
DISPLAY_LEN_COLLAPSE_RATIO = 0.15      # 번역문/원문 길이 하한. 한국어는 보통 0.5배 이상이라 0.15는 보수적

# --- CO-1: 미번역 잔류 검사 ---------------------------------------------------
# 번역 모델은 모르는 토큰을 **그대로 복사**한다. OCR이 "grants"를 "arants"로 읽으면
# 그 단어는 사전에 없으니 출력에 영어 그대로 남는다. 실측(게임 툴팁, 2026-08-05):
#     "4회씩 각 dealing 156 피해 (4.25 첨부 speed) (3.55 쿨다운,"
# 한글 비율이 0.3을 넘으니 규칙 2(tgt-script)를 통과한다. 그래서 별도 규칙이 필요하다.
# 정책: 반쯤 번역된 헛소리를 보여주느니 원문을 그대로 보게 둔다.
DISPLAY_CARRYOVER_MIN_LEN = 3          # 이 길이 이상 라틴 단어만 센다 (of/to 같은 건 무시)
DISPLAY_MAX_CARRYOVER_WORDS = 0        # **짧은 줄**의 절대 허용치. 고유명사·약어는 대문자로
                                       # 시작/구성돼 애초에 안 센다("HP", "Ss", "Ur" 통과).
                                       # 대가: 'github', 'npm' 처럼 **소문자 고유명사**가 든
                                       # 정상 번역도 숨는다. 그런 화면을 자주 본다면 1~2로 ↑.
# --- CO-2: 절대 허용치는 문단에 쓰면 안 된다 ---------------------------------
# CO-1의 0은 게임 툴팁 **한 줄**(30~60자) 기준으로 잡은 값인데, 그 뒤 PM-1(문단 병합)이
# 게이트의 판정 단위를 200~350자 문단으로 키웠다. 40단어 문단에서 OCR이 단어 하나를 틀리면
# (긴 희귀어일수록 잘 틀린다) 번역기가 그 한 단어를 그대로 복사하고, 나머지 39단어가 멀쩡해도
# **문단 전체가 사라진다**. 실측(어린왕자 4쪽, 2026-08-11) — 1글자 오독 주입 8종 중 3종이
# 문단 통째 숨김을 유발했다:
#     constrictor→consttictor : carryover(constitctor) → 346자 문단 전멸
#     elephant   →elephanl    : carryover(elephanl)    → 346자 문단 전멸
#     disheartened→disheattened: carryover(disheattened) → 204자 문단 전멸
# → 허용치를 **번역문 길이에 비례**시킨다. 짧은 줄은 지금과 똑같이 무관용(0),
#   긴 문단은 단어 한둘까지 견딘다. 게임 툴팁 fixture는 그대로 걸린다(10어절/잔류 2 > 허용 1).
DISPLAY_CARRYOVER_MAX_RATIO = 0.12     # 허용치 = max(절대치, 번역문 어절 수 × 이 비율).
                                       # 9어절 미만=0, 9~16=1, 17~25=2 … ↓면 더 엄격

# 목표 언어별로 "번역이 실제로 일어났다면 나와야 하는" 스크립트.
_LANG_TO_SCRIPTS = {
    "ko": {"HANGUL"},
    "ja": {"HIRAGANA", "KATAKANA", "CJK"},
    "zh": {"CJK"},
    "ru": {"CYRILLIC"},
    "en": {"LATIN"},
    "fr": {"LATIN"},
    "de": {"LATIN"},
}


# --- 번역 백엔드: HybridTranslator (M-5) -----------------------------------
# LZ-1: torch/transformers는 **import 시점에 끌어오지 않는다**. 실측(2026-09-05)
#   import cocktail_ui 10.24s 중 torch 2.3s + transformers 2.0s(그 안에서 torch._dynamo
#   2.0s + sklearn.metrics 1.4s를 또 끌고 온다). 사람이 기다리는 건 창이 뜨는 시간이고,
#   모델은 어차피 `BackgroundController._preload_model` 백그라운드 스레드가 로드한다.
# 그 스레드(또는 첫 translate)가 `_ensure_torch()`를 부르는 순간 진짜 import가 일어난다.
torch = None            # _ensure_torch() 이후 실제 모듈
device = None           # 같음. import 전에 읽으면 None이다
USE_FP16 = False
_LOAD_DTYPE = None
_TORCH_LOCK = threading.Lock()


def _ensure_torch():
    """torch를 지금 import하고 device/dtype 전역을 채운다 (LZ-1). 두 번째부터는 즉시 반환."""
    global torch, device, USE_FP16, _LOAD_DTYPE
    if torch is not None:
        return torch
    with _TORCH_LOCK:
        if torch is not None:
            return torch
        import torch as _torch
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        USE_FP16 = device.type == "cuda"
        # VR-1: 로드 시점에 바로 fp16으로 올린다. `from_pretrained() → .to(cuda) → .half()` 순서는
        # fp32 사본을 먼저 VRAM에 올려 **최종 크기의 2배**를 순간적으로 점유한다(m2m100_418M 기준
        # 약 1.9GB 피크 → 0.98GB). en-ko + zh-en 이 이미 떠 있는 상태에서 m2m100 폴백을 올리다
        # CUDA OOM(16 MiB 요청 실패)이 난 실측(2026-09-05)이 있어 피크를 반으로 줄인다.
        _LOAD_DTYPE = _torch.float16 if USE_FP16 else _torch.float32
        print(f"[INFO] device={device}, fp16={USE_FP16}")
        torch = _torch   # 마지막에 대입 — 다른 스레드가 "준비 완료"로 보는 순간이다
        return torch


# --- PK-2: 배포본에 동봉된 모델 --------------------------------------------
# 기본 언어쌍(en→ko, safetensors 399MB)만 동봉한다. 첫 실행에 1.6GB를 받게 하면
# "준비 5분"(합격기준 6)이 회선 상태에 걸리고, 오프라인이 강점인 앱이 첫 실행에만
# 인터넷을 요구하게 된다. 나머지(zh-en 300MB, m2m100 폴백 1.9GB)는 그 언어를 실제로
# 번역할 때만 받는다 — 전부 넣으면 인스톨러가 3GB를 넘는다.
# 배치: `<앱폴더>/models/<모델 leaf 이름>/` (build.spec 이 채운다).
_SEARCH_DIRS = [d for d in (os.path.dirname(os.path.abspath(
    (sys.argv[0] if sys.argv else "") or ".")), getattr(sys, "_MEIPASS", None)) if d]


def _bundled_model_dir(model_name: str):
    """동봉된 모델 폴더 경로. 없으면 None(= HF 캐시/네트워크 경로 그대로)."""
    leaf = model_name.split("/")[-1]
    for d in _SEARCH_DIRS:
        p = os.path.join(d, "models", leaf)
        if os.path.isfile(os.path.join(p, "config.json")):
            return p
    return None


def _load_local_first(loader, model_name, **kwargs):
    """LD-1: 로컬 HF 캐시 우선 로드.

    `local_files_only=True`로 먼저 시도해 HF Hub 네트워크 왕복(캐시가 있어도 수십 초
    걸리던 rate-limit/파일 목록 확인)을 건너뛴다. 캐시가 없거나 불완전하면 예외가 나므로
    그때만 네트워크 경로로 재시도한다 — 첫 설치는 그대로 동작.
    `HF_HUB_OFFLINE` 같은 전역 오프라인 강제는 첫 설치를 망가뜨리므로 쓰지 않는다.
    PK-2: 동봉본이 있으면 그 폴더에서 바로 읽는다(캐시 복사 없음).
    """
    model_name = _bundled_model_dir(model_name) or model_name
    try:
        return loader(model_name, local_files_only=True, **kwargs)
    except Exception as e:
        print(f"[INFO] 로컬 캐시 미스 → 네트워크 로드 ({model_name}): {type(e).__name__}")
        return loader(model_name, **kwargs)


class _BaseTranslator:
    name = "base"
    def translate_batch(self, texts, src, tgt, num_beams=1):
        raise NotImplementedError


class OpusMtTranslator(_BaseTranslator):
    """
    Helsinki-NLP/opus-mt-* — CC-BY-4.0 (출처 표시 의무). 언어쌍별 전용 모델.
    en→ko 등 자주 쓰는 쌍에서 m2m100/NLLB 대비 작고 빠르고 품질 우수.
    """
    name = "opus-mt"

    def __init__(self, model_name: str, progress_cb=None):
        print(f"[INFO] opus-mt 로드 중: {model_name}")
        if progress_cb: progress_cb("번역 엔진 초기화 중")
        _ensure_torch()   # LZ-1
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        if progress_cb: progress_cb("토크나이저 로드 중")
        # LD-1: 로컬 캐시 우선 (네트워크 체크로 로드가 수십 초 걸리던 문제)
        self.tok = _load_local_first(AutoTokenizer.from_pretrained, model_name)
        if progress_cb: progress_cb("모델 다운로드/로드 중")
        m = _load_local_first(AutoModelForSeq2SeqLM.from_pretrained, model_name,
                              torch_dtype=_LOAD_DTYPE).to(device)   # VR-1
        m.eval()
        self.model = m
        self.src_sp = self._load_source_spm(model_name)  # M-8
        if progress_cb: progress_cb("워밍업 중")
        try:
            with torch.inference_mode():
                enc = self.tok(["hello"], return_tensors="pt", padding=True).to(device)
                self.model.generate(**enc, max_new_tokens=8, num_beams=1)
        except Exception as e:
            print(f"[WARN] opus-mt 워밍업 실패: {e}")

    def _load_source_spm(self, model_name):
        """M-8: sepvoc(소스/타깃 vocab 분리) opus-mt 모델은 repo의 tokenizer_config.json이
        separate_vocabs=false로 잘못 적혀 있어 AutoTokenizer가 소스 문장을 타깃 vocab으로
        인코딩한다(→ 대부분 <unk>, 번역 결과가 쓰레기). source.spm으로 직접 인코딩한다.
        공유 vocab 모델이면 None(기존 경로 유지), 로드 실패해도 None으로 degrade."""
        try:
            import sentencepiece as spm
            from huggingface_hub import snapshot_download
            # LD-1: 로컬 스냅샷 우선 — 캐시가 있어도 매번 14개 파일 목록을 Hub에 묻던 경로.
            # PK-2: 동봉본은 그 폴더가 곧 스냅샷이다(snapshot_download는 repo id만 받는다).
            snapshot_dir = (_bundled_model_dir(model_name)
                            or _load_local_first(snapshot_download, model_name))
            sp = spm.SentencePieceProcessor(
                model_file=os.path.join(snapshot_dir, "source.spm")
            )
            vocab = self.tok.get_vocab()
            size = sp.get_piece_size()
            missing = sum(1 for i in range(0, size, 16) if sp.id_to_piece(i) not in vocab)
            if missing * 16 < size * 0.2:
                return None  # 소스 조각이 토크나이저 vocab에 있음 = 공유 vocab, 기존 경로가 정상
            print(f"[INFO] sepvoc 모델 감지, source.spm 직접 인코딩 (M-8): {model_name}")
            return sp
        except Exception as e:
            print(f"[WARN] source.spm 로드 실패, 기존 토크나이저 경로 사용 (M-8): {e}")
            return None

    def _encode_with_spm(self, texts):
        pad = self.tok.pad_token_id
        seqs = [self.src_sp.encode(t)[:255] + [self.tok.eos_token_id] for t in texts]
        n = max(len(s) for s in seqs)
        return {
            "input_ids": torch.tensor(
                [s + [pad] * (n - len(s)) for s in seqs], dtype=torch.long).to(device),
            "attention_mask": torch.tensor(
                [[1] * len(s) + [0] * (n - len(s)) for s in seqs], dtype=torch.long).to(device),
        }

    def translate_batch(self, texts, src, tgt, num_beams=1):
        # opus-mt는 언어쌍 전용이라 src/tgt는 라우팅 검증용으로만 사용 (모델은 이미 결정됨)
        if self.src_sp is not None:
            enc = self._encode_with_spm(texts)  # M-8
        else:
            enc = self.tok(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).to(device)
        in_len = enc["input_ids"].shape[1]
        max_new = min(256, int(in_len * 1.6) + 16)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=num_beams,           # RP-2: 붕괴 조각 재시도만 >1
                do_sample=False,
                early_stopping=num_beams > 1,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # RP-1
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)


class M2M100Translator(_BaseTranslator):
    """
    facebook/m2m100_418M — MIT. any-to-any 100언어 폴백.
    opus-mt에 등록 안 된 쌍에서만 호출됨.
    """
    name = "m2m100"

    def __init__(self, model_name: str = "facebook/m2m100_418M"):
        print(f"[INFO] m2m100 로드 중: {model_name}")
        _ensure_torch()   # LZ-1
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
        # LD-1: 로컬 캐시 우선
        self.tok = _load_local_first(M2M100Tokenizer.from_pretrained, model_name)
        m = _load_local_first(M2M100ForConditionalGeneration.from_pretrained, model_name,
                              torch_dtype=_LOAD_DTYPE).to(device)   # VR-1
        m.eval()
        self.model = m
        try:
            with torch.inference_mode():
                self.tok.src_lang = "en"
                enc = self.tok(["hello"], return_tensors="pt", padding=True).to(device)
                self.model.generate(
                    **enc,
                    forced_bos_token_id=self.tok.get_lang_id("ko"),
                    max_new_tokens=8,
                    num_beams=1,
                )
        except Exception as e:
            print(f"[WARN] m2m100 워밍업 실패: {e}")

    def translate_batch(self, texts, src, tgt, num_beams=1):
        # src/tgt = ISO 639-1 ("en", "ko", "fr", ...)
        self.tok.src_lang = src
        enc = self.tok(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(device)
        in_len = enc["input_ids"].shape[1]
        max_new = min(256, int(in_len * 1.6) + 16)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                forced_bos_token_id=self.tok.get_lang_id(tgt),
                max_new_tokens=max_new,
                num_beams=num_beams,           # RP-2
                do_sample=False,
                early_stopping=num_beams > 1,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,  # RP-1
            )
        return self.tok.batch_decode(out, skip_special_tokens=True)


class HybridTranslator:
    """
    M-5: 라우팅
      - BILINGUAL_MODELS에 등록된 (src, tgt) → opus-mt (작고 빠름)
      - 그 외 → m2m100 (any-to-any 폴백)
    P-9: lazy load — __init__은 가볍고, 모델은 첫 사용/preload 시 로드.
    """
    name = "hybrid"

    # ISO 639-1 (src, tgt) → opus-mt 모델. 자주 쓰는 쌍을 늘리려면 여기 등록 (라이선스 확인 필수).
    BILINGUAL_MODELS = {
        ("en", "ko"): "Helsinki-NLP/opus-mt-tc-big-en-ko",
        ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",   # M-9: zh→ko 피벗의 앞다리
        # 예: ("ko", "en"): "Helsinki-NLP/opus-mt-tc-big-ko-en",
    }

    # --- M-9: 전용 모델이 없는 쌍의 중간 언어 경유 -----------------------------
    # 중국어→한국어 전용 opus-mt 는 **없다**(2026-08-11 확인: Helsinki-NLP/opus-mt-zh-ko,
    # opus-mt-tc-big-zh-ko 둘 다 RepositoryNotFound). 그래서 이 쌍은 m2m100_418M
    # 폴백으로 흘러가 있었는데, 실측에서 주어를 통째로 흘렸다.
    #     "小美站在河边…" → "**은** 강 옆에 서서…"   (小美 증발)
    #     "小美看看四周…" → "**이 이** 주위를 둘러보고…"
    #
    # 후보 실측 (중국어 12문장 → 한국어, chrF, RTX GPU / fp16, 2026-08-11):
    #     m2m100_418M 직행 (현행)                       chrF 24.5   582 ms   MIT
    #     shun89/opus-mt-zh-ko (3rd party 파인튜닝)     chrF 28.3   294 ms   apache-2.0(업로더 신고)
    #     **zh→en(opus-mt-zh-en) → en→ko(기존 모델)**   chrF 32.0   497 ms   CC-BY-4.0  ← 채택
    #     opus-mt-tc-bible-big-mul-mul                  chrF  4.5   434 ms   apache-2.0
    # 채택 근거:
    #   - 품질 1위. 2위(3rd party)보다 +3.7, 현행보다 +7.5.
    #   - 라이선스가 이미 배포 중인 en-ko 와 같은 CC-BY-4.0 (Helsinki 공식) — 상용 가능.
    #     shun89 는 라이선스를 apache-2.0 으로 신고했지만 베이스가 CC-BY-4.0 opus-mt 라
    #     신고가 신뢰되지 않고, 학습 데이터도 문서화돼 있지 않다(다운로드 175회).
    #   - mul-mul 은 성경 코퍼스만 학습해 쓸 수 없다(출력이 고문체 한국어, 때때로 과라니어).
    #   - NLLB 계열은 CC-BY-NC(비상업)라 애초에 후보가 아니다 — M-5/E-10.
    # 비용: 중국어 화면에서만 모델 하나(312MB)와 generate 1회가 추가된다. 영어 화면은
    # 이 경로를 밟지 않으므로 그대로다.
    # 올리는 길: 양쪽 다리에 num_beams=4 → chrF 35.0 (810 ms). 품질 +3, 시간 +63%.
    PIVOT_ROUTES = {("zh", "ko"): "en"}

    def __init__(self):
        self._bilingual = {}        # (src, tgt) -> OpusMtTranslator
        self._multilingual = None   # M2M100Translator (lazy)
        self._load_lock = threading.Lock()
        self._loading = set()       # P-9b: 백그라운드 로드 중인 (src, tgt)
        self._m2m_loading = False   # P-9b: m2m100 폴백 백그라운드 로드 중

    def _ensure_bilingual(self, src, tgt, progress_cb=None, block=True):
        key = (src, tgt)
        if key in self._bilingual:
            return self._bilingual[key]
        model_name = self.BILINGUAL_MODELS.get(key)
        if not model_name:
            return None
        if not block:
            # P-9b: 워커 스레드에서 첫 로드(다운로드 포함 수십 초)를 기다리면 화면이 통째로
            # 언다. 로드는 백그라운드로 던지고 지금은 None — 호출부는 그동안 폴백을 쓴다.
            if key not in self._loading:
                self._loading.add(key)
                threading.Thread(target=self._background_load, args=(src, tgt),
                                 daemon=True).start()
            return None
        with self._load_lock:
            if key not in self._bilingual:
                self._bilingual[key] = OpusMtTranslator(model_name, progress_cb=progress_cb)
            return self._bilingual[key]

    def _background_load(self, src, tgt):
        try:
            self._ensure_bilingual(src, tgt)
        except Exception as e:
            # 실패를 기억하지 않는다 — 다음 프레임에 다시 시도한다. 그동안은 폴백이 돈다.
            self._loading.discard((src, tgt))
            print(f"[WARN] {src}->{tgt} 모델 로드 실패, m2m100 폴백 유지: {e}")

    def _ensure_multilingual(self, block=True):
        if self._multilingual is not None:
            return self._multilingual
        if not block:
            # P-9b: m2m100 첫 로드는 실측 4.0초. 워커 스레드에서 기다리면 그 프레임의
            # **모든** 줄이 그동안 안 뜬다. 로드는 백그라운드로 던지고 지금은 None —
            # 호출부는 이번 프레임만 이 언어 그룹을 포기한다(다음 프레임엔 준비돼 있다).
            if not self._m2m_loading:
                self._m2m_loading = True
                threading.Thread(target=self._background_multilingual, daemon=True).start()
            return None
        with self._load_lock:
            if self._multilingual is None:
                self._multilingual = M2M100Translator("facebook/m2m100_418M")
            return self._multilingual

    def _background_multilingual(self):
        try:
            self._ensure_multilingual()
        except Exception as e:
            print(f"[WARN] m2m100 로드 실패: {e}")
        finally:
            self._m2m_loading = False   # 실패를 기억하지 않는다 — 다음 프레임에 다시 시도

    def preload_default(self, progress_cb=None):
        """en→ko opus-mt 미리 로드 (백그라운드 호출용, P-9)."""
        self._ensure_bilingual("en", "ko", progress_cb=progress_cb)
        if progress_cb: progress_cb("준비 완료")

    def translate_batch(self, texts, src, tgt, num_beams=1):
        direct = self._ensure_bilingual(src, tgt)
        if direct is not None:
            return direct.translate_batch(texts, src, tgt, num_beams)
        # M-9: 전용 모델이 없으면 중간 언어 경유를 먼저 시도한다(m2m100 폴백보다 낫다).
        # 두 다리 다 준비돼 있을 때만 — 아직 로딩 중이면 이번 프레임은 폴백이 처리한다.
        mid = self.PIVOT_ROUTES.get((src, tgt))
        if mid:
            first = self._ensure_bilingual(src, mid, block=False)
            second = self._ensure_bilingual(mid, tgt, block=False)
            if first is not None and second is not None:
                return second.translate_batch(
                    first.translate_batch(texts, src, mid, num_beams), mid, tgt, num_beams)
        fallback = self._ensure_multilingual(block=False)
        if fallback is None:
            raise RuntimeError("m2m100 폴백 모델 로드 중")   # TE-1: 이 그룹만 이번 프레임 포기
        return fallback.translate_batch(texts, src, tgt, num_beams)


# 가벼운 인스턴스만 만들기 (모델은 아직 로드 안 됨, P-9)
TRANSLATOR = HybridTranslator()


# --- 번역 캐시 (P-5) --------------------------------------------------------
class LRU(OrderedDict):
    def __init__(self, capacity=512):
        super().__init__()
        self.capacity = capacity

    def get_or_none(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        while len(self) > self.capacity:
            self.popitem(last=False)


TRANS_CACHE = LRU(capacity=1024)
CACHE_LOCK = threading.Lock()


# --- 언어 코드 (ISO 639-1) — M-5: NLLB BCP47에서 변경 ---------------------
LANG_MAP = {
    "Korean":   "ko",
    "English":  "en",
    "French":   "fr",
    "German":   "de",
    "Chinese":  "zh",
    "Japanese": "ja",
}

# DC-1: `M2M100_LANGS`(100개 언어 집합)는 정의만 되고 어디서도 참조되지 않아 삭제했다.
# _SCRIPT_TO_LANG이 내놓는 코드(ko/ja/zh/ar/ru/he/hi/th/el/bn/pa/gu/ta/ml/km/lo/my)는
# 전부 m2m100이 아는 코드라 별도 필터가 필요 없었다.

_SCRIPT_TO_LANG = {
    "HANGUL": "ko",
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "CJK": "zh",
    "ARABIC": "ar",
    "CYRILLIC": "ru",
    "HEBREW": "he",
    "DEVANAGARI": "hi",
    "THAI": "th",
    "GREEK": "el",
    "BENGALI": "bn",
    "GURMUKHI": "pa",
    "GUJARATI": "gu",
    "TAMIL": "ta",
    "MALAYALAM": "ml",
    "KHMER": "km",
    "LAO": "lo",
    "MYANMAR": "my",
}

def _unicode_script(ch: str) -> str:
    """OCR 문자열의 Unicode script를 대략 분류한다. 숫자/기호는 COMMON으로 무시."""
    if not ch or not ch.strip():
        return "COMMON"
    name = unicodedata.name(ch, "")
    cat = unicodedata.category(ch)
    if not cat.startswith("L"):
        return "COMMON"
    if "HANGUL" in name:
        return "HANGUL"
    if "HIRAGANA" in name:
        return "HIRAGANA"
    if "KATAKANA" in name:
        return "KATAKANA"
    if "CJK UNIFIED" in name or "CJK COMPATIBILITY" in name:
        return "CJK"
    for script in _SCRIPT_TO_LANG:
        if script in name:
            return script
    if "LATIN" in name:
        return "LATIN"
    if cat.startswith("L"):
        return "OTHER"
    return "COMMON"


def detect_script(text: str) -> tuple[str, dict[str, int]]:
    """문자열의 지배적인 Unicode script를 반환한다. 라틴은 SL-1에서 무조건 en."""
    counts = {}
    for ch in text or "":
        script = _unicode_script(ch)
        if script == "COMMON":
            continue
        counts[script] = counts.get(script, 0) + 1
    if not counts:
        return "UNKNOWN", counts

    # 일본어는 CJK 한자와 kana가 섞이는 경우가 많으므로 kana가 보이면 ja로 본다.
    if counts.get("HIRAGANA", 0) or counts.get("KATAKANA", 0):
        return "HIRAGANA" if counts.get("HIRAGANA", 0) >= counts.get("KATAKANA", 0) else "KATAKANA", counts

    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    if dominant == "LATIN":
        non_latin = {k: v for k, v in counts.items() if k != "LATIN"}
        if non_latin:
            script, n = max(non_latin.items(), key=lambda kv: kv[1])
            total = sum(counts.values())
            if script in _SCRIPT_TO_LANG and (n >= 2 or n / total >= 0.35):
                return script, counts
    return dominant, counts


def identify_language(text: str, fallback: str = "en") -> str:
    """
    SL-1: 문자 스크립트로 확정되는 언어만 인정한다(한글→ko, 가나→ja, CJK→zh, 키릴→ru…).
    라틴·불명 스크립트는 추측하지 않고 전부 fallback(기본 en) — 이 한 줄이
    "영어 화면을 cy/hu/de/pt/so로 오판 → m2m100 폴백 → 5초 배치" 사슬을 끊는다.
    다른 라틴 언어(프랑스어 등)를 읽어야 하면 사용자가 src 콤보에서 명시한다.
    """
    script, _ = detect_script(text)
    return _SCRIPT_TO_LANG.get(script, fallback)


def assign_region_languages(ocr_results, src_lang_hint):
    """
    src=auto면 라인마다 결정론적 스크립트 판정만 적용한다(SL-1).
    사용자가 source 언어를 명시하면 그대로 강제한다.
    (M-7 region 투표 / LG-1 지배 언어 스냅은 라틴 추측을 보정하려고 있던 장치라
     추측이 사라진 지금은 존재 이유가 없어 함께 퇴역.)
    """
    if src_lang_hint != "auto":
        return [src_lang_hint] * len(ocr_results)
    return [identify_language(text) for text, _, _ in ocr_results]


def display_gate_reject(src_text, tgt_text, tgt_lang, line_px=None, conf=None):
    """DG-1 표시 게이트의 **단일 진입점**. 사유를 돌려주면 호출부는 그 줄을 안 그린다.

    CO-2: 숨김은 무음 실패다 — 사용자에게는 "그 문단만 자막이 없다"로만 보이고 로그엔
    아무것도 안 남아 원인 추적이 불가능했다. 여기서 사유 + 줄 식별자를 한 번 찍는다.
    (라인당 1회. 정적 화면이 초당 몇 번씩 같은 줄을 재판정해도 로그는 늘지 않는다.)
    PV-2: 식별자는 길이+해시다 — 원문은 `COCKTAIL_LOG_TEXT=1` 일 때만 찍힌다.
    """
    reason = _display_gate_reason(src_text, tgt_text, tgt_lang, line_px, conf)
    if reason:
        _log_gate_reject(reason, src_text)
    return reason


_GATE_LOG_SEEN = OrderedDict()   # (사유 종류, 줄 식별자) — 같은 줄 반복 로그 억제
_GATE_LOG_CAP = 64


def _log_gate_reject(reason, src_text):
    ident = log_text(src_text)
    key = (reason.split("(")[0], ident)
    if key in _GATE_LOG_SEEN:
        return
    _GATE_LOG_SEEN[key] = None
    while len(_GATE_LOG_SEEN) > _GATE_LOG_CAP:
        _GATE_LOG_SEEN.popitem(last=False)
    print(f"[INFO] DG-1 숨김: {reason} [{ident}]")


def _display_gate_reason(src_text, tgt_text, tgt_lang, line_px=None, conf=None):
    """DG-1 (7차): 표시 직전 최종 게이트. 거부 사유 문자열을 반환하면 그 줄은 **그리지 않는다**.

    NZ-1은 OCR 원문의 문자 구성만 본다. 초소형 캡션(7px대)에서 OCR이 "말이 되는 오독"을
    하면(예: "마우 냄비 thang에") 원문도 번역문도 형태상 멀쩡해 보여 전부 통과한다.
    그래서 번역이 끝난 뒤 "이 줄을 믿을 근거가 있는가"를 마지막으로 한 번 더 묻는다.

    규칙은 넷 다 **거짓 양성(정상 줄을 숨김)이 나기 어려운 쪽**으로 잡았다:
      1. 낮은 원문 confidence — 넷 중 **유일하게 원문을 보는 규칙**(DG-1c). 나머지 셋은
         번역문만 보므로, 원문이 그럴듯하게 깨지면 번역문도 그럴듯해서 전부 통과한다.
      2. 목표 언어 스크립트 비율 — 한국어 목표인데 번역문에 한글이 거의 없다 = 번역이 실패했다.
      3. 길이 붕괴 — 긴 원문이 초단문으로 뭉개진 경우(모델이 입력을 버린 것).
      4. 미번역 잔류(CO-1) — 허용치는 번역문 길이에 비례한다(CO-2).

    `line_px`는 더 이상 판정에 쓰지 않는다 — 1차 OCR이 만든 값이라 1차가 무너지면
    같이 오염된다(DG-1c). 인자는 호출부 계약 유지를 위해 남겨 둔다.
    """
    if not DISPLAY_GATE_ENABLED:
        return None
    tgt = (tgt_text or "").strip()
    src = (src_text or "").strip()
    if not tgt:
        return "empty"

    # 1) 원문 confidence 가 낮다 = 이 줄을 제대로 읽었다는 근거가 없다.
    #    확대 재OCR(OU-1/OU-2)로도 못 살린 영역이 여기 걸린다.
    if conf is not None and 0 <= conf < DISPLAY_SRC_MIN_CONF:
        return f"low-src-conf({conf:.0f})"

    # 2) 번역문이 목표 언어 스크립트를 거의 안 담고 있으면 번역 자체가 안 일어난 것.
    wanted = _LANG_TO_SCRIPTS.get(tgt_lang)
    if wanted:
        scripts = [s for s in (_unicode_script(ch) for ch in tgt) if s != "COMMON"]
        if not scripts:
            return "no-letters"
        hit = sum(1 for s in scripts if s in wanted)
        if hit / len(scripts) < DISPLAY_MIN_TGT_SCRIPT_RATIO:
            return f"tgt-script({hit}/{len(scripts)})"

    # 3) 원문과 무관한 초단문(길이 붕괴).
    if (len(src) >= DISPLAY_LEN_COLLAPSE_MIN_SRC
            and len(tgt) < len(src) * DISPLAY_LEN_COLLAPSE_RATIO):
        return f"len-collapse({len(tgt)}/{len(src)})"

    # 4) CO-1: 번역문에 라틴 단어가 그대로 남아 있으면 절반만 번역된 것.
    #    CO-2: 허용치는 번역문 길이에 비례한다 — 문단 하나에 오독 단어 하나가 섞였다고
    #    멀쩡한 40단어를 통째로 숨기면 그건 게이트의 실패다.
    if wanted and "LATIN" not in wanted:
        left = carryover_words(tgt)
        if left and len(left) > carryover_allowance(tgt):
            return f"carryover({','.join(left[:3])})"
    return None


def carryover_allowance(tgt_text):
    """CO-2: 이 번역문에서 눈감아 줄 미번역 라틴 단어 수."""
    units = len(str(tgt_text or "").split())
    return max(DISPLAY_MAX_CARRYOVER_WORDS, int(units * DISPLAY_CARRYOVER_MAX_RATIO))


def carryover_words(text):
    """번역문에 남은 **소문자** 라틴 단어들. 고유명사(대문자 시작)는 세지 않는다.

    "Pudding", "HP" 처럼 대문자로 시작하거나 전부 대문자인 토큰은 번역문에 남는 게
    정상이다. 남으면 안 되는 건 'dealing', 'speed', 'nesative' 같은 보통명사/동사다.

    CO-3(2026-09-07): 숫자가 **끝에 붙은** 라틴 조각은 단어가 아니라 식별자다
    (`user1`, `mp3`, `3D`, `PyQt5`). `[A-Za-z]+` 로 자르면 `user1:` 이 `user` 로 잡혀
    미번역 잔류로 오인된다 — 실측에서 채팅 12줄 중 **11줄**이 `carryover(user)` 로 통째로
    숨었다. 사용자가 지목한 주 용도(채팅)에서 게이트가 정상 번역을 92% 가린 것이라
    CO-2와 같은 종류의 사고다.
    반대로 숫자가 **글자 사이에** 있으면(`cre2ted`, `de5cription`, `applicati0n`) 그건
    OCR 오독이 만든 깨진 단어이므로 지금까지처럼 잔류로 센다. 이 구분이 없으면
    둘 중 하나를 반드시 잃는다.
    ponytail: `utf8mb4` 처럼 숫자가 가운데 든 **진짜** 식별자는 잔류로 오인된다.
              그런 화면이 잦으면 예외 목록이 아니라 `DISPLAY_MAX_CARRYOVER_WORDS` 를 올려라.
    """
    # 토큰 단위로 자르면 안 된다 — 번역문에서는 'sive를'처럼 한글 조사가 바로 붙는다.
    # (한글도 isalpha()가 True라 "영문만 남기기"를 문자 검사로 하면 통째로 새어나간다.)
    words = []
    for tok in re.findall(r"[A-Za-z0-9]+", str(text or "")):
        if tok.isalpha():
            words.append(tok)
        elif re.search(r"[A-Za-z][0-9]+[A-Za-z]", tok):   # 글자 사이 숫자 = 깨진 단어
            words += re.findall(r"[A-Za-z]+", tok)
    return [w for w in words
            if len(w) >= DISPLAY_CARRYOVER_MIN_LEN and w.islower()]


# --- RP-2: 반복 붕괴 판정 -----------------------------------------------------
def _char_ngram_repetition(text, n=3):
    """글자만 남긴 뒤 문자 n-gram 중복 비율. 0 = 반복 없음, 1에 가까울수록 같은 말 반복.

    단어 n-gram이 아니라 **문자** n-gram인 이유: 붕괴는 "그 때, 그 때, 한 때의 그 시절,
    그 시절"처럼 조사·어미를 바꿔 가며 반복하는 형태라 단어 단위로는 중복이 안 잡힌다.
    """
    seq = "".join(c for c in str(text or "") if c.isalpha()).lower()
    # 짧은 문자열의 n-gram 통계는 잡음이다. 하한을 5로 잡은 것은 실측이다:
    # 7이면 짧은 라벨을 통째로 면제해 `"Council meeting minutes" → "회의록 회의록"`
    # 같은 중복을 못 본다. 5로 내리면 그게 잡히고(초과 +0.250), 2,129건에서 새 거짓양성은
    # **0건**이다(p99 +0.062 불변). 4 이하로는 더 잡히는 것이 없다.
    if len(seq) < n + 2:
        return 0.0
    grams = [seq[i:i + n] for i in range(len(seq) - n + 1)]
    return 1 - len(set(grams)) / len(grams)


def repetition_excess(src_text, tgt_text):
    """번역문의 반복률에서 **원문 자신의 반복률**을 뺀 값 (RP-2 판정값).

    원문이 이미 반복이면(`--------`, `:py:data:` 나열, "a factor of 1.0 ... a factor of 2.0")
    정상 번역도 반복이라 절대값으로는 못 가른다. 실측 분포(영어 1,940문장 en→ko, greedy):
    정상 p50 -0.027 / p90 0.000 / p99 +0.064 · 붕괴 +0.10 ~ +0.33.
    """
    return _char_ngram_repetition(tgt_text) - _char_ngram_repetition(src_text)


# --- SE-1: 문장 단위 분할 ------------------------------------------------------
# opus-mt / m2m100 은 **문장 쌍**으로 학습된 모델이다. PM-1 문단 병합으로 5줄짜리
# 문단을 통째로 넘겼더니 앞 문장 전체를 빠뜨린 번역이 나왔다(실측 2026-08-05):
#   원문   "When World War II broke out, Saint-Exupéry rejoined the French Air Force.
#            After Nazi troops overtook France in 1940, ... fled to the United States. ..."
#   번역   "1940년에 나치 군대가 프랑스를 점령한 후, 생텍쥐페리는 미국으로 도망쳤습니다..."
#          ← 첫 문장이 통째로 증발
# 길이 잘림이 아니라 모델이 긴 입력을 감당 못 하는 것이라, 출력 토큰을 늘려도 안 낫는다.
# → 문단은 문단대로 합치되(박스 하나), 모델에는 **문장 단위**로 넣고 결과를 다시 잇는다.
_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+")
SENTENCE_SPLIT_MAX_CHARS = 200    # 종결부호가 없는 긴 덩어리는 이 길이에서 강제로 끊는다


def split_sentences(text, max_chars=SENTENCE_SPLIT_MAX_CHARS):
    out = []
    for part in _SENTENCE_END.split(str(text or "").strip()):
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars          # 공백 없는 덩어리(CJK 등)는 그냥 자른다
            out.append(part[:cut].strip())
            part = part[cut:].lstrip()
        if part.strip():
            out.append(part.strip())
    return out or [str(text or "")]


def _cache_key_text(text: str) -> str:
    """RT-1: 캐시 키용 보수적 정규화 — 연속 공백 collapse + 양끝 strip.

    OCR은 같은 정적 화면도 프레임마다 공백/줄바꿈이 미세하게 흔들려 원문 그대로를 키로 쓰면
    캐시가 계속 miss → 1~3초마다 수십 줄 재번역. 표시용 원문/번역문은 원본을 그대로 쓴다.
    (문자 자체를 건드리는 정규화는 다른 언어를 망칠 수 있어 하지 않는다.)
    """
    return " ".join(str(text).split())


# TE-1: 직전 batch_translate 에서 실패한 언어 그룹 메시지. 워커 스레드 한 곳에서만
# 쓰기 때문에 락이 필요 없다(호출부: cocktail_engine 의 OCR 루프/UIA 루프).
LAST_GROUP_ERRORS = []


def _retry_collapsed(pieces, outs, src, tgt):
    """RP-2: 반복 붕괴한 조각만 빔 서치로 다시 뽑아 `outs` 를 제자리 수정한다.

    바꿔치기는 **더 나아졌을 때만** 한다 — 재시도가 더 반복적이면 원래 것을 남긴다.
    실패해도 조용히 원본을 쓴다(번역이 통째로 사라지는 것보다 낫다 — TE-1과 같은 정신).
    """
    bad = [i for i, (p, o) in enumerate(zip(pieces, outs))
           if repetition_excess(p, o) >= RETRY_REPETITION_EXCESS]
    if not bad:
        return
    bad = bad[:RETRY_MAX_PIECES]
    try:
        again = TRANSLATOR.translate_batch([pieces[i] for i in bad], src, tgt,
                                           num_beams=RETRY_NUM_BEAMS)
    except Exception as e:
        print(f"[WARN] RP-2 재시도 실패({len(bad)}조각), greedy 결과 유지: "
              f"{type(e).__name__}: {e}")
        return
    for i, alt in zip(bad, again):
        if repetition_excess(pieces[i], alt) < repetition_excess(pieces[i], outs[i]):
            print(f"[INFO] RP-2 반복 붕괴 재번역: [{log_text(pieces[i])}]")
            outs[i] = alt


def batch_translate(texts, src_lang_hint, tgt_lang):
    """
    여러 라인을 한 번의 generate 호출로 번역.
    src_lang_hint: 'auto', ISO 639-1, 또는 라인별 ISO list.
    tgt_lang: ISO 639-1.
    동일 src끼리 묶어 모델별로 1회 호출. 한 그룹이 실패해도 나머지 그룹은 살아남고,
    실패는 LAST_GROUP_ERRORS 에 남는다 (TE-1).
    """
    LAST_GROUP_ERRORS.clear()
    if not texts:
        return []

    results = [None] * len(texts)
    pending_idx = []
    pending_texts = []
    pending_keys = []   # RT-1: 캐시 저장용 정규화 텍스트 (모델 입력은 pending_texts 원본)
    pending_src = []

    src_list = src_lang_hint if isinstance(src_lang_hint, list) else None

    for i, t in enumerate(texts):
        if src_list is not None:
            src = src_list[i] if i < len(src_list) else "en"
        else:
            src = src_lang_hint if src_lang_hint != "auto" else identify_language(t)
        if src == tgt_lang:
            # SK-1: 번역할 게 없는 줄(원문 언어 == 목표 언어)은 **그리지 않는다**.
            # 예전엔 원문을 그대로 돌려줘서, 목표=한국어인 상태로 한국어가 섞인 화면을 보면
            # 한글 줄마다 "같은 글자가 적힌 검은 박스"가 덮였다. None은 호출부의
            # `if not (tgt_text and tgt_text.strip())` 가드에서 자동으로 걸러진다.
            results[i] = None
            continue
        # RT-1: 캐시 키는 정규화 텍스트. OCR 공백 흔들림이 캐시 miss를 만들지 않게.
        norm = _cache_key_text(t)
        key = (src, tgt_lang, norm)
        with CACHE_LOCK:
            cached = TRANS_CACHE.get_or_none(key)
        if cached is None:
            # Phase 4: 영구 캐시(DPAPI 암호화)에서도 조회. 메모리 캐시로 승격.
            try:
                disk = PERSIST_CACHE.get(src, tgt_lang, norm)
            except Exception:
                disk = None
            if disk is not None:
                cached = disk
                with CACHE_LOCK:
                    TRANS_CACHE.put(key, disk)
        if cached is not None:
            results[i] = cached
            continue
        pending_idx.append(i)
        pending_texts.append(t)
        pending_keys.append(norm)
        pending_src.append(src)

    if not pending_idx:
        return results

    groups = {}
    for j, src in enumerate(pending_src):
        groups.setdefault(src, []).append(j)

    for src, js in groups.items():
        sub_texts = [pending_texts[j] for j in js]
        # SE-1: 모델에는 문장 단위로 넣고, 결과를 원래 줄로 다시 잇는다.
        pieces, owner = [], []
        for k, text in enumerate(sub_texts):
            for sentence in split_sentences(text):
                pieces.append(sentence)
                owner.append(k)
        try:
            translated_pieces = TRANSLATOR.translate_batch(pieces, src, tgt_lang)
        except Exception as e:
            # TE-1: 언어 그룹 격리. 예전엔 예외가 그대로 올라가 **프레임 전체**가 날아갔다
            # (중국어 1줄의 모델 로드 실패로 영어 50줄이 같이 사라짐, 실측 2026-09-05).
            # 이 그룹의 줄만 None으로 두고 나머지 그룹은 계속 번역한다.
            msg = f"{src}→{tgt_lang} {len(js)}줄 실패: {type(e).__name__}: {e}"
            print(f"[WARN] 번역 그룹 {msg}")
            LAST_GROUP_ERRORS.append(msg)   # CO-2: 무음 실패 금지 — 호출부가 상태줄에 띄운다
            continue
        _retry_collapsed(pieces, translated_pieces, src, tgt_lang)
        joined = [[] for _ in sub_texts]
        for piece_out, k in zip(translated_pieces, owner):
            if piece_out and piece_out.strip():
                joined[k].append(piece_out.strip())
        decoded = [" ".join(parts) for parts in joined]
        for local, j in enumerate(js):
            tgt_text = decoded[local]
            results[pending_idx[j]] = tgt_text
            with CACHE_LOCK:
                TRANS_CACHE.put((src, tgt_lang, pending_keys[j]), tgt_text)
            # Phase 4: 영구 캐시도 갱신 (저장은 closeEvent/주기적으로)
            try:
                PERSIST_CACHE.put(src, tgt_lang, pending_keys[j], tgt_text)
            except Exception:
                pass

    return results


# --- PV-1 튜닝 레버: 영구 캐시 = "화면에서 본 모든 문장"의 디스크 기록 --------
# 이 앱의 정체성은 "전부 로컬"이다. 그런데 영구 캐시는 지나간 모든 화면의 번역을
# 사용자 폴더(`~/.cocktail/translation_cache.bin`)에 무기한 쌓았고, 지우는 UI도
# 만료도 끄는 옵션도 없었다(실측 2026-09-07: 개발 PC에 2,574항목 / 265KB).
# DPAPI CurrentUser 는 **같은 윈도우 계정의 아무 프로세스나** 복호화할 수 있으므로
# "암호화돼 있다"가 "안전하다"를 뜻하지 않는다.
#
# 기본값 = **끔**. 근거:
#   ① 같은 세션의 반복은 메모리 LRU(`TRANS_CACHE`, 1024)가 이미 전부 잡는다.
#      영구 캐시가 추가로 버는 것은 "**지난 실행**에서 본 문장을 다시 볼 때"뿐이다.
#   ② 그 이득은 줄당 번역 1회(실측 en→ko 줄당 20~30ms)이고, 기준 4(화면변화→자막)는
#      애초에 영구 캐시를 끈 상태로 잰다 — 즉 채점되는 속도에 영향이 0이다.
#   ③ 은행·로그인 화면이 지나갈 수 있는 앱에서 영구 기록은 opt-in 이어야 한다.
#      민감 창 가드(SB-1)는 창 제목 기반이라 완벽하지 않다.
PERSIST_CACHE_DEFAULT_ON = False

# 만료. 상한(`max_entries`)은 **개수**만 묶고 나이는 안 묶는다 — 조밀한 실화면 한 장이
# 12~22줄(실측, bench 시료)이라 자막을 한 시간 보면 ~600항목이 쌓여 상한이 며칠이면
# 돌지만, 가끔 쓰는 사용자는 하루 수십 항목이라 **가장 오래된 기록이 반 년을 산다**.
# 상한이 지켜 주지 못하는 쪽이 오히려 위험하므로 나이로 한 번 더 자른다.
# 7일 = 영구 캐시가 실제로 버는 재사용 창(어제 보던 문서·게임을 오늘 다시 연다)이고,
# 그 너머는 다시 번역해도 줄당 수십 ms다. 시각은 **마지막으로 쓴 때**를 기준으로 한다.
# ↓면 더 안전하지만 히트율이 떨어지고, ↑면 기록이 오래 남는다.
PERSIST_CACHE_TTL_DAYS = 7


class PersistentCache:
    """DPAPI 암호화된 영구 캐시. 키는 hash, 값은 `[번역문, 마지막 사용 시각]`.

    플랫폼 비-Windows 또는 DPAPI 실패 시 자동으로 무동작 (앱은 정상).
    PV-1: 기본은 **꺼짐**이고, 켜도 TTL 만료 + LRU 축출로 무한히 쌓이지 않는다.
    """

    def __init__(self, path: str, max_entries: int = 4096,
                 ttl_days: float = PERSIST_CACHE_TTL_DAYS):
        self.path = path
        self.max_entries = max_entries
        self.ttl_s = ttl_days * 86400.0
        # OrderedDict: 앞이 "가장 오래 안 쓴 것". 축출은 여기 앞에서만 일어난다(LRU).
        self._mem = OrderedDict()
        self._dirty = False
        self._enabled = sys.platform == "win32" and PERSIST_CACHE_DEFAULT_ON
        self._loaded = False
        self._load_lock = threading.Lock()
        self._mem_lock = threading.RLock()
        # P-9 정신: 디스크 I/O는 import 시점에 X. 첫 get/put 또는 명시적 preload 시.

    def set_enabled(self, on: bool):
        """설정 토글의 단일 진입점.

        **끄면 디스크 기록도 같이 지운다** — 끈 상태로 예전 기록이 남아 있으면
        그건 "끔"이 지켜지지 않은 것이다. 앱 시작 시에도 이 경로를 탄다.
        """
        self._enabled = bool(on) and sys.platform == "win32"
        if not self._enabled:
            self.clear()

    def clear(self) -> int:
        """메모리 + 디스크 기록을 지우고 지운 항목 수를 돌려준다.

        파일은 **삭제**한다(빈 파일로 덮는 게 아니다). 덮어쓰기는 예전 블롭이
        그대로 남을 수 있는 데다, "기록 없음"과 "빈 기록"을 구분할 이유가 없다.
        """
        with self._mem_lock:
            n = len(self._mem)
            self._mem = OrderedDict()
            self._dirty = False
            self._loaded = True   # 지운 뒤 디스크에서 도로 읽어 되살리지 않는다
        for p in (self.path, self.path + ".tmp"):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                print(f"[WARN] PersistentCache 파일 삭제 실패: {e}")
        return n

    def _prune(self, now=None):
        """TTL 만료 + 상한 적용. 호출부가 `_mem_lock` 을 잡고 있어야 한다.

        put 마다 돌리지 않는다 — 만료 검사는 O(n)이고, 만료된 항목은 `get` 이
        어차피 개별로 거른다. 로드 시와 저장 시(30초 주기)에만 쓸어 낸다.
        """
        now = time.time() if now is None else now
        for k in [k for k, v in self._mem.items() if now - v[1] > self.ttl_s]:
            del self._mem[k]
            self._dirty = True
        while len(self._mem) > self.max_entries:
            self._mem.popitem(last=False)
            self._dirty = True

    def preload_async(self):
        """Phase 4: 백그라운드 스레드로 디스크 캐시 로드. UI 시작 안 막음."""
        if self._loaded or not self._enabled:
            return
        threading.Thread(target=self._load, daemon=True).start()

    @staticmethod
    def _key(src: str, tgt: str, text: str) -> str:
        import hashlib
        return hashlib.sha256(f"{src}|{tgt}|{text}".encode("utf-8")).hexdigest()

    def _load(self):
        with self._load_lock:
            if self._loaded:
                return
            self._loaded = True
            if not self._enabled or not os.path.exists(self.path):
                return
            try:
                with open(self.path, "rb") as f:
                    blob = f.read()
                if not blob:
                    return
                decrypted = dpapi_unprotect(blob)
                if decrypted is None:
                    print("[WARN] PersistentCache 복호화 실패 — 새 캐시로 시작")
                    return
                import json
                loaded = json.loads(decrypted.decode("utf-8"))
                # PV-1: 예전 형식({key: 번역문})은 **나이를 모른다** → 되살리지 않는다.
                # 한 번 느려지는 대신 만료를 못 거는 기록을 남기지 않는 쪽을 택했다.
                with self._mem_lock:
                    self._mem = OrderedDict(
                        (k, [v[0], float(v[1])]) for k, v in loaded.items()
                        if isinstance(v, list) and len(v) == 2)
                    kept = len(self._mem)
                    self._prune()
                    if len(self._mem) != kept:
                        self._dirty = True
                    print(f"[INFO] PersistentCache 로드: {len(self._mem)}/{len(loaded)} 항목"
                          f" (만료·구형식 제외)")
            except Exception as e:
                print(f"[WARN] PersistentCache 로드 실패: {e}")
                with self._mem_lock:
                    self._mem = OrderedDict()

    def save(self):
        try:
            import json
            with self._mem_lock:
                if not self._enabled or not self._dirty:
                    return
                self._prune()          # 켜 둔 채 오래 돌아도 나이 든 기록은 안 남는다
                snapshot = dict(self._mem)
            data = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
            blob = dpapi_protect(data)
            if not blob:
                return
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, self.path)
            with self._mem_lock:
                # 저장 중 새 put이 들어왔으면 dirty를 보존한다.
                if snapshot == self._mem:
                    self._dirty = False
        except Exception as e:
            print(f"[WARN] PersistentCache 저장 실패: {e}")

    def get(self, src: str, tgt: str, text: str):
        if not self._enabled:
            return None
        if not self._loaded:
            self._load()  # 동기 로드 (이미 preload_async가 마쳐있을 가능성 큼)
        key = self._key(src, tgt, text)
        now = time.time()
        with self._mem_lock:
            entry = self._mem.get(key)
            if entry is None:
                return None
            if now - entry[1] > self.ttl_s:
                del self._mem[key]      # 만료 — 다음 save 때 디스크에서도 사라진다
                self._dirty = True
                return None
            entry[1] = now              # TTL은 "마지막으로 쓴 뒤"부터 센다
            self._mem.move_to_end(key)  # LRU: 쓴 것은 뒤로
            # 여기서 _dirty 를 세우지 않는다 — 히트마다 265KB를 다시 암호화·기록하게 된다.
            # 시각 갱신은 다음 put 이 만드는 저장에 묻어 간다.
            return entry[0]

    def put(self, src: str, tgt: str, text: str, translation: str):
        if not self._enabled:
            return
        if not self._loaded:
            self._load()
        with self._mem_lock:
            key = self._key(src, tgt, text)
            if key in self._mem:
                self._mem.move_to_end(key)
            self._mem[key] = [translation, time.time()]
            # LRU 축출: 예전엔 dict 앞쪽 절반을 통째로 버려서 **자주 쓰는 항목이
            # 임의로** 날아갔다. 이제 가장 오래 안 쓴 것부터 하나씩 밀어낸다.
            while len(self._mem) > self.max_entries:
                self._mem.popitem(last=False)
            self._dirty = True


# 사용자 폴더에 캐시. 키(원문)는 해시라 복원되지 않지만 **값(번역문)은 평문**이고
# DPAPI CurrentUser 는 같은 계정이면 누구나 푼다 — 그래서 기본이 꺼짐이다(PV-1).
_PERSIST_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cocktail", "translation_cache.bin"
)
PERSIST_CACHE = PersistentCache(_PERSIST_CACHE_PATH)
# UI가 뜬 직후 백그라운드로 로드 (BackgroundController.start에서 호출)
