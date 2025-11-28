# models_utils.py
"""
Utilitaires pour les modèles d'embeddings et LLM.

Supporte deux modes:
- API: Utilise les APIs distantes (Snowflake, DALLEM)
- Local: Utilise les modèles locaux avec CUDA (BGE-M3, Mistral, BGE-Reranker)
"""
import os
import sys
import math
import time
import traceback
from typing import List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
import httpx
import logging
from logging import Logger

from openai import OpenAI
import openai

# Import conditionnel pour les modèles locaux
try:
    from local_models import (
        LocalEmbeddings,
        LocalLLM,
        LocalReranker,
        local_models_manager,
        cuda_manager,
        get_cuda_status,
    )
    LOCAL_MODELS_AVAILABLE = True
except ImportError:
    LOCAL_MODELS_AVAILABLE = False

# Import de la configuration
try:
    from config_manager import load_config, is_local_mode
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    def load_config():
        return None
    def is_local_mode():
        return False


# ---------------------------------------------------------------------
#  CONFIG réseau / modèles
# ---------------------------------------------------------------------

LLM_MODEL = "dallem-val"
EMBED_MODEL = "snowflake-arctic-embed-l-v2.0"
BATCH_SIZE = 32  # taille batch embeddings (équilibre performance/sécurité)
MAX_CHARS_PER_TEXT = 28000  # ~7000 tokens max par texte (limite Snowflake: 8192 tokens)
PARALLEL_EMBEDDING_WORKERS = 8  # Workers parallèles pour appels API (I/O bound, pas CPU)

HARDCODE = {
    "DALLEM_API_BASE": "https://api.dev.dassault-aviation.pro/dallem-pilote/v1",
    "SNOWFLAKE_API_BASE": "https://api.dev.dassault-aviation.pro/snowflake-arctic-embed-l-v2.0/v1",
    "DALLEM_API_KEY": "EMPTY",     # à surcharger par l'env
    "SNOWFLAKE_API_KEY": "token",  # à surcharger par l'env
    "DISABLE_SSL_VERIFY": "true",
}

DALLEM_API_BASE = os.getenv("DALLEM_API_BASE", HARDCODE["DALLEM_API_BASE"]).rstrip("/")
SNOWFLAKE_API_BASE = os.getenv("SNOWFLAKE_API_BASE", HARDCODE["SNOWFLAKE_API_BASE"]).rstrip("/")
DALLEM_API_KEY = os.getenv("DALLEM_API_KEY", HARDCODE["DALLEM_API_KEY"])
SNOWFLAKE_API_KEY = os.getenv("SNOWFLAKE_API_KEY", HARDCODE["SNOWFLAKE_API_KEY"])

VERIFY_SSL = not (
    os.getenv("DISABLE_SSL_VERIFY", HARDCODE["DISABLE_SSL_VERIFY"])
    .lower()
    in ("1", "true", "yes", "on")
)


def _mask(s: Optional[str]) -> str:
    if not s:
        return "<vide>"
    if len(s) <= 6:
        return "***"
    return s[:3] + "…" + s[-3:]


def make_logger(debug: bool) -> Logger:
    log = logging.getLogger("rag_da")

    # Choix des niveaux : console silencieuse en mode non-debug
    if debug:
        level_console = logging.DEBUG
        level_logger = logging.DEBUG
    else:
        level_console = logging.WARNING
        level_logger = logging.WARNING

    log.setLevel(level_logger)

    # Si le logger a déjà des handlers, on met juste à jour leurs niveaux
    if log.handlers:
        for h in log.handlers:
            if isinstance(h, logging.StreamHandler):
                h.setLevel(level_console)
        return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt)

    # Fichier : on garde tout en DEBUG pour analyse détaillée
    fh = logging.FileHandler("rag_da_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)

    # Ces logs seront visibles au moins dans le fichier
    log.info("=== Configuration RAG_DA ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
        )
    )
    return log

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if debug else logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler("rag_da_debug.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)

    log.info("=== Configuration RAG_DA ===")
    log.info(f"SNOWFLAKE_API_BASE = {SNOWFLAKE_API_BASE}")
    log.info(f"DALLEM_API_BASE    = {DALLEM_API_BASE}")
    log.info(f"VERIFY_SSL         = {VERIFY_SSL}")
    log.info(f"EMBED_MODEL        = {EMBED_MODEL}")
    log.info(f"BATCH_SIZE         = {BATCH_SIZE}")
    log.info(
        "API_KEYS           = snowflake={} | dallem={}".format(
            _mask(SNOWFLAKE_API_KEY),
            _mask(DALLEM_API_KEY),
        )
    )
    return log


def create_http_client() -> httpx.Client:
    """
    Client HTTP configuré (timeout, SSL) pour Snowflake + DALLEM.
    """
    return httpx.Client(
        verify=VERIFY_SSL,
        timeout=httpx.Timeout(300.0),
    )


# ---------------------------------------------------------------------
#  Client embeddings Snowflake (OpenAI v1-compatible)
# ---------------------------------------------------------------------


class DirectOpenAIEmbeddings:
    """
    Client embeddings minimal (OpenAI v1-compatible).
    role_prefix=True -> "passage:" / "query:".
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        http_client: Optional[httpx.Client] = None,
        role_prefix: bool = True,
        logger: Optional[Logger] = None,
    ):
        self.model = model
        self.role_prefix = role_prefix
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
        )
        self.log = logger or logging.getLogger("rag_da")

    def _apply_prefix(self, items: List[str], role: str) -> List[str]:
        if not self.role_prefix:
            return items
        pref = "query: " if role == "query" else "passage: "
        return [pref + (x or "") for x in items]

    def _retry_request(self, func, max_retries: int = 5, base_delay: float = 1.0):
        """
        Exécute func() avec retry exponentiel sur les erreurs transitoires.
        """
        for attempt in range(max_retries):
            try:
                return func()
            except (openai.APIConnectionError, openai.RateLimitError, openai.APIError) as e:
                if attempt == max_retries - 1:
                    self.log.error(
                        f"[embeddings] Échec après {max_retries} tentatives — {type(e).__name__}: {e}"
                    )
                    raise
                wait_time = base_delay * (2 ** attempt)
                self.log.warning(
                    f"[embeddings] Tentative {attempt + 1}/{max_retries} échouée "
                    f"({type(e).__name__}: {e}) — retry dans {wait_time:.1f}s"
                )
                time.sleep(wait_time)

    def _create_embeddings(self, inputs: List[str]) -> List[List[float]]:
        t0 = time.time()
        self.log.debug(
            f"[embeddings] POST {self.client.base_url} | model={self.model} "
            f"| n_inputs={len(inputs)} | len0={len(inputs[0]) if inputs else 0}"
        )

        def _do_request():
            return self.client.embeddings.create(model=self.model, input=inputs)

        try:
            resp = self._retry_request(_do_request)
            dur = (time.time() - t0) * 1000
            self.log.debug(
                f"[embeddings] OK in {dur:.1f} ms | items={len(resp.data)} "
                f"| dim≈{len(resp.data[0].embedding) if resp.data else 'n/a'}"
            )
            return [d.embedding for d in resp.data]

        except openai.NotFoundError as e:
            self.log.error(f"[embeddings] NotFoundError (modèle='{self.model}' ?) : {e}")
            self.log.debug(traceback.format_exc())
            raise
        except openai.AuthenticationError as e:
            self.log.error("[embeddings] AuthenticationError — clé invalide ?")
            self.log.debug(traceback.format_exc())
            raise
        except Exception as e:
            self.log.error(f"[embeddings] Exception — {e}")
            self.log.debug(traceback.format_exc())
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="passage")
        return self._create_embeddings(inputs)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        inputs = self._apply_prefix(list(texts or []), role="query")
        return self._create_embeddings(inputs)


def _embed_single_batch(
    batch_info: Tuple[int, List[str]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    dry_run: bool,
) -> Tuple[int, List[List[float]]]:
    """
    Embed un seul batch de textes (fonction worker pour le parallélisme).
    Retourne (batch_index, embeddings).
    """
    batch_idx, chunk = batch_info
    if dry_run:
        dim = 1024
        fake = np.random.rand(len(chunk), dim).astype(np.float32) - 0.5
        return (batch_idx, fake.tolist())
    else:
        if role == "query":
            return (batch_idx, emb_client.embed_queries(chunk))
        else:
            return (batch_idx, emb_client.embed_documents(chunk))


def _embed_sequential(
    batches: List[Tuple[int, List[str]]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool,
) -> List[List[float]]:
    """
    Méthode séquentielle d'embedding (fallback).
    """
    out: List[List[float]] = []
    total_batches = len(batches)

    for batch_idx, chunk in batches:
        log.debug(
            f"[emb-seq] batch {batch_idx + 1}/{total_batches} "
            f"| size={len(chunk)}"
        )
        try:
            _, embeddings = _embed_single_batch((batch_idx, chunk), role, emb_client, dry_run)
            out.extend(embeddings)
        except Exception as e:
            log.error(f"[emb-seq] échec batch {batch_idx} — {e}")
            raise

    return out


def _embed_parallel(
    batches: List[Tuple[int, List[str]]],
    role: str,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool,
    max_workers: int,
) -> List[List[float]]:
    """
    Méthode parallèle d'embedding avec ThreadPoolExecutor.
    """
    total_batches = len(batches)
    results: dict = {}  # batch_idx -> embeddings

    log.info(f"[emb-parallel] Démarrage avec {max_workers} workers pour {total_batches} batches")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les batches
        future_to_batch = {
            executor.submit(_embed_single_batch, batch, role, emb_client, dry_run): batch[0]
            for batch in batches
        }

        completed = 0
        for future in as_completed(future_to_batch):
            batch_idx = future_to_batch[future]
            try:
                idx, embeddings = future.result()
                results[idx] = embeddings
                completed += 1
                log.debug(f"[emb-parallel] batch {idx + 1}/{total_batches} terminé ({completed}/{total_batches})")
            except Exception as e:
                log.error(f"[emb-parallel] échec batch {batch_idx} — {e}")
                raise

    # Reconstruire la liste ordonnée
    out: List[List[float]] = []
    for i in range(total_batches):
        out.extend(results[i])

    return out


def embed_in_batches(
    texts: List[str],
    role: str,
    batch_size: int,
    emb_client: DirectOpenAIEmbeddings,
    log: Logger,
    dry_run: bool = False,
    use_parallel: bool = True,
) -> np.ndarray:
    """
    Découpe en batches, appelle le client embeddings, normalise les vecteurs (L2).
    Tronque automatiquement les textes trop longs pour éviter les erreurs de tokens.

    Args:
        use_parallel: Si True, utilise le traitement parallèle (multicoeur).
                     Si erreur, fallback automatique sur séquentiel.
    """
    # Tronquer les textes trop longs (limite Snowflake: 8192 tokens ≈ 28000 chars)
    truncated_count = 0
    safe_texts = []
    for t in texts:
        if len(t) > MAX_CHARS_PER_TEXT:
            safe_texts.append(t[:MAX_CHARS_PER_TEXT])
            truncated_count += 1
        else:
            safe_texts.append(t)

    if truncated_count > 0:
        log.warning(f"[emb] {truncated_count} texte(s) tronqué(s) à {MAX_CHARS_PER_TEXT} caractères")

    n = len(safe_texts)

    # Préparer les batches
    batches: List[Tuple[int, List[str]]] = []
    batch_idx = 0
    for i in range(0, n, batch_size):
        chunk = safe_texts[i: i + batch_size]
        batches.append((batch_idx, chunk))
        batch_idx += 1

    total_batches = len(batches)
    mode = "parallel" if use_parallel and total_batches > 1 else "sequential"
    log.info(
        f"[emb] start role={role} | n={n} | batch_size={batch_size} | "
        f"batches={total_batches} | mode={mode} | workers={PARALLEL_EMBEDDING_WORKERS} | dry_run={dry_run}"
    )

    out: List[List[float]] = []

    # Essayer le mode parallèle si activé et plusieurs batches
    if use_parallel and total_batches > 1:
        try:
            out = _embed_parallel(batches, role, emb_client, log, dry_run, PARALLEL_EMBEDDING_WORKERS)
            log.info(f"[emb] Mode parallèle OK ({total_batches} batches, {PARALLEL_EMBEDDING_WORKERS} workers)")
        except Exception as e:
            log.warning(f"[emb] Mode parallèle échoué, fallback séquentiel: {e}")
            out = _embed_sequential(batches, role, emb_client, log, dry_run)
    else:
        # Mode séquentiel direct
        out = _embed_sequential(batches, role, emb_client, log, dry_run)

    M = np.asarray(out, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] != n:
        log.error(f"[emb] shape inattendue: {M.shape} (attendu ({n}, d))")

    denom = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    if np.any(np.isnan(denom)):
        log.warning("[emb] NaN détecté dans la norme, correction appliquée.")
        denom = np.nan_to_num(denom, nan=1.0)
    M = M / denom
    log.info(
        f"[emb] terminé | shape={M.shape} | d={M.shape[1] if M.ndim == 2 else 'n/a'}"
    )
    return M


# ---------------------------------------------------------------------
#  Appel LLM DALLEM
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
#  Client LLM Local (wrapper)
# ---------------------------------------------------------------------

class LocalLLMWrapper:
    """
    Wrapper pour le LLM local qui expose une interface similaire à l'API.
    """

    def __init__(self, logger: Optional[Logger] = None):
        self.log = logger or logging.getLogger("rag_da")
        self._llm = None

    def _get_llm(self) -> Optional["LocalLLM"]:
        """Récupère ou charge le LLM local."""
        if self._llm is None and LOCAL_MODELS_AVAILABLE:
            self._llm = local_models_manager.get_llm()
        return self._llm

    def chat_completion(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Génère une réponse via le LLM local.

        Args:
            question: Question de l'utilisateur
            context: Contexte documentaire

        Returns:
            Réponse générée
        """
        llm = self._get_llm()
        if llm is None:
            raise RuntimeError("LLM local non disponible ou non configuré")

        self.log.info("[LLM-LOCAL] Génération de réponse...")

        try:
            response = llm.chat_completion(question, context)
            self.log.info(f"[LLM-LOCAL] Réponse générée ({len(response)} chars)")
            return response

        except Exception as e:
            self.log.error(f"[LLM-LOCAL] Erreur: {e}")
            raise


def call_dallem_chat(
    http_client: httpx.Client,
    question: str,
    context: str,
    log: Logger,
) -> str:
    """
    Appel simple au LLM DALLEM via /chat/completions.
    """
    if not DALLEM_API_KEY or DALLEM_API_KEY == "toto":
        raise RuntimeError("DALLEM_API_KEY manquant ou de test. Impossible d'utiliser le LLM.")

    system_msg = (
        "Tu es un assistant expert en réglementation aéronautique. "
        "Tu dois répondre en te basant sur le CONTEXTE fourni ci-dessous. "
        "Le contexte contient des extraits de documents normatifs (CS, AMC, GM). "
        "Cite toujours les références (CS xx.xxx, AMC, etc.) présentes dans le contexte."
    )

    import textwrap
    user_msg = textwrap.dedent(f"""
    === CONTEXTE DOCUMENTAIRE ===
    {context}
    === FIN DU CONTEXTE ===

    QUESTION : {question}

    INSTRUCTIONS :
    - Réponds en anglais en te basant sur les informations du contexte ci-dessus
    - Cite les références normatives (CS, AMC, GM) mentionnées dans le contexte
    - Si le contexte contient des informations pertinentes, utilise-les pour répondre
    - Seulement si le contexte ne contient AUCUNE information pertinente, réponds : "I do not have the information to answer your question."
    """)

    url = DALLEM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {DALLEM_API_KEY}",
        "Content-Type": "application/json",
    }

    # Log du contexte pour diagnostic
    log.info(f"[RAG] Contexte: {len(context)} chars, {context.count('[source=')} sources")
    if not context.strip():
        log.warning("[RAG] ⚠️ CONTEXTE VIDE - pas de chunks trouvés!")

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 2000,
        "temperature": 0.3,
    }

    log.info("[RAG] Appel DALLEM /chat/completions pour réponse RAG")

    # Retry logic avec backoff exponentiel
    max_retries = 4
    base_delay = 2  # secondes

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = http_client.post(url, headers=headers, json=payload, timeout=180.0)
            resp.raise_for_status()
            data = resp.json()

            # Vérifier que la réponse contient bien du contenu
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if not content:
                raise ValueError("Réponse LLM vide")

            log.info(f"[RAG] ✅ Réponse DALLEM reçue (attempt {attempt + 1}/{max_retries})")
            return content

        except Exception as e:
            last_error = e
            delay = base_delay * (2 ** attempt)  # 2, 4, 8, 16 secondes

            if attempt < max_retries - 1:
                log.warning(f"[RAG] ⚠️ Erreur DALLEM (attempt {attempt + 1}/{max_retries}): {e}")
                log.info(f"[RAG] Retry dans {delay}s...")
                time.sleep(delay)
            else:
                log.error(f"[RAG] ❌ Échec DALLEM après {max_retries} tentatives: {e}")

    # Toutes les tentatives ont échoué - retourner un message d'erreur spécial
    error_msg = (
        "⚠️ **ERREUR DE COMMUNICATION AVEC LE LLM**\n\n"
        f"Le serveur n'a pas pu répondre après {max_retries} tentatives.\n\n"
        f"**Erreur technique:** {str(last_error)[:200]}\n\n"
        "👉 **Veuillez reposer votre question** ou réessayer dans quelques instants."
    )
    return error_msg


# ---------------------------------------------------------------------
#  FONCTIONS UNIFIÉES (LOCAL OU API)
# ---------------------------------------------------------------------

def call_llm_chat(
    http_client: Optional[httpx.Client],
    question: str,
    context: str,
    log: Logger,
    use_local: Optional[bool] = None,
) -> str:
    """
    Appel unifié au LLM (local ou API selon la configuration).

    Args:
        http_client: Client HTTP pour l'API (peut être None si local)
        question: Question de l'utilisateur
        context: Contexte documentaire
        log: Logger
        use_local: Force le mode local (True) ou API (False). Si None, utilise la config.

    Returns:
        Réponse générée
    """
    # Déterminer le mode
    if use_local is None:
        use_local = is_local_mode() if CONFIG_AVAILABLE else False

    if use_local and LOCAL_MODELS_AVAILABLE:
        log.info("[LLM] Mode local activé")
        try:
            llm_wrapper = LocalLLMWrapper(logger=log)
            return llm_wrapper.chat_completion(question, context)
        except Exception as e:
            log.error(f"[LLM] Erreur mode local: {e}")
            # Fallback vers API si possible
            if http_client is not None:
                log.warning("[LLM] Fallback vers API...")
                return call_dallem_chat(http_client, question, context, log)
            raise

    # Mode API
    if http_client is None:
        http_client = create_http_client()

    return call_dallem_chat(http_client, question, context, log)


class UnifiedEmbeddings:
    """
    Client d'embeddings unifié qui utilise soit l'API soit les modèles locaux.
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        use_local: Optional[bool] = None,
    ):
        """
        Args:
            logger: Logger optionnel
            use_local: Force le mode local (True) ou API (False). Si None, utilise la config.
        """
        self.log = logger or logging.getLogger("rag_da")

        # Déterminer le mode
        if use_local is None:
            self.use_local = is_local_mode() if CONFIG_AVAILABLE else False
        else:
            self.use_local = use_local

        self._local_embeddings = None
        self._api_embeddings = None

        if self.use_local and LOCAL_MODELS_AVAILABLE:
            self.log.info("[EMB] Mode local activé")
            self._init_local()
        else:
            self.log.info("[EMB] Mode API activé")
            self._init_api()

    def _init_local(self):
        """Initialise le client d'embeddings local."""
        try:
            # Diagnostic: vérifier la configuration du manager
            self.log.info(f"[EMB] Tentative d'initialisation locale...")
            self.log.info(f"[EMB] local_models_manager.embedding_path = {local_models_manager.embedding_path}")
            self.log.info(f"[EMB] local_models_manager.has_embedding_model() = {local_models_manager.has_embedding_model()}")

            self._local_embeddings = local_models_manager.get_embeddings()
            if self._local_embeddings is None:
                self.log.warning("[EMB] Modèle local non disponible (get_embeddings() a retourné None), fallback API")
                self.log.warning(f"[EMB] Vérifiez que le chemin existe: {local_models_manager.embedding_path}")
                self.use_local = False
                self._init_api()
            else:
                self.log.info(f"[EMB] ✅ Modèle local chargé avec succès, dimension={self._local_embeddings.dimension}")
        except Exception as e:
            self.log.error(f"[EMB] Erreur init local: {e}")
            import traceback
            self.log.error(traceback.format_exc())
            self.use_local = False
            self._init_api()

    def _init_api(self):
        """Initialise le client d'embeddings API."""
        http_client = create_http_client()
        self._api_embeddings = DirectOpenAIEmbeddings(
            model=EMBED_MODEL,
            api_key=SNOWFLAKE_API_KEY,
            base_url=SNOWFLAKE_API_BASE,
            http_client=http_client,
            role_prefix=True,
            logger=self.log,
        )

    @property
    def dimension(self) -> int:
        """Retourne la dimension des embeddings."""
        if self.use_local and self._local_embeddings:
            return self._local_embeddings.dimension
        return 1024  # Dimension par défaut pour Snowflake

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Génère les embeddings pour des documents.

        Args:
            texts: Liste des textes

        Returns:
            Array numpy (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        if self.use_local and self._local_embeddings:
            return self._local_embeddings.embed_documents(texts)

        # Mode API
        return np.array(self._api_embeddings.embed_documents(texts))

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        """
        Génère les embeddings pour des requêtes.

        Args:
            texts: Liste des requêtes

        Returns:
            Array numpy (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        if self.use_local and self._local_embeddings:
            return self._local_embeddings.embed_queries(texts)

        # Mode API
        return np.array(self._api_embeddings.embed_queries(texts))


def create_unified_embeddings(
    logger: Optional[Logger] = None,
    use_local: Optional[bool] = None,
) -> UnifiedEmbeddings:
    """
    Crée un client d'embeddings unifié.

    Args:
        logger: Logger optionnel
        use_local: Force le mode local (True) ou API (False). Si None, utilise la config.

    Returns:
        UnifiedEmbeddings
    """
    return UnifiedEmbeddings(logger=logger, use_local=use_local)


def embed_texts_unified(
    texts: List[str],
    role: str,
    batch_size: int,
    log: Logger,
    use_local: Optional[bool] = None,
    dry_run: bool = False,
) -> np.ndarray:
    """
    Génère les embeddings de manière unifiée (local ou API).

    Args:
        texts: Textes à encoder
        role: "query" ou "passage"
        batch_size: Taille des batches (utilisé uniquement en mode API)
        log: Logger
        use_local: Force le mode local (True) ou API (False). Si None, utilise la config.
        dry_run: Si True, génère des embeddings aléatoires

    Returns:
        Array numpy normalisé (n_texts, dimension)
    """
    if dry_run:
        dim = 1024
        fake = np.random.rand(len(texts), dim).astype(np.float32) - 0.5
        # Normalisation L2
        denom = np.linalg.norm(fake, axis=1, keepdims=True) + 1e-12
        return fake / denom

    # Déterminer le mode
    if use_local is None:
        use_local = is_local_mode() if CONFIG_AVAILABLE else False

    log.info(f"[EMB] use_local={use_local}, LOCAL_MODELS_AVAILABLE={LOCAL_MODELS_AVAILABLE}")

    if use_local and LOCAL_MODELS_AVAILABLE:
        log.info(f"[EMB] Mode local, role={role}, n={len(texts)}")
        log.info(f"[EMB] local_models_manager.embedding_path = {local_models_manager.embedding_path}")

        embeddings_client = local_models_manager.get_embeddings()

        if embeddings_client is None:
            log.warning(f"[EMB] Modèle local non disponible (path={local_models_manager.embedding_path}), fallback API")
            use_local = False
        else:
            log.info(f"[EMB] ✅ Utilisation du modèle local d'embeddings")
            if role == "query":
                result = embeddings_client.embed_queries(texts)
            else:
                result = embeddings_client.embed_documents(texts)

            return result

    # Mode API - utiliser la fonction existante
    http_client = create_http_client()
    emb_client = DirectOpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SNOWFLAKE_API_KEY,
        base_url=SNOWFLAKE_API_BASE,
        http_client=http_client,
        role_prefix=True,
        logger=log,
    )

    return embed_in_batches(
        texts=texts,
        role=role,
        batch_size=batch_size,
        emb_client=emb_client,
        log=log,
        dry_run=False,
        use_parallel=True,
    )


# ---------------------------------------------------------------------
#  INFORMATIONS SUR LE MODE ACTUEL
# ---------------------------------------------------------------------

def get_models_mode_info() -> dict:
    """
    Retourne des informations sur le mode actuel (local ou API).

    Returns:
        Dict avec les informations sur le mode et les modèles
    """
    info = {
        "mode": "local" if (CONFIG_AVAILABLE and is_local_mode()) else "api",
        "local_available": LOCAL_MODELS_AVAILABLE,
        "config_available": CONFIG_AVAILABLE,
    }

    if LOCAL_MODELS_AVAILABLE:
        try:
            cuda_info = get_cuda_status()
            info["cuda"] = cuda_info
        except Exception:
            info["cuda"] = {"available": False}

        if CONFIG_AVAILABLE:
            config = load_config()
            if config:
                info["local_paths"] = {
                    "embedding": config.local_embedding_path,
                    "llm": config.local_llm_path,
                    "reranker": config.local_reranker_path,
                }

    return info
