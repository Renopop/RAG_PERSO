# local_models.py
"""
Gestionnaire de modeles locaux avec support CUDA intelligent.

Ce module fournit:
- Detection automatique de GPU/CUDA
- Gestion dynamique de la VRAM et du batch size
- Gestion des erreurs OOM avec fallback automatique
- Classes pour: Embeddings (BGE-M3), LLM (Mistral), Reranker (BGE-Reranker)
"""

import os
import gc
import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# =====================================================================
#  CONFIGURATION CUDA / GPU
# =====================================================================

class DeviceType(Enum):
    """Types de devices disponibles."""
    CUDA = "cuda"
    CPU = "cpu"


@dataclass
class GPUInfo:
    """Informations sur le GPU."""
    name: str = ""
    total_memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    used_memory_gb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)
    is_available: bool = False


@dataclass
class BatchConfig:
    """Configuration dynamique du batch size."""
    initial_batch_size: int = 32
    min_batch_size: int = 1
    max_batch_size: int = 256
    current_batch_size: int = 32
    vram_threshold_gb: float = 1.0  # VRAM libre minimum avant reduction
    reduction_factor: float = 0.5  # Facteur de reduction sur OOM
    increase_factor: float = 1.2  # Facteur d'augmentation si OK


class CUDAManager:
    """Gestionnaire centralisé pour CUDA et la VRAM."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern pour le gestionnaire CUDA."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.torch = None
        self.cuda_available = False
        self.gpu_info = GPUInfo()
        self.batch_configs: Dict[str, BatchConfig] = {}
        self._initialize_cuda()
        self._initialized = True

    def _initialize_cuda(self):
        """Initialise CUDA et detecte le GPU."""
        try:
            import torch
            self.torch = torch

            if torch.cuda.is_available():
                self.cuda_available = True
                device_idx = 0

                props = torch.cuda.get_device_properties(device_idx)
                total_mem = props.total_memory / (1024**3)
                free_mem = (props.total_memory - torch.cuda.memory_allocated(device_idx)) / (1024**3)
                used_mem = torch.cuda.memory_allocated(device_idx) / (1024**3)

                self.gpu_info = GPUInfo(
                    name=props.name,
                    total_memory_gb=total_mem,
                    free_memory_gb=free_mem,
                    used_memory_gb=used_mem,
                    compute_capability=(props.major, props.minor),
                    is_available=True
                )

                logger.info(f"[CUDA] GPU detecte: {props.name}")
                logger.info(f"[CUDA] VRAM totale: {total_mem:.2f} GB")
                logger.info(f"[CUDA] VRAM libre: {free_mem:.2f} GB")
                logger.info(f"[CUDA] Compute capability: {props.major}.{props.minor}")
            else:
                logger.warning("[CUDA] CUDA non disponible, utilisation du CPU")

        except ImportError:
            logger.warning("[CUDA] PyTorch non installe, utilisation du CPU uniquement")
        except Exception as e:
            logger.error(f"[CUDA] Erreur lors de l'initialisation CUDA: {e}")

    def get_device(self) -> str:
        """Retourne le device optimal ('cuda' ou 'cpu')."""
        if self.cuda_available and self.torch is not None:
            return "cuda"
        return "cpu"

    def get_free_vram_gb(self) -> float:
        """Retourne la VRAM libre en GB."""
        if not self.cuda_available or self.torch is None:
            return 0.0

        try:
            torch = self.torch
            free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            return free / (1024**3)
        except Exception:
            return 0.0

    def get_batch_config(self, model_name: str) -> BatchConfig:
        """Recupere ou cree une config de batch pour un modele."""
        if model_name not in self.batch_configs:
            # Configuration initiale basee sur la VRAM disponible
            free_vram = self.get_free_vram_gb()

            if free_vram >= 16:
                initial_batch = 64
            elif free_vram >= 8:
                initial_batch = 32
            elif free_vram >= 4:
                initial_batch = 16
            elif free_vram >= 2:
                initial_batch = 8
            else:
                initial_batch = 4

            self.batch_configs[model_name] = BatchConfig(
                initial_batch_size=initial_batch,
                current_batch_size=initial_batch
            )

            logger.info(f"[BATCH] Config pour {model_name}: batch_size={initial_batch} (VRAM libre={free_vram:.2f}GB)")

        return self.batch_configs[model_name]

    def adjust_batch_on_oom(self, model_name: str) -> int:
        """Reduit le batch size apres une erreur OOM."""
        config = self.get_batch_config(model_name)

        new_size = max(
            config.min_batch_size,
            int(config.current_batch_size * config.reduction_factor)
        )

        if new_size == config.current_batch_size:
            new_size = max(config.min_batch_size, new_size - 1)

        config.current_batch_size = new_size
        logger.warning(f"[BATCH] OOM detecte pour {model_name}, reduction batch_size: {new_size}")

        # Nettoyer la memoire
        self.clear_cuda_cache()

        return new_size

    def adjust_batch_on_success(self, model_name: str) -> int:
        """Augmente le batch size apres un succes."""
        config = self.get_batch_config(model_name)

        # Verifier qu'il y a assez de VRAM libre
        free_vram = self.get_free_vram_gb()
        if free_vram > config.vram_threshold_gb * 2:
            new_size = min(
                config.max_batch_size,
                int(config.current_batch_size * config.increase_factor)
            )
            if new_size > config.current_batch_size:
                config.current_batch_size = new_size
                logger.debug(f"[BATCH] Augmentation batch_size pour {model_name}: {new_size}")

        return config.current_batch_size

    def clear_cuda_cache(self):
        """Libere le cache CUDA."""
        if self.cuda_available and self.torch is not None:
            try:
                self.torch.cuda.empty_cache()
                gc.collect()
                logger.debug("[CUDA] Cache CUDA libere")
            except Exception as e:
                logger.warning(f"[CUDA] Erreur lors du nettoyage du cache: {e}")

    def estimate_optimal_batch_size(
        self,
        model_name: str,
        sample_size_bytes: int,
        model_overhead_gb: float = 2.0
    ) -> int:
        """
        Estime le batch size optimal en fonction de la VRAM.

        Args:
            model_name: Nom du modele
            sample_size_bytes: Taille estimee d'un echantillon en bytes
            model_overhead_gb: VRAM utilisee par le modele lui-meme
        """
        free_vram = self.get_free_vram_gb()
        usable_vram = max(0, free_vram - model_overhead_gb - 0.5)  # 0.5 GB de marge

        if usable_vram <= 0:
            return 1

        # Estimer le nombre d'echantillons qui tiennent en memoire
        usable_bytes = usable_vram * (1024**3)
        estimated_batch = int(usable_bytes / sample_size_bytes)

        config = self.get_batch_config(model_name)
        optimal = min(max(1, estimated_batch), config.max_batch_size)

        logger.debug(f"[BATCH] Estimation pour {model_name}: {optimal} (VRAM usable={usable_vram:.2f}GB)")
        return optimal


# Instance globale du gestionnaire CUDA
cuda_manager = CUDAManager()


# =====================================================================
#  EMBEDDINGS LOCAUX (BGE-M3)
# =====================================================================

class LocalEmbeddings:
    """
    Client d'embeddings local utilisant BGE-M3.

    Gere automatiquement:
    - Le batch size adaptatif
    - Les erreurs OOM avec fallback
    - Le passage CPU si necessaire
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        normalize: bool = True,
        max_length: int = 8192,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model_path: Chemin vers le modele BGE-M3
            device: 'cuda', 'cpu' ou None (auto)
            normalize: Normaliser les embeddings (L2)
            max_length: Longueur max des tokens
            logger: Logger optionnel
        """
        self.model_path = model_path
        self.normalize = normalize
        self.max_length = max_length
        self.log = logger or logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self.device = device or cuda_manager.get_device()
        self.dimension = 1024  # BGE-M3 dimension

        self._load_model()

    def _load_model(self):
        """Charge le modele d'embeddings."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch

            self.log.info(f"[EMB-LOCAL] Chargement du modele depuis: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            # Deplacer vers le device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.log.info("[EMB-LOCAL] Modele charge sur GPU")
            else:
                self.device = "cpu"
                self.log.info("[EMB-LOCAL] Modele charge sur CPU")

            self.model.eval()

            # Determiner la dimension des embeddings
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors="pt", truncation=True)
                if self.device == "cuda":
                    test_input = {k: v.cuda() for k, v in test_input.items()}
                test_output = self.model(**test_input)
                self.dimension = test_output.last_hidden_state.shape[-1]

            self.log.info(f"[EMB-LOCAL] Dimension des embeddings: {self.dimension}")

        except Exception as e:
            self.log.error(f"[EMB-LOCAL] Erreur de chargement du modele: {e}")
            raise

    def _embed_batch(
        self,
        texts: List[str],
        batch_size: int
    ) -> np.ndarray:
        """Genere les embeddings pour un batch de textes."""
        import torch

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # Tokenization
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)

                    # Mean pooling sur les tokens (excluant le padding)
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state

                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                        input_mask_expanded.sum(1), min=1e-9
                    )

                    # Normalisation L2
                    if self.normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    all_embeddings.append(embeddings.cpu().numpy())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise MemoryError(f"CUDA OOM during embedding batch {i//batch_size}")
                raise

        return np.vstack(all_embeddings) if all_embeddings else np.array([])

    def embed_documents(
        self,
        texts: List[str],
        prefix: str = "passage: "
    ) -> np.ndarray:
        """
        Genere les embeddings pour des documents.

        Args:
            texts: Liste des textes
            prefix: Prefixe pour les documents (passage: pour BGE)

        Returns:
            Array numpy (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        # Appliquer le prefixe
        prefixed_texts = [prefix + t for t in texts]

        return self._embed_with_fallback(prefixed_texts, "embedding_docs")

    def embed_queries(
        self,
        texts: List[str],
        prefix: str = "query: "
    ) -> np.ndarray:
        """
        Genere les embeddings pour des requetes.

        Args:
            texts: Liste des requetes
            prefix: Prefixe pour les requetes (query: pour BGE)

        Returns:
            Array numpy (n_texts, dimension)
        """
        if not texts:
            return np.array([])

        # Appliquer le prefixe
        prefixed_texts = [prefix + t for t in texts]

        return self._embed_with_fallback(prefixed_texts, "embedding_queries")

    def _embed_with_fallback(
        self,
        texts: List[str],
        model_name: str
    ) -> np.ndarray:
        """
        Genere les embeddings avec gestion OOM et fallback.

        En cas d'erreur OOM:
        1. Reduit le batch size
        2. Retente
        3. Si echec, passe en CPU
        """
        batch_config = cuda_manager.get_batch_config(model_name)
        batch_size = batch_config.current_batch_size

        max_retries = 5
        cpu_fallback_done = False

        for attempt in range(max_retries):
            try:
                self.log.debug(f"[EMB-LOCAL] Tentative {attempt+1}/{max_retries}, batch_size={batch_size}")

                result = self._embed_batch(texts, batch_size)

                # Succes - ajuster le batch size a la hausse si possible
                if attempt == 0:
                    cuda_manager.adjust_batch_on_success(model_name)

                return result

            except (MemoryError, RuntimeError) as e:
                if "out of memory" in str(e).lower() or isinstance(e, MemoryError):
                    self.log.warning(f"[EMB-LOCAL] OOM (tentative {attempt+1}): {e}")

                    # Reduire le batch size
                    batch_size = cuda_manager.adjust_batch_on_oom(model_name)

                    # Si batch_size == 1 et toujours OOM, passer en CPU
                    if batch_size <= 1 and not cpu_fallback_done:
                        self.log.warning("[EMB-LOCAL] Fallback vers CPU")
                        self._switch_to_cpu()
                        cpu_fallback_done = True
                        batch_size = 4  # Reset batch size pour CPU
                        batch_config.current_batch_size = batch_size
                        continue

                    if batch_size < 1:
                        raise RuntimeError("Impossible de generer les embeddings meme avec batch_size=1 sur CPU")
                else:
                    raise

        raise RuntimeError(f"Echec apres {max_retries} tentatives")

    def _switch_to_cpu(self):
        """Deplace le modele vers le CPU."""
        try:
            self.model = self.model.cpu()
            self.device = "cpu"
            cuda_manager.clear_cuda_cache()
            self.log.info("[EMB-LOCAL] Modele deplace vers CPU")
        except Exception as e:
            self.log.error(f"[EMB-LOCAL] Erreur lors du passage CPU: {e}")


# =====================================================================
#  LLM LOCAL (MISTRAL)
# =====================================================================

class LocalLLM:
    """
    Client LLM local utilisant Mistral-7B-Instruct.

    Gere automatiquement:
    - Le chargement avec quantization si necessaire
    - Les erreurs OOM avec fallback
    - La generation de texte avec parametres ajustables
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,  # 4-bit par defaut pour economiser la VRAM
        max_new_tokens: int = 2048,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model_path: Chemin vers le modele Mistral
            device: 'cuda', 'cpu' ou None (auto)
            load_in_8bit: Charger en 8-bit quantization
            load_in_4bit: Charger en 4-bit quantization (prioritaire)
            max_new_tokens: Nombre max de tokens a generer
            logger: Logger optionnel
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.log = logger or logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self.device = device or cuda_manager.get_device()
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self._load_model()

    def _load_model(self):
        """Charge le modele LLM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch

            self.log.info(f"[LLM-LOCAL] Chargement du modele depuis: {self.model_path}")

            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Configuration de quantization
            quantization_config = None
            if self.device == "cuda" and torch.cuda.is_available():
                free_vram = cuda_manager.get_free_vram_gb()

                if self.load_in_4bit or free_vram < 16:
                    self.log.info("[LLM-LOCAL] Chargement en 4-bit (economie VRAM)")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif self.load_in_8bit or free_vram < 24:
                    self.log.info("[LLM-LOCAL] Chargement en 8-bit")
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
                else:
                    self.log.info("[LLM-LOCAL] Chargement en fp16")

            # Charger le modele
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
            }

            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            elif self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )

            if self.device == "cpu":
                self.model = self.model.cpu()

            self.model.eval()
            self.log.info(f"[LLM-LOCAL] Modele charge sur {self.device}")

        except Exception as e:
            self.log.error(f"[LLM-LOCAL] Erreur de chargement du modele: {e}")
            raise

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Genere une reponse a partir d'un prompt.

        Args:
            prompt: Le prompt utilisateur
            system_prompt: Prompt systeme optionnel
            max_new_tokens: Nombre max de tokens (defaut: self.max_new_tokens)
            temperature: Temperature de sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Activer le sampling stochastique

        Returns:
            Texte genere
        """
        import torch

        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Construire le message au format Mistral Instruct
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Format pour Mistral Instruct
        formatted_prompt = self._format_messages(messages)

        try:
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )

            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decoder uniquement les nouveaux tokens
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.log.error("[LLM-LOCAL] OOM lors de la generation")
                cuda_manager.clear_cuda_cache()
                # Retenter avec moins de tokens
                if max_new_tokens > 256:
                    self.log.warning("[LLM-LOCAL] Retry avec moins de tokens")
                    return self.generate(
                        prompt, system_prompt,
                        max_new_tokens=max_new_tokens // 2,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample
                    )
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Formate les messages au format Mistral Instruct.
        """
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                formatted += f"<s>[INST] {content}\n"
            elif role == "user":
                if formatted:
                    formatted += f"{content} [/INST]"
                else:
                    formatted += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>"

        return formatted

    def chat_completion(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Interface compatible avec l'API pour les appels RAG.

        Args:
            question: Question de l'utilisateur
            context: Contexte documentaire
            system_prompt: Prompt systeme optionnel

        Returns:
            Reponse generee
        """
        if system_prompt is None:
            system_prompt = (
                "Tu es un assistant expert en reglementation aeronautique. "
                "Tu dois repondre en te basant sur le CONTEXTE fourni ci-dessous. "
                "Le contexte contient des extraits de documents normatifs (CS, AMC, GM). "
                "Cite toujours les references (CS xx.xxx, AMC, etc.) presentes dans le contexte."
            )

        user_prompt = f"""=== CONTEXTE DOCUMENTAIRE ===
{context}
=== FIN DU CONTEXTE ===

QUESTION : {question}

INSTRUCTIONS :
- Reponds en anglais en te basant sur les informations du contexte ci-dessus
- Cite les references normatives (CS, AMC, GM) mentionnees dans le contexte
- Si le contexte contient des informations pertinentes, utilise-les pour repondre
- Seulement si le contexte ne contient AUCUNE information pertinente, reponds : "I do not have the information to answer your question."
"""

        return self.generate(user_prompt, system_prompt)


# =====================================================================
#  RERANKER LOCAL (BGE-RERANKER-V2-M3)
# =====================================================================

class LocalReranker:
    """
    Reranker local utilisant BGE-Reranker-v2-M3.

    Gere automatiquement:
    - Le batch size adaptatif
    - Les erreurs OOM avec fallback
    """

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            model_path: Chemin vers le modele BGE-Reranker
            device: 'cuda', 'cpu' ou None (auto)
            max_length: Longueur max des tokens
            logger: Logger optionnel
        """
        self.model_path = model_path
        self.max_length = max_length
        self.log = logger or logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self.device = device or cuda_manager.get_device()

        self._load_model()

    def _load_model(self):
        """Charge le modele reranker."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            self.log.info(f"[RERANK-LOCAL] Chargement du modele depuis: {self.model_path}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.model = self.model.half()  # FP16 pour economiser la VRAM
                self.log.info("[RERANK-LOCAL] Modele charge sur GPU (FP16)")
            else:
                self.device = "cpu"
                self.log.info("[RERANK-LOCAL] Modele charge sur CPU")

            self.model.eval()

        except Exception as e:
            self.log.error(f"[RERANK-LOCAL] Erreur de chargement du modele: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Reordonne les documents par pertinence.

        Args:
            query: La requete
            documents: Liste des documents a reordonner
            top_k: Nombre de resultats a retourner (None = tous)

        Returns:
            Liste de dicts avec index, score et document
        """
        if not documents:
            return []

        import torch

        batch_config = cuda_manager.get_batch_config("reranker")
        batch_size = batch_config.current_batch_size

        all_scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]

            try:
                # Creer les paires query-document
                pairs = [[query, doc] for doc in batch_docs]

                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )

                if self.device == "cuda":
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    scores = outputs.logits.squeeze(-1)

                    # Appliquer sigmoid pour avoir des scores entre 0 et 1
                    scores = torch.sigmoid(scores)
                    all_scores.extend(scores.cpu().numpy().tolist())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.log.warning(f"[RERANK-LOCAL] OOM, reduction batch_size")
                    batch_size = cuda_manager.adjust_batch_on_oom("reranker")

                    # Retry ce batch avec un batch_size plus petit
                    if batch_size < 1:
                        # Fallback CPU
                        self._switch_to_cpu()
                        batch_size = 4

                    # Recommencer depuis le debut du batch actuel
                    all_scores = all_scores[:i]  # Garder les scores deja calcules
                    continue
                raise

        # Creer les resultats tries
        results = []
        for idx, (score, doc) in enumerate(zip(all_scores, documents)):
            results.append({
                "index": idx,
                "score": float(score),
                "relevance_score": float(score),
                "document": doc
            })

        # Trier par score decroissant
        results.sort(key=lambda x: x["score"], reverse=True)

        if top_k:
            results = results[:top_k]

        self.log.info(f"[RERANK-LOCAL] {len(results)} documents reordonnes")
        return results

    def _switch_to_cpu(self):
        """Deplace le modele vers le CPU."""
        try:
            self.model = self.model.float().cpu()
            self.device = "cpu"
            cuda_manager.clear_cuda_cache()
            self.log.info("[RERANK-LOCAL] Modele deplace vers CPU")
        except Exception as e:
            self.log.error(f"[RERANK-LOCAL] Erreur lors du passage CPU: {e}")


# =====================================================================
#  FACTORY / GESTIONNAIRE DE MODELES LOCAUX
# =====================================================================

class LocalModelsManager:
    """
    Gestionnaire centralise des modeles locaux.

    Singleton qui gere:
    - Le chargement paresseux des modeles
    - La configuration globale
    - Le nettoyage des ressources
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.embedding_model: Optional[LocalEmbeddings] = None
        self.llm_model: Optional[LocalLLM] = None
        self.reranker_model: Optional[LocalReranker] = None

        self.embedding_path: Optional[str] = None
        self.llm_path: Optional[str] = None
        self.reranker_path: Optional[str] = None

        self._initialized = True

    def configure(
        self,
        embedding_path: Optional[str] = None,
        llm_path: Optional[str] = None,
        reranker_path: Optional[str] = None
    ):
        """Configure les chemins des modeles."""
        if embedding_path:
            self.embedding_path = embedding_path
        if llm_path:
            self.llm_path = llm_path
        if reranker_path:
            self.reranker_path = reranker_path

    def get_embeddings(self) -> Optional[LocalEmbeddings]:
        """Retourne le modele d'embeddings (charge paresseusement)."""
        if self.embedding_model is None and self.embedding_path:
            if os.path.exists(self.embedding_path):
                self.embedding_model = LocalEmbeddings(self.embedding_path)
            else:
                logger.warning(f"[LOCAL] Chemin embeddings non trouve: {self.embedding_path}")
        return self.embedding_model

    def get_llm(self) -> Optional[LocalLLM]:
        """Retourne le LLM (charge paresseusement)."""
        if self.llm_model is None and self.llm_path:
            if os.path.exists(self.llm_path):
                self.llm_model = LocalLLM(self.llm_path)
            else:
                logger.warning(f"[LOCAL] Chemin LLM non trouve: {self.llm_path}")
        return self.llm_model

    def get_reranker(self) -> Optional[LocalReranker]:
        """Retourne le reranker (charge paresseusement)."""
        if self.reranker_model is None and self.reranker_path:
            if os.path.exists(self.reranker_path):
                self.reranker_model = LocalReranker(self.reranker_path)
            else:
                logger.warning(f"[LOCAL] Chemin reranker non trouve: {self.reranker_path}")
        return self.reranker_model

    def is_configured(self) -> bool:
        """Verifie si au moins un modele est configure."""
        return any([self.embedding_path, self.llm_path, self.reranker_path])

    def has_embedding_model(self) -> bool:
        """Verifie si un modele d'embeddings est disponible."""
        return self.embedding_path is not None and os.path.exists(self.embedding_path)

    def has_llm_model(self) -> bool:
        """Verifie si un LLM est disponible."""
        return self.llm_path is not None and os.path.exists(self.llm_path)

    def has_reranker_model(self) -> bool:
        """Verifie si un reranker est disponible."""
        return self.reranker_path is not None and os.path.exists(self.reranker_path)

    def unload_all(self):
        """Decharge tous les modeles et libere la memoire."""
        self.embedding_model = None
        self.llm_model = None
        self.reranker_model = None
        cuda_manager.clear_cuda_cache()
        gc.collect()
        logger.info("[LOCAL] Tous les modeles decharges")

    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut des modeles."""
        return {
            "cuda_available": cuda_manager.cuda_available,
            "gpu_info": {
                "name": cuda_manager.gpu_info.name,
                "total_memory_gb": cuda_manager.gpu_info.total_memory_gb,
                "free_memory_gb": cuda_manager.get_free_vram_gb(),
            } if cuda_manager.cuda_available else None,
            "embedding": {
                "configured": self.embedding_path is not None,
                "path": self.embedding_path,
                "loaded": self.embedding_model is not None,
                "exists": self.has_embedding_model(),
            },
            "llm": {
                "configured": self.llm_path is not None,
                "path": self.llm_path,
                "loaded": self.llm_model is not None,
                "exists": self.has_llm_model(),
            },
            "reranker": {
                "configured": self.reranker_path is not None,
                "path": self.reranker_path,
                "loaded": self.reranker_model is not None,
                "exists": self.has_reranker_model(),
            },
        }


# Instance globale du gestionnaire de modeles
local_models_manager = LocalModelsManager()


# =====================================================================
#  FONCTIONS UTILITAIRES
# =====================================================================

def configure_local_models(
    embedding_path: Optional[str] = None,
    llm_path: Optional[str] = None,
    reranker_path: Optional[str] = None
):
    """Configure les modeles locaux (fonction utilitaire)."""
    local_models_manager.configure(embedding_path, llm_path, reranker_path)


def get_cuda_status() -> Dict[str, Any]:
    """Retourne le statut CUDA."""
    return {
        "available": cuda_manager.cuda_available,
        "device": cuda_manager.get_device(),
        "gpu_name": cuda_manager.gpu_info.name if cuda_manager.cuda_available else None,
        "total_vram_gb": cuda_manager.gpu_info.total_memory_gb if cuda_manager.cuda_available else 0,
        "free_vram_gb": cuda_manager.get_free_vram_gb(),
    }


def get_models_status() -> Dict[str, Any]:
    """Retourne le statut complet des modeles."""
    return local_models_manager.get_status()
