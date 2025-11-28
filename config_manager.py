# config_manager.py
"""
Gestionnaire de configuration pour les répertoires de stockage et les modèles.

Ce module permet de:
- Charger/sauvegarder la configuration des chemins
- Valider l'existence des répertoires
- Proposer la création des répertoires manquants
- Afficher une interface de configuration si nécessaire
- Gérer la configuration des modèles locaux (embeddings, LLM, reranker)
- Supporter le mode local vs API
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


# Fichier de configuration local
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")


class ModelMode(Enum):
    """Mode d'utilisation des modèles."""
    API = "api"      # Utilise les APIs distantes (Snowflake, DALLEM)
    LOCAL = "local"  # Utilise les modèles locaux


# Valeurs par défaut (chemins réseau PROP)
DEFAULT_CONFIG = {
    "base_root_dir": r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\BaseDB",
    "csv_import_dir": r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\CSV_Ingestion",
    "csv_export_dir": r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\Fichiers_Tracking_CSV",
    "feedback_dir": r"N:\DA\SOC\RDA\ORG\DGT\POLE-SYSTEME\ENERGIE\RESERVE\PROP\Knowledge\IA_PROP\FAISS_DATABASE\Feedbacks",
}

# Configuration par défaut pour le mode local
DEFAULT_LOCAL_CONFIG = {
    "base_root_dir": r"D:\FAISS_DATABASE\BaseDB",
    "csv_import_dir": r"D:\FAISS_DATABASE\CSV_Ingestion",
    "csv_export_dir": r"D:\FAISS_DATABASE\Fichiers_Tracking_CSV",
    "feedback_dir": r"D:\FAISS_DATABASE\Feedbacks",
    "local_embedding_path": r"D:\IA_Test\models\BAAI\bge-m3",
    "local_llm_path": r"D:\IA_Test\models\mistralai\Mistral-7B-Instruct-v0.3",
    "local_reranker_path": r"D:\IA_Test\models\BAAI\bge-reranker-v2-m3",
    "selected_llm": "mistral-7b",  # LLM sélectionné par défaut
}

# Catalogue des LLMs locaux disponibles
AVAILABLE_LOCAL_LLMS = {
    "mistral-7b": {
        "name": "Mistral 7B Instruct v0.3",
        "path": r"D:\IA_Test\models\mistralai\Mistral-7B-Instruct-v0.3",
        "description": "Modèle 7B paramètres, bon équilibre performance/ressources",
        "vram_required_gb": 6.0,  # En 4-bit quantization
        "model_type": "mistral",
    },
    "qwen-3b": {
        "name": "Qwen 2.5 3B Instruct",
        "path": r"D:\IA_Test\models\Qwen\Qwen2.5-3B-Instruct",
        "description": "Modèle 3B paramètres, léger et rapide",
        "vram_required_gb": 3.0,  # En 4-bit quantization
        "model_type": "qwen",
    },
}


@dataclass
class LocalModelsConfig:
    """Configuration des modèles locaux."""
    embedding_path: str = ""
    llm_path: str = ""
    reranker_path: str = ""
    selected_llm: str = "mistral-7b"  # ID du LLM sélectionné

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "LocalModelsConfig":
        return cls(
            embedding_path=data.get("local_embedding_path", ""),
            llm_path=data.get("local_llm_path", ""),
            reranker_path=data.get("local_reranker_path", ""),
            selected_llm=data.get("selected_llm", "mistral-7b"),
        )

    def has_any_model(self) -> bool:
        """Vérifie si au moins un modèle local est configuré."""
        return bool(self.embedding_path or self.llm_path or self.reranker_path)

    def get_selected_llm_info(self) -> Optional[Dict]:
        """Retourne les infos du LLM sélectionné."""
        return AVAILABLE_LOCAL_LLMS.get(self.selected_llm)

    def get_selected_llm_path(self) -> str:
        """Retourne le chemin du LLM sélectionné."""
        llm_info = self.get_selected_llm_info()
        if llm_info:
            return llm_info["path"]
        return self.llm_path  # Fallback sur le chemin configuré

    def get_selected_llm_type(self) -> str:
        """Retourne le type du LLM sélectionné (mistral, qwen, etc.)."""
        llm_info = self.get_selected_llm_info()
        if llm_info:
            return llm_info.get("model_type", "mistral")
        return "mistral"  # Default

    def validate_paths(self) -> Dict[str, Tuple[bool, str]]:
        """Valide que les chemins des modèles existent."""
        results = {}
        if self.embedding_path:
            exists = os.path.exists(self.embedding_path)
            results["embedding"] = (exists, self.embedding_path if exists else f"Non trouvé: {self.embedding_path}")

        # Valider le LLM sélectionné
        llm_path = self.get_selected_llm_path()
        if llm_path:
            exists = os.path.exists(llm_path)
            llm_info = self.get_selected_llm_info()
            llm_name = llm_info["name"] if llm_info else "LLM"
            results["llm"] = (exists, f"{llm_name}: {llm_path}" if exists else f"Non trouvé: {llm_path}")

        if self.reranker_path:
            exists = os.path.exists(self.reranker_path)
            results["reranker"] = (exists, self.reranker_path if exists else f"Non trouvé: {self.reranker_path}")

        return results


@dataclass
class StorageConfig:
    """Configuration des répertoires de stockage et des modèles."""
    base_root_dir: str
    csv_import_dir: str
    csv_export_dir: str
    feedback_dir: str
    # Mode de fonctionnement (api ou local)
    model_mode: str = "api"
    # Configuration des modèles locaux
    local_embedding_path: str = ""
    local_llm_path: str = ""
    local_reranker_path: str = ""
    # LLM sélectionné (ID dans AVAILABLE_LOCAL_LLMS)
    selected_llm: str = "mistral-7b"

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "StorageConfig":
        return cls(
            base_root_dir=data.get("base_root_dir", DEFAULT_CONFIG["base_root_dir"]),
            csv_import_dir=data.get("csv_import_dir", DEFAULT_CONFIG["csv_import_dir"]),
            csv_export_dir=data.get("csv_export_dir", DEFAULT_CONFIG["csv_export_dir"]),
            feedback_dir=data.get("feedback_dir", DEFAULT_CONFIG["feedback_dir"]),
            model_mode=data.get("model_mode", "api"),
            local_embedding_path=data.get("local_embedding_path", ""),
            local_llm_path=data.get("local_llm_path", ""),
            local_reranker_path=data.get("local_reranker_path", ""),
            selected_llm=data.get("selected_llm", "mistral-7b"),
        )

    def is_local_mode(self) -> bool:
        """Retourne True si le mode local est activé."""
        return self.model_mode == "local"

    def get_local_models_config(self) -> LocalModelsConfig:
        """Retourne la configuration des modèles locaux."""
        return LocalModelsConfig(
            embedding_path=self.local_embedding_path,
            llm_path=self.local_llm_path,
            reranker_path=self.local_reranker_path,
            selected_llm=self.selected_llm,
        )

    def get_selected_llm_path(self) -> str:
        """Retourne le chemin du LLM sélectionné."""
        llm_info = AVAILABLE_LOCAL_LLMS.get(self.selected_llm)
        if llm_info:
            return llm_info["path"]
        return self.local_llm_path  # Fallback

    def get_selected_llm_type(self) -> str:
        """Retourne le type du LLM sélectionné."""
        llm_info = AVAILABLE_LOCAL_LLMS.get(self.selected_llm)
        if llm_info:
            return llm_info.get("model_type", "mistral")
        return "mistral"

    @classmethod
    def create_local_config(cls) -> "StorageConfig":
        """Crée une configuration pour le mode local avec les valeurs par défaut."""
        return cls(
            base_root_dir=DEFAULT_LOCAL_CONFIG["base_root_dir"],
            csv_import_dir=DEFAULT_LOCAL_CONFIG["csv_import_dir"],
            csv_export_dir=DEFAULT_LOCAL_CONFIG["csv_export_dir"],
            feedback_dir=DEFAULT_LOCAL_CONFIG["feedback_dir"],
            model_mode="local",
            local_embedding_path=DEFAULT_LOCAL_CONFIG["local_embedding_path"],
            local_llm_path=DEFAULT_LOCAL_CONFIG["local_llm_path"],
            local_reranker_path=DEFAULT_LOCAL_CONFIG["local_reranker_path"],
            selected_llm=DEFAULT_LOCAL_CONFIG.get("selected_llm", "mistral-7b"),
        )


def load_config() -> StorageConfig:
    """
    Charge la configuration depuis le fichier config.json.
    Si le fichier n'existe pas, utilise les valeurs par défaut.
    """
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return StorageConfig.from_dict(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[CONFIG] Erreur de lecture du fichier config: {e}")
            # Fallback aux valeurs par défaut
            pass

    return StorageConfig.from_dict(DEFAULT_CONFIG)


def save_config(config: StorageConfig) -> bool:
    """
    Sauvegarde la configuration dans le fichier config.json.

    Returns:
        True si la sauvegarde a réussi, False sinon.
    """
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"[CONFIG] Erreur de sauvegarde du fichier config: {e}")
        return False


def validate_directory(path: str) -> Tuple[bool, str]:
    """
    Valide qu'un répertoire existe et est accessible.

    Args:
        path: Chemin du répertoire à valider

    Returns:
        Tuple (valide, message)
    """
    if not path or path.strip() == "":
        return False, "Chemin vide"

    path = path.strip()

    # Vérifier si le chemin existe
    if not os.path.exists(path):
        return False, f"Le répertoire n'existe pas: {path}"

    # Vérifier si c'est un répertoire
    if not os.path.isdir(path):
        return False, f"Le chemin n'est pas un répertoire: {path}"

    # Vérifier les permissions d'écriture
    try:
        test_file = os.path.join(path, ".write_test_temp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except (IOError, PermissionError) as e:
        return False, f"Pas de permission d'écriture: {path}"

    return True, "OK"


def validate_all_directories(config: StorageConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Valide tous les répertoires de la configuration.

    Returns:
        Dict avec le nom du répertoire et le tuple (valide, message)
    """
    results = {}

    directories = {
        "base_root_dir": ("Bases FAISS", config.base_root_dir),
        "csv_import_dir": ("CSV d'ingestion", config.csv_import_dir),
        "csv_export_dir": ("CSV de tracking", config.csv_export_dir),
        "feedback_dir": ("Feedbacks", config.feedback_dir),
    }

    for key, (label, path) in directories.items():
        valid, message = validate_directory(path)
        results[key] = (valid, message, label, path)

    return results


def create_directory(path: str) -> Tuple[bool, str]:
    """
    Crée un répertoire (avec les parents si nécessaire).

    Returns:
        Tuple (succès, message)
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True, f"Répertoire créé: {path}"
    except PermissionError:
        return False, f"Permission refusée pour créer: {path}"
    except OSError as e:
        return False, f"Erreur de création: {e}"


def get_missing_directories(config: StorageConfig) -> List[Tuple[str, str, str]]:
    """
    Retourne la liste des répertoires manquants.

    Returns:
        Liste de tuples (key, label, path) pour les répertoires manquants
    """
    results = validate_all_directories(config)
    missing = []

    for key, (valid, message, label, path) in results.items():
        if not valid:
            missing.append((key, label, path, message))

    return missing


def is_config_valid(config: StorageConfig) -> bool:
    """
    Vérifie si la configuration est valide (tous les répertoires existent).
    """
    results = validate_all_directories(config)
    return all(valid for valid, _, _, _ in results.values())


def ensure_directories_exist(config: StorageConfig, create_if_missing: bool = False) -> Tuple[bool, List[str]]:
    """
    S'assure que tous les répertoires existent.

    Args:
        config: Configuration à valider
        create_if_missing: Si True, tente de créer les répertoires manquants

    Returns:
        Tuple (tous_valides, liste_des_erreurs)
    """
    errors = []
    results = validate_all_directories(config)

    for key, (valid, message, label, path) in results.items():
        if not valid:
            if create_if_missing:
                success, create_msg = create_directory(path)
                if not success:
                    errors.append(f"{label}: {create_msg}")
            else:
                errors.append(f"{label}: {message}")

    return len(errors) == 0, errors


# =====================================================================
#  FONCTIONS POUR STREAMLIT
# =====================================================================

def render_config_page_streamlit():
    """
    Affiche la page de configuration dans Streamlit.
    À utiliser quand les répertoires ne sont pas valides.
    """
    import streamlit as st

    st.title("Configuration des répertoires de stockage")
    st.warning("Les répertoires de stockage ne sont pas configurés ou inaccessibles.")
    st.info("Veuillez configurer les chemins ci-dessous pour continuer.")

    # Charger la configuration actuelle
    config = load_config()

    st.markdown("---")

    # Formulaire de configuration
    st.subheader("Répertoires de stockage")

    new_base_root = st.text_input(
        "Répertoire des bases FAISS",
        value=config.base_root_dir,
        help="Chemin absolu vers le dossier contenant les bases FAISS"
    )

    new_csv_import = st.text_input(
        "Répertoire des CSV d'ingestion",
        value=config.csv_import_dir,
        help="Chemin absolu vers le dossier contenant les CSV pour l'ingestion"
    )

    new_csv_export = st.text_input(
        "Répertoire des CSV de tracking",
        value=config.csv_export_dir,
        help="Chemin absolu vers le dossier pour exporter les CSV de suivi"
    )

    new_feedback = st.text_input(
        "Répertoire des feedbacks",
        value=config.feedback_dir,
        help="Chemin absolu vers le dossier pour stocker les feedbacks utilisateurs"
    )

    # Créer une nouvelle configuration
    new_config = StorageConfig(
        base_root_dir=new_base_root,
        csv_import_dir=new_csv_import,
        csv_export_dir=new_csv_export,
        feedback_dir=new_feedback,
    )

    # Afficher le statut de chaque répertoire
    st.markdown("---")
    st.subheader("Statut des répertoires")

    results = validate_all_directories(new_config)
    all_valid = True

    for key, (valid, message, label, path) in results.items():
        if valid:
            st.success(f"✅ {label}: {path}")
        else:
            st.error(f"❌ {label}: {message}")
            all_valid = False

    st.markdown("---")

    # Boutons d'action
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Créer les répertoires manquants", type="secondary"):
            created = []
            failed = []

            for key, (valid, message, label, path) in results.items():
                if not valid:
                    success, msg = create_directory(path)
                    if success:
                        created.append(label)
                    else:
                        failed.append(f"{label}: {msg}")

            if created:
                st.success(f"Créés: {', '.join(created)}")
            if failed:
                st.error(f"Échecs: {', '.join(failed)}")

            st.rerun()

    with col2:
        if st.button("Sauvegarder la configuration", type="primary", disabled=not all_valid):
            if save_config(new_config):
                st.success("Configuration sauvegardée!")
                st.rerun()
            else:
                st.error("Erreur lors de la sauvegarde")

    with col3:
        if st.button("Utiliser les valeurs par défaut"):
            default_config = StorageConfig.from_dict(DEFAULT_CONFIG)
            save_config(default_config)
            st.rerun()

    # Message d'aide
    st.markdown("---")
    st.info("""
    **Aide:**
    - Les chemins doivent être des chemins absolus (ex: `C:\\Data\\FAISS` ou `N:\\Partage\\Data`)
    - Les répertoires doivent être accessibles en lecture et écriture
    - Si les répertoires n'existent pas, cliquez sur "Créer les répertoires manquants"
    - Une fois tous les répertoires valides (✅), cliquez sur "Sauvegarder la configuration"
    """)

    return all_valid


def check_and_show_config_if_needed() -> Optional[StorageConfig]:
    """
    Vérifie la configuration et affiche la page de configuration si nécessaire.

    Returns:
        StorageConfig si valide, None si la page de configuration est affichée
    """
    import streamlit as st

    config = load_config()

    if is_config_valid(config):
        return config

    # Configuration invalide - afficher la page de configuration
    render_config_page_streamlit()
    st.stop()  # Arrêter l'exécution du reste de l'app
    return None


# =====================================================================
#  COMPATIBILITÉ AVEC L'ANCIENNE INTERFACE
# =====================================================================

def get_base_root_dir() -> str:
    """Retourne le répertoire des bases FAISS."""
    return load_config().base_root_dir


def get_csv_import_dir() -> str:
    """Retourne le répertoire des CSV d'ingestion."""
    return load_config().csv_import_dir


def get_csv_export_dir() -> str:
    """Retourne le répertoire des CSV de tracking."""
    return load_config().csv_export_dir


def get_feedback_dir() -> str:
    """Retourne le répertoire des feedbacks."""
    return load_config().feedback_dir


# =====================================================================
#  FONCTIONS POUR LES MODÈLES LOCAUX
# =====================================================================

def is_local_mode() -> bool:
    """Vérifie si le mode local est activé."""
    return load_config().is_local_mode()


def get_local_models_config() -> LocalModelsConfig:
    """Retourne la configuration des modèles locaux."""
    return load_config().get_local_models_config()


def get_local_embedding_path() -> str:
    """Retourne le chemin du modèle d'embeddings local."""
    return load_config().local_embedding_path


def get_local_llm_path() -> str:
    """Retourne le chemin du LLM local sélectionné."""
    return load_config().get_selected_llm_path()


def get_selected_llm_id() -> str:
    """Retourne l'ID du LLM sélectionné."""
    return load_config().selected_llm


def get_selected_llm_type() -> str:
    """Retourne le type du LLM sélectionné (mistral, qwen, etc.)."""
    return load_config().get_selected_llm_type()


def get_selected_llm_info() -> Optional[Dict]:
    """Retourne les informations du LLM sélectionné."""
    config = load_config()
    return AVAILABLE_LOCAL_LLMS.get(config.selected_llm)


def get_local_reranker_path() -> str:
    """Retourne le chemin du reranker local."""
    return load_config().local_reranker_path


def switch_to_local_mode() -> StorageConfig:
    """
    Bascule vers le mode local avec les valeurs par défaut.
    Sauvegarde la configuration et retourne la nouvelle config.
    """
    config = StorageConfig.create_local_config()
    save_config(config)
    return config


def switch_to_api_mode() -> StorageConfig:
    """
    Bascule vers le mode API avec les valeurs par défaut.
    Sauvegarde la configuration et retourne la nouvelle config.
    """
    config = StorageConfig.from_dict(DEFAULT_CONFIG)
    config.model_mode = "api"
    save_config(config)
    return config


def render_local_models_config_streamlit():
    """
    Affiche la section de configuration des modèles locaux dans Streamlit.
    """
    import streamlit as st

    config = load_config()

    st.markdown("---")
    st.subheader("Configuration des modèles locaux")

    # Mode de fonctionnement
    mode_options = ["API (distant)", "Local (GPU)"]
    current_mode_idx = 1 if config.is_local_mode() else 0

    new_mode = st.radio(
        "Mode de fonctionnement",
        options=mode_options,
        index=current_mode_idx,
        horizontal=True,
        help="API: utilise les serveurs distants. Local: utilise les modèles sur votre machine avec GPU/CUDA."
    )

    is_local = (new_mode == "Local (GPU)")

    if is_local:
        st.info("Mode local activé - Les modèles seront chargés sur votre GPU/CPU")

        # Afficher le statut CUDA si disponible
        try:
            from local_models import get_cuda_status
            cuda_status = get_cuda_status()

            if cuda_status["available"]:
                st.success(f"GPU détecté: {cuda_status['gpu_name']}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("VRAM totale", f"{cuda_status['total_vram_gb']:.1f} GB")
                with col2:
                    st.metric("VRAM libre", f"{cuda_status['free_vram_gb']:.1f} GB")
            else:
                st.warning("Aucun GPU CUDA détecté - Les modèles utiliseront le CPU (plus lent)")
        except ImportError:
            st.warning("Module local_models non disponible")

        st.markdown("#### Modèle d'embeddings")

        new_embedding_path = st.text_input(
            "Chemin du modèle BGE-M3",
            value=config.local_embedding_path or DEFAULT_LOCAL_CONFIG["local_embedding_path"],
            help="Chemin vers le modèle BGE-M3 pour les embeddings"
        )

        st.markdown("#### Modèle LLM (génération de texte)")

        # Menu déroulant pour sélectionner le LLM
        llm_options = list(AVAILABLE_LOCAL_LLMS.keys())
        llm_display_names = [
            f"{AVAILABLE_LOCAL_LLMS[k]['name']} ({AVAILABLE_LOCAL_LLMS[k]['vram_required_gb']:.1f} GB VRAM)"
            for k in llm_options
        ]

        # Trouver l'index du LLM actuellement sélectionné
        current_llm_idx = 0
        if config.selected_llm in llm_options:
            current_llm_idx = llm_options.index(config.selected_llm)

        selected_llm_display = st.selectbox(
            "Sélectionner le LLM",
            options=llm_display_names,
            index=current_llm_idx,
            help="Choisissez le modèle LLM à utiliser pour la génération de réponses"
        )

        # Récupérer l'ID du LLM sélectionné
        selected_llm_idx = llm_display_names.index(selected_llm_display)
        selected_llm_id = llm_options[selected_llm_idx]
        selected_llm_info = AVAILABLE_LOCAL_LLMS[selected_llm_id]

        # Afficher les infos du LLM sélectionné
        st.caption(f"**Description:** {selected_llm_info['description']}")
        st.caption(f"**Chemin:** `{selected_llm_info['path']}`")

        # Stocker le chemin du LLM sélectionné
        new_llm_path = selected_llm_info['path']

        st.markdown("#### Modèle Reranker")

        new_reranker_path = st.text_input(
            "Chemin du modèle BGE-Reranker-v2-M3",
            value=config.local_reranker_path or DEFAULT_LOCAL_CONFIG["local_reranker_path"],
            help="Chemin vers le modèle BGE-Reranker pour le re-ranking"
        )

        # Valider les chemins
        st.markdown("#### Statut des modèles")
        models_valid = True

        # Vérifier l'embedding
        if new_embedding_path:
            if os.path.exists(new_embedding_path):
                st.success(f"✅ Embeddings (BGE-M3): `{new_embedding_path}`")
            else:
                st.error(f"❌ Embeddings: Chemin non trouvé - `{new_embedding_path}`")
                models_valid = False
        else:
            st.warning("⚠️ Embeddings: Non configuré")

        # Vérifier le LLM
        if new_llm_path:
            if os.path.exists(new_llm_path):
                st.success(f"✅ LLM ({selected_llm_info['name']}): `{new_llm_path}`")
            else:
                st.error(f"❌ LLM ({selected_llm_info['name']}): Chemin non trouvé - `{new_llm_path}`")
                models_valid = False

        # Vérifier le reranker
        if new_reranker_path:
            if os.path.exists(new_reranker_path):
                st.success(f"✅ Reranker (BGE-Reranker): `{new_reranker_path}`")
            else:
                st.error(f"❌ Reranker: Chemin non trouvé - `{new_reranker_path}`")
                models_valid = False
        else:
            st.warning("⚠️ Reranker: Non configuré")

        return {
            "is_local": True,
            "embedding_path": new_embedding_path,
            "llm_path": new_llm_path,
            "reranker_path": new_reranker_path,
            "selected_llm": selected_llm_id,
            "selected_llm_type": selected_llm_info.get("model_type", "mistral"),
            "valid": models_valid
        }

    else:
        st.info("Mode API activé - Utilisation des serveurs Snowflake et DALLEM")
        return {
            "is_local": False,
            "embedding_path": "",
            "llm_path": "",
            "reranker_path": "",
            "valid": True
        }


def initialize_local_models_if_needed():
    """
    Initialise les modèles locaux si le mode local est activé.
    À appeler au démarrage de l'application.
    """
    config = load_config()

    print(f"[CONFIG] Mode actuel: {config.model_mode}")
    print(f"[CONFIG] Config file exists: {os.path.exists(CONFIG_FILE)}")

    if not config.is_local_mode():
        print("[CONFIG] Mode API activé, pas d'initialisation des modèles locaux")
        return

    try:
        from local_models import configure_local_models, local_models_manager

        # Récupérer le chemin et le type du LLM sélectionné
        llm_path = config.get_selected_llm_path()
        llm_type = config.get_selected_llm_type()
        llm_info = AVAILABLE_LOCAL_LLMS.get(config.selected_llm, {})
        llm_name = llm_info.get("name", config.selected_llm)

        print(f"[CONFIG] Configuration des modèles locaux:")
        print(f"  - Embeddings: {config.local_embedding_path}")
        print(f"  - Embeddings existe: {os.path.exists(config.local_embedding_path) if config.local_embedding_path else 'N/A'}")
        print(f"  - LLM ({llm_name}): {llm_path}")
        print(f"  - LLM existe: {os.path.exists(llm_path) if llm_path else 'N/A'}")
        print(f"  - LLM Type: {llm_type}")
        print(f"  - Reranker: {config.local_reranker_path}")
        print(f"  - Reranker existe: {os.path.exists(config.local_reranker_path) if config.local_reranker_path else 'N/A'}")

        configure_local_models(
            embedding_path=config.local_embedding_path,
            llm_path=llm_path,
            reranker_path=config.local_reranker_path,
            llm_type=llm_type
        )

        # Vérifier que les chemins ont bien été configurés
        print(f"[CONFIG] local_models_manager.embedding_path = {local_models_manager.embedding_path}")
        print(f"[CONFIG] local_models_manager.llm_path = {local_models_manager.llm_path}")
        print(f"[CONFIG] local_models_manager.llm_type = {local_models_manager.llm_type}")
        print(f"[CONFIG] Modèles locaux configurés avec succès")

    except ImportError as e:
        print(f"[CONFIG] Impossible de charger le module local_models: {e}")
    except Exception as e:
        print(f"[CONFIG] Erreur lors de l'initialisation des modèles locaux: {e}")
        import traceback
        traceback.print_exc()
