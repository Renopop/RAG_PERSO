# confluence_connector.py
# Module pour l'intégration Confluence avec le système RAG
# Permet de récupérer les pages d'un espace Confluence entier

import os
import re
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import html
from html.parser import HTMLParser

try:
    import requests
    from requests.auth import HTTPBasicAuth
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


logger = logging.getLogger(__name__)


# =====================================================================
#  CONFIGURATION CONFLUENCE
# =====================================================================

@dataclass
class ConfluenceConfig:
    """Configuration pour la connexion Confluence."""
    base_url: str = ""  # URL de base Confluence (ex: https://confluence.company.com)
    username: str = ""  # Identifiant utilisateur
    password: str = ""  # Mot de passe ou token API
    space_key: str = ""  # Clé de l'espace à charger (ex: "PROJ")

    # Options de synchronisation
    sync_frequency_days: int = 7  # Fréquence de synchro (hebdomadaire par défaut)
    last_sync: Optional[str] = None  # Date ISO de la dernière synchro
    include_attachments: bool = True  # Inclure les pièces jointes
    include_comments: bool = False  # Inclure les commentaires
    page_limit: int = 500  # Nombre max de pages à charger

    # Filtres
    exclude_labels: List[str] = field(default_factory=list)  # Labels à exclure
    include_only_labels: List[str] = field(default_factory=list)  # Labels requis (vide = tous)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ConfluenceConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_valid(self) -> Tuple[bool, str]:
        """Vérifie si la configuration est valide."""
        if not self.base_url:
            return False, "URL Confluence manquante"
        if not self.username:
            return False, "Identifiant manquant"
        if not self.password:
            return False, "Mot de passe/token manquant"
        if not self.space_key:
            return False, "Clé d'espace manquante"
        return True, "OK"

    def should_sync(self) -> bool:
        """Vérifie si une synchronisation est nécessaire."""
        if not self.last_sync:
            return True
        try:
            last = datetime.fromisoformat(self.last_sync)
            return datetime.now() - last >= timedelta(days=self.sync_frequency_days)
        except:
            return True


# =====================================================================
#  PARSER HTML SIMPLE (sans BeautifulSoup)
# =====================================================================

class SimpleHTMLTextExtractor(HTMLParser):
    """Extracteur de texte simple depuis HTML (fallback sans BeautifulSoup)."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.skip_tags = {'script', 'style', 'head', 'meta', 'link'}
        self.block_tags = {'p', 'div', 'br', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                          'li', 'tr', 'td', 'th', 'blockquote', 'pre'}

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag.lower()
        if tag.lower() in self.block_tags:
            self.text_parts.append('\n')

    def handle_endtag(self, tag):
        if tag.lower() in self.block_tags:
            self.text_parts.append('\n')
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            self.text_parts.append(data)

    def get_text(self) -> str:
        text = ''.join(self.text_parts)
        # Nettoyer les espaces multiples
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        return text.strip()


def html_to_text(html_content: str) -> str:
    """Convertit du HTML en texte brut."""
    if not html_content:
        return ""

    # Utiliser BeautifulSoup si disponible
    if HAS_BS4:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Supprimer les scripts et styles
        for tag in soup(['script', 'style', 'head', 'meta', 'link']):
            tag.decompose()

        # Extraire le texte
        text = soup.get_text(separator='\n')
    else:
        # Fallback sans BeautifulSoup
        parser = SimpleHTMLTextExtractor()
        try:
            parser.feed(html_content)
            text = parser.get_text()
        except:
            # Si le parsing échoue, nettoyer basiquement
            text = re.sub(r'<[^>]+>', ' ', html_content)
            text = html.unescape(text)

    # Nettoyer le texte
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)

    return text


# =====================================================================
#  CLIENT CONFLUENCE
# =====================================================================

class ConfluenceClient:
    """Client pour l'API REST Confluence."""

    def __init__(self, config: ConfluenceConfig):
        if not HAS_REQUESTS:
            raise ImportError("Le module 'requests' est requis pour Confluence. Installez-le avec: pip install requests")

        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.auth = HTTPBasicAuth(config.username, config.password)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def test_connection(self) -> Tuple[bool, str]:
        """Teste la connexion à Confluence."""
        try:
            # Essayer de récupérer les infos de l'espace
            response = self.session.get(
                f"{self.base_url}/rest/api/space/{self.config.space_key}",
                timeout=10
            )

            if response.status_code == 200:
                space_info = response.json()
                space_name = space_info.get('name', self.config.space_key)
                return True, f"Connexion réussie ! Espace: {space_name}"
            elif response.status_code == 401:
                return False, "Authentification échouée. Vérifiez vos identifiants."
            elif response.status_code == 404:
                return False, f"Espace '{self.config.space_key}' non trouvé."
            else:
                return False, f"Erreur HTTP {response.status_code}: {response.text[:200]}"

        except requests.exceptions.ConnectionError:
            return False, f"Impossible de se connecter à {self.base_url}"
        except requests.exceptions.Timeout:
            return False, "Timeout lors de la connexion"
        except Exception as e:
            return False, f"Erreur: {str(e)}"

    def get_space_info(self) -> Optional[Dict]:
        """Récupère les informations de l'espace."""
        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/space/{self.config.space_key}",
                params={'expand': 'description.plain,homepage'},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None

    def get_all_pages(self, progress_callback=None) -> List[Dict]:
        """
        Récupère toutes les pages d'un espace Confluence.

        Args:
            progress_callback: Fonction callback(current, total, message) pour le suivi

        Returns:
            Liste des pages avec leur contenu
        """
        pages = []
        start = 0
        limit = 50  # Confluence limite généralement à 50-100 résultats par requête
        total = None

        while True:
            try:
                # Construire l'URL avec expansion du body
                params = {
                    'spaceKey': self.config.space_key,
                    'start': start,
                    'limit': limit,
                    'expand': 'body.storage,version,ancestors,metadata.labels',
                    'status': 'current'  # Seulement les pages actives
                }

                response = self.session.get(
                    f"{self.base_url}/rest/api/content",
                    params=params,
                    timeout=60
                )

                if response.status_code != 200:
                    logger.error(f"Erreur API Confluence: {response.status_code}")
                    break

                data = response.json()
                results = data.get('results', [])

                if total is None:
                    total = data.get('size', 0) + data.get('_links', {}).get('next', 0)

                if not results:
                    break

                for page_data in results:
                    page = self._process_page(page_data)
                    if page:
                        # Appliquer les filtres de labels
                        if self._should_include_page(page):
                            pages.append(page)

                if progress_callback:
                    progress_callback(len(pages), self.config.page_limit,
                                     f"Chargement des pages... ({len(pages)} récupérées)")

                # Vérifier s'il y a plus de pages
                if len(results) < limit or len(pages) >= self.config.page_limit:
                    break

                start += limit

            except Exception as e:
                logger.error(f"Erreur lors de la récupération des pages: {e}")
                break

        logger.info(f"Récupéré {len(pages)} pages de l'espace {self.config.space_key}")
        return pages

    def _process_page(self, page_data: Dict) -> Optional[Dict]:
        """Traite les données d'une page Confluence."""
        try:
            page_id = page_data.get('id')
            title = page_data.get('title', 'Sans titre')

            # Extraire le contenu HTML
            body = page_data.get('body', {}).get('storage', {}).get('value', '')

            # Convertir en texte
            text_content = html_to_text(body)

            if not text_content or len(text_content.strip()) < 10:
                return None

            # Extraire les métadonnées
            version = page_data.get('version', {}).get('number', 1)
            last_modified = page_data.get('version', {}).get('when', '')

            # Extraire les labels
            labels = []
            labels_data = page_data.get('metadata', {}).get('labels', {}).get('results', [])
            for label in labels_data:
                labels.append(label.get('name', ''))

            # Construire le chemin hiérarchique
            ancestors = page_data.get('ancestors', [])
            path_parts = [a.get('title', '') for a in ancestors]
            path_parts.append(title)
            page_path = ' > '.join(path_parts)

            # URL de la page
            page_url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"

            return {
                'id': page_id,
                'title': title,
                'content': text_content,
                'path': page_path,
                'url': page_url,
                'version': version,
                'last_modified': last_modified,
                'labels': labels,
                'space_key': self.config.space_key,
                'content_hash': hashlib.md5(text_content.encode()).hexdigest()
            }

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la page: {e}")
            return None

    def _should_include_page(self, page: Dict) -> bool:
        """Vérifie si une page doit être incluse selon les filtres."""
        labels = page.get('labels', [])

        # Vérifier les labels à exclure
        if self.config.exclude_labels:
            for label in labels:
                if label in self.config.exclude_labels:
                    return False

        # Vérifier les labels requis
        if self.config.include_only_labels:
            if not any(label in self.config.include_only_labels for label in labels):
                return False

        return True

    def get_page_attachments(self, page_id: str) -> List[Dict]:
        """Récupère les pièces jointes d'une page."""
        if not self.config.include_attachments:
            return []

        try:
            response = self.session.get(
                f"{self.base_url}/rest/api/content/{page_id}/child/attachment",
                params={'expand': 'version'},
                timeout=30
            )

            if response.status_code != 200:
                return []

            attachments = []
            for att in response.json().get('results', []):
                attachments.append({
                    'id': att.get('id'),
                    'title': att.get('title'),
                    'media_type': att.get('metadata', {}).get('mediaType', ''),
                    'download_url': f"{self.base_url}{att.get('_links', {}).get('download', '')}"
                })

            return attachments

        except Exception as e:
            logger.error(f"Erreur récupération pièces jointes: {e}")
            return []


# =====================================================================
#  GESTIONNAIRE DE SYNCHRONISATION
# =====================================================================

class ConfluenceSyncManager:
    """Gestionnaire de synchronisation Confluence vers RAG."""

    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, 'confluence_config.json')
        self.sync_state_file = os.path.join(config_dir, 'confluence_sync_state.json')

        os.makedirs(config_dir, exist_ok=True)

    def load_config(self) -> ConfluenceConfig:
        """Charge la configuration Confluence."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return ConfluenceConfig.from_dict(data)
            except Exception as e:
                logger.error(f"Erreur chargement config Confluence: {e}")

        return ConfluenceConfig()

    def save_config(self, config: ConfluenceConfig) -> bool:
        """Sauvegarde la configuration Confluence."""
        try:
            # Ne pas sauvegarder le mot de passe en clair
            config_dict = config.to_dict()
            # Option: chiffrer le mot de passe ou utiliser un gestionnaire de secrets

            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde config Confluence: {e}")
            return False

    def load_sync_state(self) -> Dict:
        """Charge l'état de synchronisation."""
        if os.path.exists(self.sync_state_file):
            try:
                with open(self.sync_state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {'pages': {}, 'last_full_sync': None}

    def save_sync_state(self, state: Dict) -> bool:
        """Sauvegarde l'état de synchronisation."""
        try:
            with open(self.sync_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Erreur sauvegarde état sync: {e}")
            return False

    def get_pages_to_update(self, pages: List[Dict], sync_state: Dict) -> Tuple[List[Dict], List[str]]:
        """
        Compare les pages récupérées avec l'état précédent.

        Returns:
            Tuple (pages_to_add_or_update, page_ids_to_delete)
        """
        current_page_ids = {p['id'] for p in pages}
        previous_pages = sync_state.get('pages', {})
        previous_page_ids = set(previous_pages.keys())

        # Pages à ajouter ou mettre à jour
        pages_to_update = []
        for page in pages:
            page_id = page['id']
            content_hash = page['content_hash']

            if page_id not in previous_pages:
                # Nouvelle page
                pages_to_update.append(page)
            elif previous_pages[page_id].get('content_hash') != content_hash:
                # Page modifiée
                pages_to_update.append(page)

        # Pages supprimées
        pages_to_delete = list(previous_page_ids - current_page_ids)

        return pages_to_update, pages_to_delete

    def prepare_for_ingestion(self, pages: List[Dict], output_dir: str) -> str:
        """
        Prépare les pages pour l'ingestion RAG.
        Crée un CSV et des fichiers texte temporaires.

        Returns:
            Chemin vers le CSV généré
        """
        import csv

        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'confluence_pages.csv')

        rows = []
        for page in pages:
            # Créer un fichier texte pour chaque page
            safe_title = re.sub(r'[^\w\s-]', '_', page['title'])[:50]
            filename = f"confluence_{page['id']}_{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)

            # Écrire le contenu avec métadonnées
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {page['title']}\n\n")
                f.write(f"**Espace:** {page['space_key']}\n")
                f.write(f"**Chemin:** {page['path']}\n")
                f.write(f"**URL:** {page['url']}\n")
                if page['labels']:
                    f.write(f"**Labels:** {', '.join(page['labels'])}\n")
                f.write(f"\n---\n\n")
                f.write(page['content'])

            rows.append({
                'path': filepath,
                'group': f"confluence_{page['space_key']}"
            })

        # Créer le CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['path', 'group'])
            writer.writeheader()
            writer.writerows(rows)

        return csv_path


# =====================================================================
#  FONCTIONS UTILITAIRES
# =====================================================================

def check_dependencies() -> Tuple[bool, List[str]]:
    """Vérifie les dépendances requises pour Confluence."""
    missing = []

    if not HAS_REQUESTS:
        missing.append("requests")

    if not HAS_BS4:
        # BeautifulSoup est optionnel mais recommandé
        pass

    return len(missing) == 0, missing


def get_next_sync_date(config: ConfluenceConfig) -> Optional[datetime]:
    """Calcule la prochaine date de synchronisation."""
    if not config.last_sync:
        return None

    try:
        last = datetime.fromisoformat(config.last_sync)
        return last + timedelta(days=config.sync_frequency_days)
    except:
        return None


def format_sync_status(config: ConfluenceConfig) -> str:
    """Formate le statut de synchronisation pour l'affichage."""
    if not config.last_sync:
        return "Jamais synchronisé"

    try:
        last = datetime.fromisoformat(config.last_sync)
        next_sync = get_next_sync_date(config)

        status = f"Dernière synchro: {last.strftime('%d/%m/%Y %H:%M')}"

        if next_sync:
            if datetime.now() >= next_sync:
                status += " | ⚠️ Synchronisation recommandée"
            else:
                status += f" | Prochaine: {next_sync.strftime('%d/%m/%Y')}"

        return status
    except:
        return "Erreur de date"
