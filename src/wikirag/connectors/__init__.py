from wikirag.connectors.base import BaseConnector
from wikirag.connectors.local_files import LocalFilesConnector
from wikirag.connectors.mediawiki import MediaWikiConnector
from wikirag.connectors.web import GenericWebConnector

__all__ = ["BaseConnector", "MediaWikiConnector", "GenericWebConnector", "LocalFilesConnector"]
