import os
import json
import pandas as pd
from fuzzywuzzy import process
from crsgpt.communicate.communicate import *
from crsgpt.prompter.prompter import *
from concurrent.futures import ThreadPoolExecutor
import pickle
import numpy as np
from tqdm import trange

import logging
import re
from typing import List, Optional
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
from enum import Enum

from typing import Any, Dict, List, Optional
from pydantic import * 
import copy

# from langchain.pydantic_v1 import BaseModel, Extra, root_validator
# from langchain.utils import get_from_dict_or_env
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from typing import Any, Sequence
import asyncio
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Union, cast

import aiohttp
import requests

class Document:
    """Class for storing a piece of text and associated metadata."""

    def __init__(
        self,
        page_content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new Document."""
        self.page_content = page_content
        self.metadata = metadata or {}

def get_from_dict_or_env(
    data: Dict[str, Any], key: str, env_key: str, default: Optional[str] = None
) -> str:
    """Get a value from a dictionary or an environment variable."""
    if key in data and data[key]:
        return data[key]
    else:
        return get_from_env(key, env_key, default=default)


def get_from_env(key: str, env_key: str, default: Optional[str] = None) -> str:
    """Get a value from a dictionary or an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )


class GoogleSearchAPIWrapper(BaseModel):
    """Wrapper for Google Search API.

    Adapted from: Instructions adapted from https://stackoverflow.com/questions/
    37083058/
    programmatically-searching-google-in-python-using-custom-search

    TODO: DOCS for using it
    1. Install google-api-python-client
    - If you don't already have a Google account, sign up.
    - If you have never created a Google APIs Console project,
    read the Managing Projects page and create a project in the Google API Console.
    - Install the library using pip install google-api-python-client

    2. Enable the Custom Search API
    - Navigate to the APIs & Services→Dashboard panel in Cloud Console.
    - Click Enable APIs and Services.
    - Search for Custom Search API and click on it.
    - Click Enable.
    URL for it: https://console.cloud.google.com/apis/library/customsearch.googleapis.com

    3. To create an API key:
    - Navigate to the APIs & Services → Credentials panel in Cloud Console.
    - Select Create credentials, then select API key from the drop-down menu.
    - The API key created dialog box displays your newly created key.
    - You now have an API_KEY

    Alternatively, you can just generate an API key here:
    https://developers.google.com/custom-search/docs/paid_element#api_key

    4. Setup Custom Search Engine so you can search the entire web
    - Create a custom search engine here: https://programmablesearchengine.google.com/.
    - In `What to search` to search, pick the `Search the entire Web` option.
    After search engine is created, you can click on it and find `Search engine ID`
      on the Overview page.

    """

    search_engine: Any  #: :meta private:
    google_api_key: Optional[str] = "AIzaSyCivmran69DztFFl3Vf-AhAUQXE43SQZSA"
    google_cse_id: Optional[str] = "f0871233b543b4836"
    k: int = 10
    siterestrict: bool = False

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _google_search_results(self, search_term: str, **kwargs: Any) -> List[dict]:
        cse = self.search_engine.cse()
        if self.siterestrict:
            cse = cse.siterestrict()
        res = cse.list(q=search_term, cx=self.google_cse_id, **kwargs).execute()
        return res.get("items", [])

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        google_api_key = get_from_dict_or_env(
            values, "google_api_key", "GOOGLE_API_KEY"
        )
        values["google_api_key"] = google_api_key

        google_cse_id = get_from_dict_or_env(values, "google_cse_id", "GOOGLE_CSE_ID")
        values["google_cse_id"] = google_cse_id

        try:
            from googleapiclient.discovery import build

        except ImportError:
            raise ImportError(
                "google-api-python-client is not installed. "
                "Please install it with `pip install google-api-python-client"
                ">=2.100.0`"
            )

        service = build("customsearch", "v1", developerKey=google_api_key)
        values["search_engine"] = service

        return values

    def run(self, query: str) -> str:
        """Run query through GoogleSearch and parse result."""
        snippets = []
        results = self._google_search_results(query, num=self.k)
        if len(results) == 0:
            return "No good Google Search Result was found"
        for result in results:
            if "snippet" in result:
                snippets.append(result["snippet"])

        return " ".join(snippets)

    def results(
        self,
        query: str,
        num_results: int,
        search_params: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """Run query through GoogleSearch and return metadata.

        Args:
            query: The query to search for.
            num_results: The number of results to return.
            search_params: Parameters to be passed on search

        Returns:
            A list of dictionaries with the following keys:
                snippet - The description of the result.
                title - The title of the result.
                link - The link to the result.
        """
        metadata_results = []
        results = self._google_search_results(
            query, num=num_results, **(search_params or {})
        )
        if len(results) == 0:
            return [{"Result": "No good Google Search Result was found"}]
        for result in results:
            metadata_result = {
                "title": result["title"],
                "link": result["link"],
            }
            if "snippet" in result:
                metadata_result["snippet"] = result["snippet"]
            metadata_results.append(metadata_result)

        return metadata_results


class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"

def _split_text_with_regex(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = [_splits[0]] + splits
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]

class RecursiveCharacterTextSplitter:
    """Splitting text by recursively look at characters.

    Recursively tries to split by different characters to find one
    that works.
    """

    def __init__(
        self,
        chunk_size,
        chunk_overlap,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = False,
        add_start_index: bool = False,
        strip_whitespace: bool = True
    ) -> None:
        """Create a new TextSplitter."""
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        self._keep_separator = keep_separator
        self._length_function = len
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._strip_whitespace = strip_whitespace
        self._add_start_index = add_start_index
        

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        if self._strip_whitespace:
            text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits, separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    print(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs


    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        return self._split_text(text, self._separators)
    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)


default_header_template = {
    "User-Agent": "",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*"
    ";q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


class AsyncHtmlLoader:
    """Load `HTML` asynchronously."""

    def __init__(
        self,
        web_path: Union[str, List[str]],
        header_template: Optional[dict] = None,
        verify_ssl: Optional[bool] = True,
        proxies: Optional[dict] = None,
        requests_per_second: int = 2,
        requests_kwargs: Optional[Dict[str, Any]] = None,
        raise_for_status: bool = False,
    ):
        """Initialize with a webpage path."""

        # TODO: Deprecate web_path in favor of web_paths, and remove this
        # left like this because there are a number of loaders that expect single
        # urls
        if isinstance(web_path, str):
            self.web_paths = [web_path]
        elif isinstance(web_path, List):
            self.web_paths = web_path

        headers = header_template or default_header_template
        if not headers.get("User-Agent"):
            try:
                from fake_useragent import UserAgent

                headers["User-Agent"] = UserAgent().random
            except ImportError:
                self.file_logger.info(
                    "fake_useragent not found, using default user agent."
                    "To get a realistic header for requests, "
                    "`pip install fake_useragent`."
                )

        self.session = requests.Session()
        self.session.headers = dict(headers)
        self.session.verify = verify_ssl

        if proxies:
            self.session.proxies.update(proxies)

        self.requests_per_second = requests_per_second
        self.requests_kwargs = requests_kwargs or {}
        self.raise_for_status = raise_for_status

    async def _fetch(
        self, url: str, retries: int = 3, cooldown: int = 2, backoff: float = 1.5
    ) -> str:
        try:
            async with aiohttp.ClientSession() as session:
                for i in range(retries):
                    try:
                        async with session.get(
                            url,
                            headers=self.session.headers,
                            ssl=None if self.session.verify else False,
                            timeout=aiohttp.ClientTimeout(total=4),
                        ) as response:
                            try:
                                text = await response.text()
                            except UnicodeDecodeError:
                                print(f"Failed to decode content from {url}")
                                text = ""
                            return text
                    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                        if i == retries - 1:
                            raise
                        else:
                            print(
                                f"Error fetching {url} with attempt "
                                f"{i + 1}/{retries}: {e}. Retrying..."
                            )
                            await asyncio.sleep(cooldown * backoff**i)
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

    async def _fetch_with_rate_limit(
        self, url: str, semaphore: asyncio.Semaphore
    ) -> str:
        async with semaphore:
            return await self._fetch(url)

    async def fetch_all(self, urls: List[str]) -> Any:
        """Fetch all urls concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.requests_per_second)
        tasks = []
        for url in urls:
            task = asyncio.ensure_future(self._fetch_with_rate_limit(url, semaphore))
            tasks.append(task)
        try:
            from tqdm.asyncio import tqdm_asyncio

            return await tqdm_asyncio.gather(
                *tasks, desc="Fetching pages", ascii=True, mininterval=1
            )
        except ImportError:
            warnings.warn("For better logging of progress, `pip install tqdm`")
            return await asyncio.gather(*tasks)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load text from the url(s) in web_path."""
        for doc in self.load():
            yield doc

    def load(self) -> List[Document]:
        """Load text from the url(s) in web_path."""

        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, self.fetch_all(self.web_paths))
                results = future.result()
        except RuntimeError:
            results = asyncio.run(self.fetch_all(self.web_paths))
        docs = []
        for i, text in enumerate(cast(List[str], results)):
            metadata = {"source": self.web_paths[i]}
            docs.append(Document(page_content=text, metadata=metadata))

        return docs



class Html2TextTransformer:
    """Replace occurrences of a particular search pattern with a replacement string

    Arguments:
        ignore_links: Whether links should be ignored; defaults to True.
        ignore_images: Whether images should be ignored; defaults to True.

    Example:
        .. code-block:: python
            from langchain.document_transformers import Html2TextTransformer
            html2text = Html2TextTransformer()
            docs_transform = html2text.transform_documents(docs)
    """

    def __init__(self, ignore_links: bool = True, ignore_images: bool = True) -> None:
        self.ignore_links = ignore_links
        self.ignore_images = ignore_images

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        try:
            import html2text
        except ImportError:
            raise ImportError(
                """html2text package not found, please 
                install it with `pip install html2text`"""
            )

        # Create a html2text.HTML2Text object and override some properties
        h = html2text.HTML2Text()
        h.ignore_links = self.ignore_links
        h.ignore_images = self.ignore_images

        for d in documents:
            d.page_content = h.handle(d.page_content)
        return documents





class VecotrStore:
    def __init__(self,model = "text-embedding-ada-002"):
        self.vectorstore = {}
        self.model = model
        
    
    def add_documents(self, documents: Iterable[Document]):
        for doc in documents:
            self.vectorstore[doc.page_content] = get_embedding(doc.page_content, self.model)
    
    def similarity_search(self, query: str, top_k: int = 1) -> List[Document]:
        """Find the most similar documents to the query.

        Args:
            query: The query to search for.
            top_k: The number of results to return.

        Returns:
            A list of the most similar documents.
        """
        query_embedding = get_embedding(query, self.model)
        if query_embedding is None:
            return []
        results = []
        
        for page_content, vector in self.vectorstore.items():
            score = distances_from_embeddings(query_embedding, [vector])
            results.append((page_content, score))
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return [Document(page_content=page_content, metadata={"score": score}) for page_content, score in results[:top_k]]
        

class WEB:
    def __init__(self,
                 file_logger,
                 verbose):

        self.file_logger=file_logger
        self.verbose=verbose
        self.text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150
        )
        self.search = GoogleSearchAPIWrapper()
        self.num_search_results = 1
        self.url_database = []
        self.vectorstore = VecotrStore()
        
    def clean_search_query(self, query: str) -> str:
        # Some search tools (e.g., Google) will
        # fail to return results if query has a
        # leading digit: 1. "LangCh..."
        # Check if the first character is a digit
        if query[0].isdigit():
            # Find the position of the first quote
            first_quote_pos = query.find('"')
            if first_quote_pos != -1:
                # Extract the part of the string after the quote
                query = query[first_quote_pos + 1 :]
                # Remove the trailing quote if present
                if query.endswith('"'):
                    query = query[:-1]
        return query.strip()
    
    def search_tool(self, query: str, num_search_results: int = 1) -> List[dict]:
        """Returns num_search_results pages per Google search."""
        # import pdb; pdb.set_trace()
        query_clean = self.clean_search_query(query)
        result = self.search.results(query_clean, num_search_results)
        return result
    
    def get_relevant_documents(
        self,
        goal, instruction, preference, context):
        """Search Google for documents related to the query input.

        Args:
            query: user query

        Returns:
            Relevant documents from all various urls.
        """

        # Get search questions
        self.file_logger.info("Generating questions for Google Search ...")
        queries = general_json_chat(
                self.file_logger,self.verbose,
                compose_messages(
                    {'s':compose_system_prompts(
                        {'prompt':Prompter.SYSTEM_PROMPT},
                        {'prompt':Prompter.WEB_QUERY_PROMPT},
                        {'attribute':'Goal','content':goal},
                        {'attribute':'Instruction','content':instruction},
                        {'attribute':'User Preference','content':preference},
                        {'attribute':'Chat History','content':context[:-1]},
                        {'attribute':'User Input','content':context[-1]}
                    )}
                ),
                "web_search_queries",max_tokens=150
            )
        questions = queries["Queries"]
        self.file_logger.info(f"Questions for Google Search: {questions}")

        # Get urls
        self.file_logger.info("Searching for relevant urls...")
        urls_to_look = []
        for query in questions:
            # Google search
            search_results = self.search_tool(query, self.num_search_results)
            self.file_logger.info("Searching for relevant urls...")
            self.file_logger.info(f"Search results: {search_results}")
            for res in search_results:
                if res.get("link", None):
                    urls_to_look.append(res["link"])

        # Relevant urls
        urls = set(urls_to_look)                  

        # Check for any new urls that we have not processed
        new_urls = list(urls.difference(self.url_database))

        self.file_logger.info(f"New URLs to load: {new_urls}")
        # Load, split, and add new urls to vectorstore
        if new_urls:
            loader = AsyncHtmlLoader(new_urls)
            html2text = Html2TextTransformer()
            self.file_logger.info("Indexing new urls...")
            docs = loader.load()
            docs = list(html2text.transform_documents(docs))
            docs = self.text_splitter.split_documents(docs)
            self.vectorstore.add_documents(docs)
            self.url_database.extend(new_urls)

        # Search for relevant splits
        # TODO: make this async
        self.file_logger.info("Grabbing most relevant splits from urls...")
        docs = []
        for query in questions:
            docs.extend(self.vectorstore.similarity_search(query))

        docs = [doc.page_content for doc in docs]
        return docs
