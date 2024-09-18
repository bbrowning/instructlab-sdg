# Standard
from pathlib import Path
import json
import logging

# Third Party
from datasets import Dataset, concatenate_datasets
from tabulate import tabulate
from transformers import AutoTokenizer
import yaml

# Local
from .chunking import chunk_document

logger = logging.getLogger(__name__)

def fuse_texts(text_list, short_length_threshold=100):
    """
    Fuse short texts with preceding longer texts if their word count is below the threshold.
    Args:
        text_list (list): List of text chunks to process.
        short_length_threshold (int): The word count threshold for determining short texts.
    Returns:
        list: List of fused texts.
    """
    fused_texts = []
    previous_long_text = ""

    for text in text_list:
        word_count = len(text.split())

        if word_count <= short_length_threshold and previous_long_text:
            # Append the short text to the last long text
            fused_texts[-1] += "\n\n" + text
        else:
            # This is a long text, so add it to the list and remember it
            fused_texts.append(text)
            previous_long_text = text

    return fused_texts

def create_tokenizer():
    """
    Create a tokenizer instance from a pre-trained model.
    Returns:
        AutoTokenizer: The tokenizer instance.
    """
    return AutoTokenizer.from_pretrained("instructlab/granite-7b-lab") # model name needs to be fixed


def get_token_count(text, tokenizer):
    """
    Get the number of tokens in a text using the provided tokenizer.
    Args:
        text (str): The text to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.
    Returns:
        int: Number of tokens.
    """
    return len(tokenizer.tokenize(text))


def add_heading_formatting(text):
    """
    Add heading formatting to the text if the first part is short.
    Args:
        text (str): The input text to format.
    Returns:
        str: Formatted text with headings applied.
    """
    text = text.split(".")
    
    # Change this from hardcoded to something more flexible
    # Docling fails at identifying the header and gives a single word instead.
    # Hence we split the sentences and check if header was mistakenly identified as a paragaraph.

    if len(text) > 1 and len(text[0].split(" ")) < 3:
        text = f"**{text[0]}**" + ".".join(text[1:])
    else:
        text = ".".join(text)
    return text


def generate_table_from_parsed_rep(item):
    """
    Generate the table from the parsed representation and return as a string.
    Args:
        item (dict): Parsed representation of a table.
    Returns:
        str: Formatted table as a string.
    """
    caption = ""
    if "text" in item:
        caption = item["text"]

    data = item["data"]

    if len(data) <= 1 or len(data[0]) <= 1:
        return ""

    table = []
    for i, row in enumerate(data):
        trow = []
        for j, cell in enumerate(row):
            trow.append(cell["text"])
        table.append(trow)

    table_text = tabulate(table, tablefmt="github")
    if caption:
        table_text += f"\nCaption: {caption}\n"
    return table_text


def get_table(json_book, table_ref):
    """
    Retrieve a table from a document based on a reference string.
    Args:
        json_book (dict): JSON representation of the document.
        table_ref (str): Reference path to the table within the document.
    Returns:
        str: Formatted table string.
    """
    parts = table_ref.split("/")
    table_text = generate_table_from_parsed_rep(json_book[parts[1]][int(parts[2])])
    return table_text


def get_table_page_number(json_book, idx):
    """
    Get the page number of a table or other document element.
    Args:
        json_book (dict): JSON representation of the document.
        idx (int): Index of the element in the document.
    Returns:
        int: Page number of the element.
    """
    prev_page_num, next_page_num = None, None
    for book_element in json_book["main-text"][idx - 1 :: -1]:
        if "prov" in book_element:
            prev_page_num = book_element["prov"][0]["page"]
            break
    for book_element in json_book["main-text"][idx:]:
        if "prov" in book_element:
            next_page_num = book_element["prov"][0]["page"]
            break
    if prev_page_num is not None and next_page_num is not None:
        if prev_page_num == next_page_num:
            return prev_page_num
        else:
            return next_page_num
    elif prev_page_num is not None:
        return prev_page_num
    elif next_page_num is not None:
        return next_page_num


def build_chunks_from_docling_json(
    json_book,
    max_token_per_chunk,
    tokenizer,
    keep_same_page_thing_together=False,
    chunking_criteria=None,
):
    """
    Build document chunks from a docling JSON representation.
    Args:
        json_book (dict): JSON document to process.
        max_token_per_chunk (int): Maximum token count per chunk.
        tokenizer (AutoTokenizer): Tokenizer instance to use.
        keep_same_page_thing_together (bool): Whether to keep content on the same page together.
        chunking_criteria (callable): Custom function for determining chunk breaks.
    Returns:
        list: List of document chunks.
    """
    current_buffer = []
    document_chunks = []
    prev_page_number = None
    book_title = None

    for idx, book_element in enumerate(json_book["main-text"]):
        if book_element["type"] in [
            "page-footer",
            "picture",
            "reference",
            "meta-data",
            "figure",
            "page-header",
        ]:
            continue
        elif book_element["type"] == "footnote":
            current_book_page_number = book_element["prov"][0]["page"]
        elif book_element["type"] in [
            "subtitle-level-1",
            "paragraph",
            "table",
            "title",
            "equation",
        ]:
            if book_element["type"] == "table":
                current_book_page_number = get_table_page_number(json_book, idx)
            else:
                current_book_page_number = book_element["prov"][0]["page"]
                book_text = book_element["text"]

            if book_element["type"] == "subtitle-level-1":
                if book_title is None:
                    book_title = book_text
                    book_text = f"# Title: **{book_text}**"
                else:
                    book_text = f"## **{book_text}**"

            if book_element["type"] == "title":
                book_text = f"# **{book_text}**"
            if book_element["type"] == "page-header":
                book_text = f"Page Header: **{book_text}**\n\n"

            if chunking_criteria is not None:
                # custom break function that can be used to chunk document
                if chunking_criteria(book_text):
                    document_chunks.append("\n\n".join(current_buffer))
                    current_buffer = []
            elif (
                prev_page_number is not None
                and prev_page_number != current_book_page_number
            ) and keep_same_page_thing_together:
                document_chunks.append("\n\n".join(current_buffer))
                current_buffer = []
            else:
                if (
                    get_token_count("\n\n".join(current_buffer), tokenizer)
                    >= max_token_per_chunk
                    and len(current_buffer) > 1
                ):
                    document_chunks.append("\n\n".join(current_buffer[:-1]))

                    if (
                        get_token_count(current_buffer[-1], tokenizer)
                        >= max_token_per_chunk
                    ):
                        document_chunks.append(current_buffer[-1])
                        current_buffer = []
                    else:
                        current_buffer = current_buffer[-1:]

            if book_element["type"] == "paragraph":
                book_text = add_heading_formatting(book_text)
            elif book_element["type"] == "table":
                book_text = get_table(json_book, book_element["$ref"])

            if "## References" in book_text or "## Acknowledgements" in book_text:
                # For research papers we ignore everything after these sections
                break
            current_buffer.append(book_text)

        try:
            prev_page_number = current_book_page_number
        except Exception as e:
            logger.error(f"Error processing book element: {book_element}, {str(e)}")

    if "\n\n".join(current_buffer) not in document_chunks:
        document_chunks.append("\n\n".join(current_buffer))
    return document_chunks


def safe_concatenate_datasets(datasets: list):
    """
    Concatenate datasets safely, ignoring any datasets that are None or empty.
    """
    filtered_datasets = [ds for ds in datasets if ds is not None and ds.num_rows > 0]

    if not filtered_datasets:
        return None

    return concatenate_datasets(filtered_datasets)

class DocProcessor:
    def __init__(
        self,
        parsed_doc_dir: Path,
        tokenizer: str = "instructlab/granite-7b-lab",
        user_config_path: Path = None,
    ):
        self.parsed_doc_dir = self._path_validator(parsed_doc_dir)
        self.user_config = self._load_user_config(
            self._path_validator(user_config_path)
        )
        self.docling_jsons = list(self.parsed_doc_dir.glob("*.json"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def _path_validator(self, path) -> Path:
        """
        Validate the path and return a Path object.
        Args:
            path (str): Path to be validated.
        Returns:
            Path`: Path object.
        """
        if isinstance(path, str):
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"{path} does not exist.")
        return path

    def _load_user_config(self, user_config_path: Path) -> dict:
        """
        Load the user config file.
        Args:
            user_config_path (Path): Path to the user config file.
        Returns:
            dict: User config dictionary.
        """
        # load user config as yaml
        with open(user_config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _process_parsed_docling_json(self, json_fp: Path) -> Dataset:
        """
        Process the parsed docling json file and return a dataset.
        Args:
            json_fp (str): Path to the parsed docling json file.
        Returns:
            Dataset: Dataset object.
        """
        logger.info(f"Processing parsed docling json file: {json_fp}")
        with open(json_fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = json_fp.name.split(".")[0]
        chunks = build_chunks_from_docling_json(
            data,
            max_token_per_chunk=500,
            tokenizer=self.tokenizer,
        )
        chunks = fuse_texts(chunks, 200)
        return Dataset.from_dict(
            {
                "document": chunks,
                "document_outline": [self.user_config["document_outline"]]
                * len(chunks),
                "document_title": [file_name] * len(chunks),
                "domain": [self.user_config["domain"]] * len(chunks),
            }
        )

    def _add_icls(self, chunked_document: Dataset) -> Dataset:
        """
        Add the ICLS label to the dataset.
        Args:
            dataset (Dataset): Dataset object.
        Returns:
            Dataset: Dataset object with ICLS label.
        """
        icl = self.user_config["seed_examples"]
        chunked_document_all_icl = []
        for icl_ in icl:
            chunked_document_all_icl.append(
                chunked_document.map(
                    lambda x: {
                        "icl_document": icl_["context"],
                        "icl_query_1": icl_["questions_and_answers"][0]["question"],
                        "icl_response_1": icl_["questions_and_answers"][0]["answer"],
                        "icl_query_2": icl_["questions_and_answers"][1]["question"],
                        "icl_response_2": icl_["questions_and_answers"][1]["answer"],
                        "icl_query_3": icl_["questions_and_answers"][2]["question"],
                        "icl_response_3": icl_["questions_and_answers"][2]["answer"],
                    }
                )
            )
        chunked_document_all_icl = safe_concatenate_datasets(chunked_document_all_icl)
        chunked_document_all_icl = chunked_document_all_icl.map(
            lambda x: {
                "chunks": chunk_document(
                    [x["document"]], server_ctx_size=4096, chunk_word_count=1024
                )
                if get_token_count(x["document"], self.tokenizer) > 1024
                else [x["document"]]
            }
        )
        df = chunked_document_all_icl.to_pandas()
        df_exploded = df.explode("chunks").reset_index(drop=True)
        new_ds = Dataset.from_pandas(df_exploded)
        new_ds = new_ds.remove_columns("document").rename_columns(
            {"chunks": "document"}
        )

        # Only keep document greater than 100 tokens
        new_ds = new_ds.filter(
            lambda x: get_token_count(x["document"], self.tokenizer) > 100
        )
        return new_ds

    def get_processed_dataset(self) -> Dataset:
        """
        Process all the parsed docling json files and return a dataset.
        Returns:
            Dataset: Dataset object.
        """
        datasets = []
        for json_fp in self.docling_jsons:
            chunk_ds = self._process_parsed_docling_json(json_fp)
            chunk_ds_with_icls = self._add_icls(chunk_ds)
            datasets.append(chunk_ds_with_icls)
        return safe_concatenate_datasets(datasets)
