# SPDX-License-Identifier: Apache-2.0

# Standard
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union
import glob
import logging
import os
import re
import tempfile

# Third Party
from docling_parse.docling_parse import pdf_parser
from instructlab.schema.taxonomy import DEFAULT_TAXONOMY_FOLDERS as TAXONOMY_FOLDERS
from instructlab.schema.taxonomy import (
    TaxonomyMessageFormat,
    TaxonomyParser,
    TaxonomyReadingException,
)
import git
import gitdb
import yaml

# Local
from . import docprocessor

# Initialize the pdf parser
parser = pdf_parser()

logger = logging.getLogger(__name__)
DOC_FILEPATH = Path("~/.local/share/instructlab/documents").expanduser()


def _is_taxonomy_file(fn: str) -> bool:
    path = Path(fn)
    if path.parts[0] not in TAXONOMY_FOLDERS:
        return False
    if path.name == "qna.yaml":
        return True
    if path.name.casefold() in {"qna.yml", "qna.yaml"}:
        # warning for incorrect extension or case variants
        logger.warning(
            "Found a '%s' file: %s: taxonomy files must be named 'qna.yaml'. File will not be checked.",
            path.name,
            path,
        )
    return False


def _get_taxonomy_diff(
    repo_path: str | Path = "taxonomy", base: str = "origin/main"
) -> list[str]:
    repo = git.Repo(repo_path)
    untracked_files = [u for u in repo.untracked_files if _is_taxonomy_file(u)]

    branches = [b.name for b in repo.branches]

    head_commit = None
    if "/" in base:
        re_git_branch = re.compile(f"remotes/{base}$", re.MULTILINE)
    elif base in branches:
        re_git_branch = re.compile(f"{base}$", re.MULTILINE)
    else:
        try:
            head_commit = repo.commit(base)
        except gitdb.exc.BadName as e:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy git ref "{base}" from the current HEAD'
                )
            ) from e

    # Move backwards from HEAD until we find the first commit that is part of base
    # then we can take our diff from there
    current_commit = repo.commit("HEAD")
    while not head_commit:
        branches = repo.git.branch("-a", "--contains", current_commit.hexsha)
        if re_git_branch.findall(branches):
            head_commit = current_commit
            break
        try:
            current_commit = current_commit.parents[0]
        except IndexError as e:
            raise SystemExit(
                yaml.YAMLError(
                    f'Couldn\'t find the taxonomy base branch "{base}" from the current HEAD'
                )
            ) from e

    modified_files = [
        d.b_path
        for d in head_commit.diff(None)
        if not d.deleted_file and _is_taxonomy_file(d.b_path)
    ]

    updated_taxonomy_files = list(set(untracked_files + modified_files))
    return updated_taxonomy_files


def _get_taxonomy(repo="taxonomy"):
    repo = Path(repo)
    taxonomy_file_paths = []
    for root, _, files in os.walk(repo):
        for file in files:
            file_path = Path(root).joinpath(file).relative_to(repo)
            if _is_taxonomy_file(file_path):
                taxonomy_file_paths.append(str(file_path))
    return taxonomy_file_paths


def _get_documents(
    source: Dict[str, Union[str, List[str]]],
    skip_checkout: bool = False,
    output_dir: Path = Path(),
) -> Tuple[List[str], List[Path]]:
    """
    Retrieve the content of files (Markdown and PDF) from a Git repository.

    Args:
        source (dict): Source info containing repository URL, commit hash, and list of file patterns.
        skip_checkout (bool, optional): If True, skips checking out the specific commit. Defaults to False.
        output_dir (Path, optional): Directory to clone the repository into. Defaults to current directory.

    Returns:
        Tuple[List[str], List[Path]]:
            - List of document contents (Markdown as text and PDFs as extracted text).
            - List of corresponding file paths.

    Raises:
        SystemExit: If no valid documents are found.
        OSError, GitCommandError, FileNotFoundError: For errors during Git operations or file access.
    """
    repo_url = source.get("repo")
    commit_hash = source.get("commit")
    file_patterns = source.get("patterns", [])

    try:
        repo = git.Repo.clone_from(repo_url, output_dir)

        if not skip_checkout and commit_hash:
            repo.git.checkout(commit_hash)

        file_contents = []
        filepaths = []

        logger.info("Processing files...")
        for pattern in file_patterns:
            # Use glob to find files matching the pattern
            matched_files = glob.glob(
                os.path.join(repo.working_dir, pattern), recursive=True
            )
            logger.info(f"Pattern '{pattern}' matched {len(matched_files)} files.")

            for file_path in matched_files:
                if os.path.isfile(file_path):
                    logger.info(f"Processing file: {file_path}")
                    try:
                        if file_path.lower().endswith(".md"):
                            # Process Markdown files
                            with open(file_path, "r", encoding="utf-8") as file:
                                content = file.read()
                                file_contents.append(content)
                                filepaths.append(Path(file_path))
                                logger.info(
                                    f"Appended Markdown content from {file_path}"
                                )

                        elif file_path.lower().endswith(".pdf"):
                            # Process PDF files using docling_parse's pdf_parser
                            doc_key = f"key_{os.path.basename(file_path)}"  # Unique document key
                            logger.info(f"Loading PDF document from {file_path}")

                            success = parser.load_document(doc_key, file_path)
                            if not success:
                                logger.warning(
                                    f"Failed to load PDF document: {file_path}"
                                )
                                continue

                            num_pages = parser.number_of_pages(doc_key)
                            logger.info(f"PDF '{file_path}' has {num_pages} pages.")

                            pdf_text = ""

                            for page in range(num_pages):
                                try:
                                    json_doc = parser.parse_pdf_from_key_on_page(
                                        doc_key, page
                                    )
                                    if "pages" not in json_doc or not json_doc["pages"]:
                                        logger.warning(
                                            f"Page {page + 1} could not be parsed in '{file_path}'"
                                        )
                                        continue

                                    json_page = json_doc["pages"][0]

                                    # Extract text from cells
                                    for cell in json_page.get("cells", []):
                                        text = cell.get("content", {}).get(
                                            "rnormalized", ""
                                        )
                                        if text.strip():  # Only append non-empty text
                                            pdf_text += text.strip() + "\n"
                                except Exception as page_error:
                                    logger.warning(
                                        f"Error parsing page {page + 1} of '{file_path}': {page_error}"
                                    )
                                    continue

                            if pdf_text:
                                file_contents.append(pdf_text)
                                filepaths.append(Path(file_path))

                            # Unload the document to free memory
                            parser.unload_document(doc_key)
                            logger.info(f"Unloaded PDF document: {file_path}")

                        else:
                            logger.info(f"Skipping unsupported file type: {file_path}")
                    except Exception as file_error:
                        logger.error(
                            f"Error processing file '{file_path}': {file_error}"
                        )
                        continue
                else:
                    logger.info(f"Skipping non-file path: {file_path}")

        if file_contents:
            return file_contents, filepaths
        raise SystemExit("Couldn't find knowledge documents")

    except (OSError, git.exc.GitCommandError, FileNotFoundError) as e:
        logger.error("Error retrieving documents: %s", str(e))
        raise e


# pylint: disable=broad-exception-caught
def _read_taxonomy_file(
    file_path: str | Path, yamllint_config: str | None = None, output_dir: Path = Path()
):
    seed_instruction_data = []

    parser = TaxonomyParser(
        schema_version=0,  # Use version value in yaml
        message_format=TaxonomyMessageFormat.LOGGING,  # Report warnings and errors to the logger
        yamllint_config=yamllint_config,
        yamllint_strict=True,  # Report yamllint warnings as errors
    )
    taxonomy = parser.parse(file_path)

    if taxonomy.warnings or taxonomy.errors:
        return seed_instruction_data, taxonomy.warnings, taxonomy.errors

    try:
        # get seed instruction data
        tax_path = "->".join(taxonomy.path.parent.parts)
        contents = taxonomy.contents
        task_description = contents.get("task_description", None)
        domain = contents.get("domain")
        documents = contents.get("document")
        print(f"")
        if documents:
            date_suffix = (
                datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
            )
            document_contents, doc_filepaths = _get_documents(
                source=documents, output_dir=output_dir / f"documents-{date_suffix}"
            )
            logger.debug("Content from git repo fetched")

        for seed_example in contents.get("seed_examples"):
            context = seed_example.get("context", "")
            if "questions_and_answers" in seed_example:
                question_answer_list = seed_example.get("questions_and_answers")
                seed_instruction_data.append(
                    {
                        "questions_and_answers": question_answer_list,
                        "context": context,
                        "taxonomy_path": tax_path,
                        "documents": document_contents,
                        "filepaths": doc_filepaths,
                        "domain": domain,
                        "document_outline": contents.get("document_outline"),
                    }
                )
            else:
                question = seed_example.get("question")
                answer = seed_example.get("answer")

                seed_instruction_data.append(
                    {
                        "instruction": question,
                        "input": context,
                        "output": answer,
                        "taxonomy_path": tax_path,
                        "task_description": task_description,
                        "document": documents,
                        "domain": domain,
                    }
                )
    except Exception as e:
        raise TaxonomyReadingException(f"Exception {e} raised in {file_path}") from e

    return seed_instruction_data, 0, 0


def read_taxonomy(
    taxonomy: str | Path,
    taxonomy_base: str,
    yaml_rules: str | None = None,
    output_dir: Path = Path(),
):
    yamllint_config = None  # If no custom rules file, use default config
    if yaml_rules is not None:  # user attempted to pass custom rules file
        yaml_rules_path = Path(yaml_rules)
        if yaml_rules_path.is_file():  # file was found, use specified config
            logger.debug("Using YAML rules from %s", yaml_rules)
            yamllint_config = yaml_rules_path.read_text(encoding="utf-8")
        else:
            logger.debug("Cannot find %s. Using default rules.", yaml_rules)

    seed_instruction_data = []
    is_file = os.path.isfile(taxonomy)
    if is_file:  # taxonomy is file
        seed_instruction_data, warnings, errors = _read_taxonomy_file(
            taxonomy, yamllint_config, output_dir
        )
        if warnings:
            logger.warning(
                f"{warnings} warnings (see above) due to taxonomy file not (fully) usable."
            )
        if errors:
            raise SystemExit(yaml.YAMLError("Taxonomy file with errors! Exiting."))
    else:  # taxonomy is dir
        if taxonomy_base == "empty":
            # Gather all the yamls - equivalent to a diff against "the null tree"
            taxonomy_files = _get_taxonomy(taxonomy)
        else:
            # Gather the new or changed YAMLs using git diff, including untracked files
            taxonomy_files = _get_taxonomy_diff(taxonomy, taxonomy_base)
        total_errors = 0
        total_warnings = 0
        if taxonomy_files:
            logger.debug("Found taxonomy files:")
            for e in taxonomy_files:
                logger.debug(f"* {e}")
        for f in taxonomy_files:
            file_path = os.path.join(taxonomy, f)
            data, warnings, errors = _read_taxonomy_file(
                file_path, yamllint_config, output_dir
            )
            total_warnings += warnings
            total_errors += errors
            if data:
                seed_instruction_data.extend(data)
        if total_warnings:
            logger.warning(
                f"{total_warnings} warnings (see above) due to taxonomy files that were not (fully) usable."
            )
        if total_errors:
            raise SystemExit(
                yaml.YAMLError(f"{total_errors} taxonomy files with errors! Exiting.")
            )
    return seed_instruction_data


def read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules, output_dir):
    seed_instruction_data = read_taxonomy(
        taxonomy, taxonomy_base, yaml_rules, output_dir
    )

    # Transform into a more convenient format to feed into our updated SDG library
    leaf_nodes = {}
    for seed in seed_instruction_data:
        node = leaf_nodes.setdefault(seed["taxonomy_path"], [])
        node.append(seed)
        leaf_nodes[seed["taxonomy_path"]] = node

    return leaf_nodes


def _knowledge_leaf_node_to_samples(
    leaf_node, server_ctx_size, chunk_word_count, output_dir, model_name
):
    samples = []
    # document is the same for the whole leaf node
    chunks = (
        docprocessor.chunk_documents(
            leaf_node=leaf_node,
            server_ctx_size=server_ctx_size,
            chunk_word_count=chunk_word_count,
            output_dir=output_dir,
            model_name=model_name,
        )
        if leaf_node[0].get("documents")
        else []
    )

    # domain is the same for the whole leaf node
    domain = leaf_node[0].get("domain")

    for chunk in chunks:
        # pylint: disable=consider-using-enumerate
        for icl_ in leaf_node:
            icl_query = {
                f"icl_query_{idx+1}": val["question"]
                for idx, val in enumerate(icl_["questions_and_answers"])
            }
            icl_resp = {
                f"icl_response_{idx+1}": val["answer"]
                for idx, val in enumerate(icl_["questions_and_answers"])
            }
            samples_row = {
                "icl_document": icl_["context"],
                "document": chunk,
                "document_outline": icl_["document_outline"],
                "domain": domain,
            }
            samples_row.update(icl_query)
            samples_row.update(icl_resp)
            samples.append(samples_row)

    return samples


def _skill_leaf_node_to_samples(leaf_node):
    samples = []

    # pylint: disable=consider-using-enumerate
    for i in range(len(leaf_node)):
        samples.append({})
        samples[-1]["task_description"] = leaf_node[i]["task_description"]
        if leaf_node[i].get("input"):
            samples[-1]["seed_context"] = leaf_node[i]["input"]
        samples[-1]["seed_question"] = leaf_node[i]["instruction"]
        samples[-1]["seed_response"] = leaf_node[i]["output"]

    return samples


def leaf_node_to_samples(
    leaf_node, server_ctx_size, chunk_word_count, output_dir, model_name
):
    if not leaf_node:
        return []
    if leaf_node[0].get("documents"):
        return _knowledge_leaf_node_to_samples(
            leaf_node, server_ctx_size, chunk_word_count, output_dir, model_name
        )
    return _skill_leaf_node_to_samples(leaf_node)
