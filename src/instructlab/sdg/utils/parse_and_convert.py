# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
import json
import random
import uuid

# Third Party
from datasets import Dataset

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg import utils
from instructlab.sdg.logger_config import setup_logger

logger = setup_logger(__name__)


class TaxonomyType(Enum):
    KNOWLEDGE = "knowledge"
    SKILL = "skill"


def _unescape(s):
    return bytes(s, "utf-8").decode("utf-8").strip()


# This is a hack because the simple workflow returns a q/a pair as a single output.
# We could possibly try to ask for them separately, but it would cost twice the inference
# API calls. All of this is because the smallest models we use on small environments
# for testing and demos weren't good enough to follow the strict formatting instructions used
# in the full pipeline.
def _get_question(synth_example: dict):
    if "question" in synth_example:
        return synth_example["question"]

    if not synth_example.get("output"):
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[0].strip() + "?" if len(parts) == 2 else ""


# This is also a hack. See the comment above _get_question.
def _get_response(synth_example: dict):
    if "response" in synth_example:
        return synth_example["response"]

    if "output" not in synth_example:
        raise utils.GenerateException(
            f"Error: output not found in synth_example: {synth_example}"
        )

    parts = synth_example["output"].split("?", 1)
    if len(parts) != 2:
        logger.warning(f"Failed to split generated q&a: {synth_example['output']}")
    return parts[1].strip() if len(parts) == 2 else parts[0].strip()


def _convert_messages_to_legacy(sample: dict):
    """
    Convert new format messages to the legacy format 'system', 'user', and
    'assistant' columns.

    Note: We should remove this function in the future when standardize the
    format to messages.
    """
    skipSample = False
    if len(sample["messages"]) >= 2 and sample["messages"][1]["role"] == "pretraining":
        # TODO: Handle converting pretraining messages to legacy format
        skipSample = True
    elif len(sample["messages"]) < 3:
        logger.warning(
            f"Cannot convert sample to legacy format as it's missing one or more of system, user, or assistant messages: {sample}"
        )
        skipSample = False

    if skipSample:
        sample["system"] = ""
        sample["user"] = ""
        sample["assistant"] = ""
    else:
        sample["system"] = _unescape(sample["messages"][0]["content"])
        sample["user"] = _unescape(sample["messages"][1]["content"])
        sample["assistant"] = _unescape(sample["messages"][2]["content"])

    return sample


def _convert_to_messages(sample: dict, sys_prompt: str):
    """
    Convert a sample dictionary to contain 'messages'
    and 'metadata' columns required for training.
    """
    # Create user query message
    user_query = _unescape(_get_question(sample))
    response = _unescape(_get_response(sample))

    sample["id"] = str(uuid.uuid4())
    sample["messages"] = [
        {"content": sys_prompt, "role": "system"},
        {"content": user_query, "role": "user"},
        {"content": response, "role": "assistant"},
    ]

    return sample


def generate_knowledge_qa_dataset(
    generated_dataset: Dataset, keep_context_separate=False
):
    def __create_qa_row(rec):
        context = rec["document"]
        instruction = _get_question(rec)
        response = _get_response(rec)
        metadata = {
            "sdg_document": rec["document"],
            "domain": rec["domain"],
            "dataset": "document_knowledge_qa",
        }
        if "raw_document" in rec and "dataset_type" in rec:
            metadata.update(
                {
                    "raw_document": rec["raw_document"],
                    "dataset_type": rec["dataset_type"],
                }
            )
        metadata = json.dumps(metadata)
        if keep_context_separate:
            messages = [
                {"role": "user", "content": f"{instruction}"},
                {"role": "assistant", "content": response},
            ]
            return {
                "messages": messages,
                "metadata": metadata,
                "id": str(uuid.uuid4()),
                "context": context,
            }
        messages = [
            {"role": "user", "content": f"{context}\n\n{instruction}"},
            {"role": "assistant", "content": response},
        ]

        return {"messages": messages, "metadata": metadata, "id": str(uuid.uuid4())}

    knowledge_ds = generated_dataset.map(
        __create_qa_row, remove_columns=generated_dataset.column_names
    )
    return knowledge_ds


def build_raft_dataset(ds: Dataset, p, num_doc_in_context=4):
    all_context = ds["context"]
    all_context = [
        " ".join(e.split(" ")[: random.randint(100, 500)]) for e in all_context
    ]
    ds = ds.add_column("row_idx", range(ds.num_rows))

    def __pick_documents(rec, p):
        # Loop until we find enough other documents to add to the context
        # for this document. Exit the loop early if we have fewer total
        # documents than the number of documents we want in our context
        # so that we don't end up looping forever. This handles edge
        # cases where the number of generated instructions is very low,
        # like in CI or user's testing small sizes.
        while True:
            selected_docs = random.choices(range(ds.num_rows), k=num_doc_in_context)
            if ds.num_rows <= num_doc_in_context:
                break
            if rec["row_idx"] not in selected_docs:
                break
        if random.uniform(0, 1) < p:
            docs = [
                all_context[idx] for idx in selected_docs[: num_doc_in_context - 1]
            ] + [rec["context"]]
            # rec['indicator'] ='golden'
        else:
            docs = [all_context[idx] for idx in selected_docs]
            # rec['indicator'] = 'distractor'
        random.shuffle(docs)
        docs = "\n".join(([f"Document:\n{e}\n\n" for idx, e in enumerate(docs)]))
        user_idx, user_msg = [
            (idx, rec_msg)
            for idx, rec_msg in enumerate(rec["messages"])
            if rec_msg["role"] == "user"
        ][0]
        user_inst = user_msg["content"]
        rec["messages"][user_idx]["content"] = f"{docs}\n\n{user_inst}"
        rec["messages"] = rec["messages"]
        metadata = json.loads(rec["metadata"])
        metadata["dataset"] += f"_raft_p{p}"
        rec["metadata"] = json.dumps(metadata)
        return rec

    ds = ds.map(__pick_documents, fn_kwargs={"p": p}, remove_columns=["context"])
    return ds


def _conv_pretrain(rec):
    rec["messages"] = [
        {
            "role": "pretraining",
            "content": f"<|user|>\n{rec['messages'][0]['content']}\n<|assistant|>\n{rec['messages'][1]['content']}",
        }
    ]
    return rec


def create_phase10_ds(generated_dataset: Dataset):
    # Phase 1.0
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=True
    )
    knowledge_ds = build_raft_dataset(knowledge_ds, p=0.4)

    return knowledge_ds


def create_phase07_ds(generated_dataset: Dataset):
    # Phase 0.7
    knowledge_ds = generate_knowledge_qa_dataset(
        generated_dataset, keep_context_separate=False
    )
    knowledge_ds = knowledge_ds.map(_conv_pretrain)

    return knowledge_ds
