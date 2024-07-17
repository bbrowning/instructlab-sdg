# SPDX-License-Identifier: Apache-2.0

# Standard
from enum import Enum
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Optional
import json
import os
import time

# Third Party
# instructlab - All of these need to go away (other than sdg) - issue #6
from datasets import concatenate_datasets, Dataset
import httpx
import openai
import platformdirs

# First Party
# pylint: disable=ungrouped-imports
from instructlab.sdg.llmblock import MODEL_FAMILY_MERLINITE, MODEL_FAMILY_MIXTRAL
from instructlab.sdg.pipeline import (
    FULL_PIPELINES_PACKAGE,
    SIMPLE_PIPELINES_PACKAGE,
    Pipeline,
    PipelineContext,
)
from instructlab.sdg.sdg import SDG
from instructlab.sdg.utils import (
    GenerateException,
    models,
)
from instructlab.sdg.utils.taxonomy import (
    leaf_node_to_samples,
    read_taxonomy_leaf_nodes,
)
from instructlab.sdg.utils.parse_and_convert import (
    _unescape,
    _convert_to_messages,
    _convert_to_hack_fmt,
    create_phase07_ds,
    create_phase10_ds
)
from instructlab.sdg.utils.datamixing import (
    Recipe,
)
from instructlab.sdg.logger_config import setup_logger

# Constants
logger = setup_logger(__name__)
NUM_SYNTH_SKILLS = 30


def _check_pipeline_dir(pipeline):
    for file in ["knowledge.yaml", "freeform_skills.yaml", "grounded_skills.yaml"]:
        if not os.path.exists(os.path.join(pipeline, file)):
            raise GenerateException(
                f"Error: pipeline directory ({pipeline}) does not contain {file}."
            )


def _sdg_init(
    pipeline: Pipeline,
    client: openai.OpenAI,
    model_family: str,
    model_id: str,
    num_instructions_to_generate: int,
    batch_num_workers: Optional[int],
    batch_size: Optional[int],
):
    pipeline_pkg = None

    # Search for the pipeline in User and Site data directories
    # then for a package defined pipeline
    # and finally pipelines referenced by absolute path
    pd = platformdirs.PlatformDirs(
        appname=os.path.join("instructlab", "sdg"), multipath=True
    )
    for d in pd.iter_data_dirs():
        if os.path.exists(os.path.join(d, pipeline)):
            pipeline = os.path.join(d, pipeline)
            _check_pipeline_dir(pipeline)
            break
    else:
        if pipeline == "full":
            pipeline_pkg = FULL_PIPELINES_PACKAGE
        elif pipeline == "simple":
            pipeline_pkg = SIMPLE_PIPELINES_PACKAGE
        else:
            # Validate that pipeline is a valid directory and that it contains the required files
            if not os.path.exists(pipeline):
                raise GenerateException(
                    f"Error: pipeline directory ({pipeline}) does not exist."
                )
            _check_pipeline_dir(pipeline)

    extra_kwargs = {}
    if batch_size is not None:
        extra_kwargs["batch_size"] = batch_size
        extra_kwargs["batch_num_workers"] = batch_num_workers

    ctx = PipelineContext(
        client=client,
        model_family=model_family,
        model_id=model_id,
        num_instructions_to_generate=num_instructions_to_generate,
        **extra_kwargs,
    )

    def load_pipeline(yaml_basename):
        if pipeline_pkg:
            with resources.as_file(
                resources.files(pipeline_pkg).joinpath(yaml_basename)
            ) as yaml_path:
                return Pipeline.from_file(ctx, yaml_path)
        else:
            return Pipeline.from_file(ctx, os.path.join(pipeline, yaml_basename))

    return (
        SDG([load_pipeline("knowledge.yaml")]),
        SDG([load_pipeline("freeform_skills.yaml")]),
        SDG([load_pipeline("grounded_skills.yaml")]),
    )


def get_taxonomy_data(
    leaf_nodes,
    sys_prompt,
):
    taxonomy_data = []
    for _, leaf_node in leaf_nodes.items():
        for seed_example in leaf_node:
            user = seed_example["instruction"]  # question

            if len(seed_example["input"]) > 0:
                user += "\n" + seed_example["input"]  # context

            taxonomy_data.append(
                {
                    "system": sys_prompt,
                    "user": _unescape(user),
                    "assistant": _unescape(seed_example["output"]),  # answer
                }
            )
    taxonomy_ds = Dataset.from_list(taxonomy_data)
    return taxonomy_ds


# This is part of the public API, and used by instructlab.
# TODO - parameter removal needs to be done in sync with a CLI change.
# pylint: disable=unused-argument
def generate_data(
    logger,
    api_base,
    api_key: Optional[str] = None,
    model_family: Optional[str] = None,
    model_name: Optional[str] = None,
    num_cpus: Optional[int] = None,
    num_instructions_to_generate: Optional[int] = 30,
    taxonomy: Optional[str] = None,
    taxonomy_base: Optional[str] = None,
    output_dir: Optional[str] = None,
    # TODO - not used and should be removed from the CLI
    prompt_file_path: Optional[str] = None,
    # TODO - probably should be removed
    rouge_threshold: Optional[float] = None,
    console_output=True,
    yaml_rules: Optional[str] = None,
    chunk_word_count=None,
    server_ctx_size=None,
    tls_insecure=False,
    tls_client_cert: Optional[str] = None,
    tls_client_key: Optional[str] = None,
    tls_client_passwd: Optional[str] = None,
    pipeline: Optional[str] = "simple",
    batch_size: Optional[int] = None,
) -> None:
    """Generate data for training and testing a model.

    This currently serves as the primary interface from the `ilab` CLI to the `sdg` library.
    It is somewhat a transitionary measure, as this function existed back when all of the
    functionality was embedded in the CLI. At some stage, we expect to evolve the CLI to
    use the SDG library constructs directly, and this function will likely be removed.

    Args:
        pipeline: This argument may be either an alias defined in a user or site "data directory"
                  or an alias defined by the sdg library ("simple", "full")(if the data directory has no matches),
                  or an absolute path to a directory containing the pipeline YAML files.
                  We expect three files to be present in this directory: "knowledge.yaml",
                    "freeform_skills.yaml", and "grounded_skills.yaml".
    """
    logger = setup_logger(__name__)
    generate_start = time.time()

    # FIXME: remove this when ilab knows to pass batch_size=0 with llama.cpp
    if batch_size is None:
        batch_size = 0

    knowledge_recipe = Recipe("src/instructlab/sdg/configs/knowledge/data_recipe/default_recipe.yaml")
    skills_recipe = Recipe("src/instructlab/sdg/configs/skills/data_recipe/default_recipe.yaml")

    sys_prompt = knowledge_recipe.sys_prompt    
    logger.info(f"System prompt: {sys_prompt}")
    assert sys_prompt == skills_recipe.sys_prompt, "System prompts must be the same for both knowledge and skills"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not (taxonomy and os.path.exists(taxonomy)):
        raise GenerateException(f"Error: taxonomy ({taxonomy}) does not exist.")

    leaf_nodes = read_taxonomy_leaf_nodes(taxonomy, taxonomy_base, yaml_rules)
    logger.info(f"Found {len(leaf_nodes)} leaf nodes in the taxonomy.")
    logger.info(f"Generating data for {list(leaf_nodes.keys())} leaf nodes.")

    if not leaf_nodes:
        raise GenerateException("Error: No new leaf nodes found in the taxonomy.")

    name = Path(model_name).stem  # Just in case it is a file path
    date_suffix = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
    output_file_train = f"train_{name}_{date_suffix}.jsonl"

    # this file needs to be revisted later - retaining for now
    output_file_test = f"test_{name}_{date_suffix}.jsonl"
    
    #TODO: AB change this to handle new knowledge
    # taxonomy_ds = get_taxonomy_data(leaf_nodes, sys_prompt=sys_prompt)
    # logger.info(f"Generating to: {os.path.join(output_dir, output_file_test)}")
    # taxonomy_ds.to_json(os.path.join(output_dir, output_file_test))

    orig_cert = (tls_client_cert, tls_client_key, tls_client_passwd)
    cert = tuple(item for item in orig_cert if item)
    verify = not tls_insecure
    client = openai.OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(cert=cert, verify=verify),
    )

    if models.get_model_family(model_family, model_name) == "mixtral":
        model_family = MODEL_FAMILY_MIXTRAL
    else:
        model_family = MODEL_FAMILY_MERLINITE

    sdg_knowledge, sdg_freeform_skill, sdg_grounded_skill = _sdg_init(
        pipeline,
        client,
        model_family,
        model_name,
        num_instructions_to_generate,
        batch_size=batch_size,
        batch_num_workers=num_cpus,
    )

    if console_output:
        logger.info(
            "Synthesizing new instructions. If you aren't satisfied with the generated instructions, interrupt training (Ctrl-C) and try adjusting your YAML files. Adding more examples may help."
        )

    is_knowledge = False

    for i, leaf_node in enumerate(leaf_nodes.values()):
        samples = leaf_node_to_samples(leaf_node, server_ctx_size, chunk_word_count)
        ds = Dataset.from_list(samples)

        if not samples:
            raise GenerateException("Error: No samples found in leaf node.")

        if samples[0].get("document"):
            sdg = sdg_knowledge
            logger.info(f"Generating data for leaf node {i} with knowledge pipeline.")
            is_knowledge = True
            # add to 0.7 recipe
            # add to 1.0 recipe

        elif samples[0].get("seed_context"):
            sdg = sdg_grounded_skill
            logger.info(f"Generating data for leaf node {i} with grounded skill pipeline.")
            # add to 1.0 recipe

        else:
            sdg = sdg_freeform_skill
            logger.info(f"Generating data for leaf node {i} with freeform skill pipeline.")
            # add to 1.0 recipe
        

        generated_data = sdg.generate(ds)

        if is_knowledge:
            knowledge_phase_data = create_phase07_ds(generated_data)
            skills_phase_data = create_phase10_ds(generated_data)
            
            knowledge_fpath = os.path.join(output_dir, f"node_datasets_{date_suffix}/node_{i}_p07.jsonl")
            skills_fpath = os.path.join(output_dir, f"node_datasets_{date_suffix}/node_{i}_p10.jsonl")
            knowledge_phase_data.to_json(knowledge_fpath, orient="records", lines=True)
            skills_phase_data.to_json(skills_fpath, orient="records", lines=True)
            
            knowledge_recipe.add_dataset(knowledge_fpath)
            skills_recipe.add_dataset(skills_fpath)
        else:
            messages = generated_data.map(
                _convert_to_messages,
                fn_kwargs={"sys_prompt": sys_prompt},
                num_proc=8,
            )

            fpath = os.path.join(output_dir, f"node_datasets_{date_suffix}/node_{i}.jsonl")
            messages.to_json(fpath, orient="records", lines=True)
            skills_recipe.add_dataset(fpath, NUM_SYNTH_SKILLS)
    

    if knowledge_recipe.dataset_added:
        knowledge_recipe.save_recipe(f"{output_dir}/knowledge_recipe_{date_suffix}.yaml")
        knowledge_recipe.save_mixed_dataset(f"{output_dir}/knowledge_train_msgs_{date_suffix}.jsonl")
                                                                                 
    if skills_recipe.dataset_added:
        skills_recipe.save_recipe(f"{output_dir}/skills_recipe_{date_suffix}.yaml")
        skills_recipe.save_mixed_dataset(f"{output_dir}/skills_train_msgs_{date_suffix}.jsonl")

    logger.info(f"Generation complete in {time.time() - generate_start:.2f}s")
