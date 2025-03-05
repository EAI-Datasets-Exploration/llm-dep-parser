"""
Module Entrypoint

This file is used to allow the project to be invoked via:
    `python -m traditional_feature_extraction`
"""
import configparser
import json
import os
import pandas as pd
import multiprocessing as mp
import time

from llm_dep_parser.gpu_utils import process_on_gpu
from llm_dep_parser.text_utils import clean_determiners

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensure CUDA-safe multiprocessing

    config = configparser.ConfigParser()
    config.read("config.ini")

    # Load Configuration Variables
    dataset_dir_path = config["paths"]["dataset_dir_path"]
    results_dir = config["paths"]["results_dir"]

    dataset_name = config["experiment"]["dataset_name"]

    if config.getboolean("icl", "is_icl"):
        is_icl = True
        labels_df_fp = config["icl"]["labels_df_fp"]
        matrix_fp = config["icl"]["matrix_fp"]
        vectorizer_fp = config["icl"]["vectorizer_fp"]
    else:
        is_icl = False
        labels_df_fp = ""
        matrix_fp = ""
        vectorizer_fp = ""

    # Save results
    out_path = results_dir + "/" + dataset_name
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    with open(f"{dataset_dir_path}metadata.json", "r", encoding="utf-8") as json_file:
        # The metadata file referenced here is contained in the repo:
        # dataset-download-scripts package hosted in the larger GitHub group.
        metadata_dict = json.load(json_file)

    ds_path = metadata_dict[dataset_name]

    # Begin text feature processing
    df = pd.read_csv(dataset_dir_path + ds_path)

    unique_texts = df[config["data"]["nl_column"]].dropna().unique().tolist()

    gpu_ids = list(map(int, config["experiment"]["gpu_ids"].split(",")))
    num_gpus = len(gpu_ids)
    split_texts = [unique_texts[i::num_gpus] for i in range(num_gpus)]

    with mp.Pool(processes=num_gpus) as pool:
        gpu_results = pool.starmap(
            process_on_gpu,
            [
                (
                    config["experiment"]["model_name"],
                    gpu_ids[i],
                    split_texts[i],
                    int(config["experiment"]["batch_size"]),
                    is_icl,
                    labels_df_fp,
                    matrix_fp,
                    vectorizer_fp,
                )
                for i in range(num_gpus)
            ],
        )

    pool.close()
    pool.join()

    step_1_results = [item for sublist in gpu_results for item in sublist]

    step_1_df = pd.DataFrame(
        step_1_results,
        columns=["full_prompt", "llm_response"],
    )
    step_1_df.to_csv(f"{out_path}/{dataset_name}_dep_parse_results.csv", index=False)
