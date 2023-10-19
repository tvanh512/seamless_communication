# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import torch
from seamless_communication.models.unit_extraction import UnitExtractor
import os
import glob
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw audio to units (and optionally audio) using UnitExtractor."
    )
    parser.add_argument("--indir", type=str, help="Audio WAV input")
    parser.add_argument("--outdir", type=str, help="discrete token output")
    parser.add_argument("--audiotype", type=str, help="file type of audio, e.g wav, flac..")
    parser.add_argument(
        "--kmeans_uri",
        type=str,
        help="URL path to the K-Means model.",
        default="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Feature extraction model name (`xlsr2_1b_v2`)",
        default="xlsr2_1b_v2",
    )
    parser.add_argument(
        "--out_layer_number",
        type=int,
        help="Layer number of the feature extraction model to pull out features from.",
        default=35,
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Running unit_extraction on the GPU.")
    else:
        device = torch.device("cpu")
        logger.info("Running unit_extraction on the CPU.")

    unit_extractor = UnitExtractor(args.model_name, args.kmeans_uri, device=device)
    indir = args.indir + '/*.' + args.audiotype
    audio_files = glob.glob(indir)
    os.makedirs(args.outdir, exist_ok=True)
    for i in range(len(audio_files)):
        units = unit_extractor.predict(audio_files[i], args.out_layer_number - 1)
        audio_file_name = os.path.basename(audio_files[i])
        discrete_file_name = audio_file_name.replace(args.audiotype, 'txt')
        units = units.detach().cpu().numpy()
        units = units.reshape(1, -1)
        # Save the array to a text file with spaces between numbers
        np.savetxt(os.path.join(args.outdir,discrete_file_name), units, fmt='%d', delimiter=' ')

if __name__ == "__main__":
    main()
