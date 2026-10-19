#!/usr/bin/env bash
# Build the WikiArt subset used in this project (5 styles x 300 images, 224 x 224 px).
#
# Why these options:
#   --strategy spread --seed 3  visits the dataset shards in a random order. It downloads more
#                               (18 shards, about 8.4 GB streamed once) but gives 618 images with a
#                               known painter instead of 98 with the default plan. We need known
#                               painters to check "artist leakage" (same painter in train and test).
#                               The seed was chosen from the dry-run plans only (metadata, no results).
#   --max-per-artist 50         (default) one painter gives at most 50 images.
#   --size 224                  the input size of the pretrained ResNet. The MLP uses smaller copies.
#
# Images are written to data/ (ignored by Git). The exact command is also saved in
# data/wikiart_style5/subset_info.json, copied to reports/subset_info.json.
set -e
cd "$(dirname "$0")/.."

python make_wikiart_subset.py --label style \
    --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
    --per-class 300 --size 224 --strategy spread --seed 3 \
    --out data/wikiart_style5 "$@"
