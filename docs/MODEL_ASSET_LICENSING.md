# Model asset licensing gate

The project treats model weights as a separate dependency from source code.

## InsightFace / SCRFD / ArcFace

InsightFace source code is MIT-licensed, but the project's distributed pretrained model packages have separate terms. The official project currently states that its pretrained models are available for non-commercial research, and its current README directs users to contact InsightFace for licensing of open-sourced recognition models intended for other use cases.

Therefore this repository must not silently bundle or download unlicensed InsightFace recognition/detection weights for a production deployment.

## Required asset manifest

Before P0.1 real-model benchmarking can become a release gate, record for every external model:

- model family and exact filename;
- SHA-256;
- model/version identifier;
- source URL or provider;
- license identifier and permitted use;
- input/output contract;
- preprocessing contract;
- expected embedding dimension;
- benchmark dataset split used with the asset.

## Recognition gallery compatibility

Embeddings are model-space specific. Changing the recognition model/version requires a new embedding gallery; existing vectors must not be treated as mathematically convertible to another model space. Keep old/new galleries separated until the replacement model passes the same false-match and accuracy gates.

## Release rule

A model may enter production only when its license explicitly permits the intended deployment or a separate license has been obtained and recorded outside the repository. The benchmark runner should fingerprint the exact artifact so results cannot be attributed to an unverified model file.
