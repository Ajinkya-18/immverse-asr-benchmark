# Report
- The dirty subset of People's Speech was selected to simulate real-world noisy customer call conditions, characterised by varied recording environments, background noise, and diverse accents.

- The test split of the dirty configuration was selected to ensure no data leakage, as train splits may have been used in pre-training pipelines of evaluated models. The dirty subset simulates real-world noisy customer call conditions.

- Loading Wav2Vec2 for inference produces a benign warning about missing training-only parameters (masked_spec_embed), which does not affect inference performance.

- 