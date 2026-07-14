# Train a tokenizer for the pixel ColorField environment

Train and freeze a temporal tokenizer for the **pixel/RGB ColorField** environment under
`autoresearch/frozen/`. This is the image-based tier: the model must consume rendered `64x64`
RGB observations from the procedural ColorField dataset. Do **not** use `colorfield-sym`,
`train_sym.py`, or direct one-hot viewport rows; those belong to the tokenizer-free symbolic
search tier and do not validate the pixel pipeline.

## Why

The autoresearch pixel tier needs a frozen lossy visual latent space before dynamics and memory
experiments can be meaningful. The tokenizer must preserve the per-cell palette information in
the egocentric rendered frames, including the out-of-map border colour that anchors position.

## Scope

- Use the procedural datasets `data/colorfield` and `data/colorfield_val` (render frames on the
  fly through the frozen ColorField datagen), not symbolic observations.
- Use the vendored pixel tokenizer/trainer in `autoresearch/frozen/`. Start from the established
  ColorField configuration: embedding `256`, depth `9`, heads `16`, `n_latents=4`,
  `bottleneck_dim=64`, temporal window `16`.
- Keep the tokenizer training-only: no dynamics training or latent-cache build in this task.
- Save the accepted checkpoint at `checkpoints/colorfield/tokenizer.pt`, with run logs and
  reconstruction artefacts kept under a dedicated `experiments/` directory (large artefacts stay
  gitignored).

## Acceptance

- Train on a GPU and retain the best validation reconstruction checkpoint using the trainer's
  stability safeguards.
- On held-out `data/colorfield_val`, decode tokenizer reconstructions and run the frozen
  ColorField cell readout. Require readout-exact reconstruction at the established bar:
  cell accuracy at least `0.9999` and frame-exact accuracy at least `0.998`.
- Inspect a reconstruction sheet covering interior and border-heavy views; it must preserve
  palette IDs and the out-of-map colour without visible spatial shifts.
- Record checkpoint SHA-256, config, dataset hashes, training metrics, and readout metrics in
  the experiment notes. If the frozen-layer manifest is active, register the accepted checkpoint
  according to its integrity workflow.

## Done when

A readout-exact, frozen pixel ColorField tokenizer checkpoint and its reproducible result record
exist. The result must clearly state that it was trained on rendered RGB frames, not direct
one-hot symbolic inputs.
