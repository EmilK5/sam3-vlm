# Adaptive tiling subsystem

The tiling subsystem is deliberately independent of the controller. It receives
one cleaned stage-1 detection batch and one normalized SAM3 query.

## Density score

The default score combines:

- summed box coverage of the source image;
- object count normalized at 50 detections;
- inverse average object-size ratio.

The default weights are 0.30, 0.50, and 0.20. The trigger rules reproduce the
released SAM3Count image inference defaults:

- more than 90 objects: `LARGE`;
- coverage above 0.40 with average size below 0.02: `MEDIUM`;
- density score above 0.70 with average size below 0.01: `SMALL`.

## Tile rules

- `LARGE`: target 2 x 1 grid, 35% overlap;
- `MEDIUM`: target 4 x 2 grid, 30% overlap;
- `SMALL`: target 6 x 4 grid, 25% overlap.

The final square tile size is derived from the target grid and clipped to the
configured minimum and maximum. Tile calls are also truncated by the remaining
ASHT query budget and the configured maximum tiles per action.

## Provenance

A `TilingDecisionRecord` stores the complete decision. Every tile has a
`TileRecord`, and every SAM3 detection retains its `tile_id`. Cross-tile NMS
emits one `DedupComparisonRecord` per comparison.
