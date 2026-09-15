# Repainting sprites with an image model

How the Saga games turn their code-rendered stand-ins into painted sprites, and how to
do it again for a new unit, a new race or a new game. The library is
`sagaforge/restyle.py`; each game drives it from `tools/restyle.py`.

```bash
cd ~/saga/warband
uv run python tools/restyle.py refresh /tmp/restyle              # dump, render (Codex), cut, preview: every human unit
uv run python tools/restyle.py --units knight refresh /tmp/restyle
uv run python tools/restyle.py showcase /tmp/skirmish.gif        # a clip through the real renderer
```

Tribes (`tools/restyle.py refresh DIR`, seven tokens on one sheet) and Shardbound
(`tools/restyle.py refresh DIR`, 18 miniatures on one sheet) work the same way.

## The contract: one subject, one sheet

A *subject* is everything that shares a look: a Warband unit of one race (with a
carrying variant as its own subject), a Tribes token, a Shardbound piece kind. The
game renders every frame of the subject the way it renders them for play, and
`Sheet.layout` packs them into a grid:

- equal cells, the subject's frames laid out facings across and frames down;
- every frame's anchor (the feet) on the same pixel of its cell (`Sheet.origin`), so
  the game can place any frame with one `Placement`;
- a flat magenta `#FF00FF` ground, thin dark cell borders, and an empty half-cell
  margin around the grid;
- the layout saved as JSON next to the PNG (`Sheet.save`).

Why each part is there, learned the hard way:

- **Borders.** Without them Gemini doubled the rows and dropped the facings; with them
  both models keep the grid.
- **Margin.** A model may return a different aspect ratio or shift the drawing; the margin
  keeps every cell inside the picture, and the borders let `locate_grid` find the frame
  and `align` map it back onto the sheet's pixels.
- **Chroma, not transparency.** Asked for a real alpha channel, Codex's tool painted a
  checkerboard into an opaque image. `key_out` keys by hue, so a ground shadow the model
  painted in a darker magenta becomes translucent black instead of a purple blob.
- **The model copies the layout it is given, exactly.** A packing bug once put every
  figure on its cell corner; Codex kept them there. If a stand-in is wrong (a stone on
  top of the arm, logs floating over a head), the painting is wrong the same way. Fix the
  rig when it is cheap, and always tell the prompt (see below).
- **One team colour.** Sheets are painted for player 0; `recolor` moves pixels near that
  hue to another team's colour, leaving steel and skin alone. Say in the prompt which
  colour is the team colour and that it must stay.

## The prompt

`prompt()` in each game's tool assembles it from the sheet geometry and tables; keep it
factual and in this order:

1. What the image is: pixel size, rows × columns of W×H cells, margin, magenta ground,
   borders that must stay where they are, each figure centred in its cell.
2. What each row and column means (frame names in words, facings in words).
3. The subject in one sentence, naming the team colour.
4. The style paragraph for the game (Warband: hand-painted Warcraft 2 / Heroes feel;
   Tribes: chunky Polytopia figurines; Shardbound: gouache miniatures in ink-teal-brass).
5. The plausibility licence and the subject's known shortcomings (`FIXES` in the tool):
   what the stand-in gets wrong and how the plausible version should look, in the same
   place and pose.
6. What to keep exactly (position, scale, pose, facing, feet) and what the background
   must be (flat magenta, nothing else), and the output size.

Rows that differ on purpose (a walk cycle, the phases of a blow) need the sentence
"the poses differ from row to row on purpose"; without it a model tends to average them.

## Providers

| Provider | How | Cost | Fidelity | Notes |
|---|---|---|---|---|
| Codex built-in image tool (`render_with_codex`) | `codex exec` with the sheet attached, prompt on stdin, the agent copies the PNG out of `$CODEX_HOME/generated_images` | $0 on the ChatGPT plan | keeps grid, camera, scale, poses, team colour | about 2 min a sheet, sheets render concurrently (`--jobs`); the plan has a usage limit (about 25 edits, then "try again at HH:MM"); the prompt must go on stdin because `-i` is variadic |
| OpenRouter Gemini 3.1 Flash Image (`render_with_openrouter`) | chat completion with the sheet as an image input, `image_config` aspect ratio and size | about $0.07 at 1K, $0.15 at 2K | keeps the grid with borders; at 1K it redrew the camera and scale, at 2K it kept 72 distinct poses | use `image_size="2K"` for sheets over 32 cells; aspect ratios are limited to the model's list (1:1, 3:2, 2:3, 4:3, 3:4, 4:5, 5:4, 16:9, 9:16, 21:9) |
| Other OpenRouter image models | same call, `--model` | $0.15–0.30 | untested | |

Keys: `openrouter_api_key()` reads `OPENROUTER_API_KEY` from the environment, else the
stack's secrets index. The stored OpenAI and Gemini API keys are dead.

Output size: the Codex tool returns about 1.5 megapixels whatever the prompt asks for
(a 2142×2569 sheet came back 1144×1375 even when the request named the size), so a
72-cell sheet gets roughly 130 px per cell, a 32-cell sheet about 190 px. Enough for
sprites shown at 45–95 logical px; for more detail split a subject into a walk sheet and a
blow sheet (at some risk of style drift between them) or use Gemini at `image_size="2K"`.

## Cutting, checking, installing

`cut(sheet, rendered, original)`:

1. `align`: find the border frame in the output (`locate_grid`: thin, evenly spaced dark
   lines with the expected counts) and resample the output onto the sheet's pixel grid.
   Without a frame the whole image is assumed to be the whole sheet.
2. `key_out`: hue key with shadow recovery. A margin that is not the key colour rejects
   the output (that is the painted-checkerboard case).
3. Register: one scale (median height ratio) and one shift (median centre and feet
   offsets) for the whole sheet, never per cell, so frames do not jitter against each other.
4. Report per cell: coverage within 0.5–2.2× of the original, centroid drift under 12% of
   the cell, nothing touching a border, a figure present. `Cut.flagged` lists failures;
   the tools reject a sheet over their tolerance and print why. Re-render rather than
   patch a bad sheet; a second run usually differs.
5. `save_frames` writes one RGBA sheet without margins plus the layout JSON into the
   game's asset folder; `load_frames` reads it back into cell-sized frames.

At runtime each game checks for a painted subject before rendering the stand-in:
Warband `textures.unit_image` (recolouring per player; a sheet whose frames no longer
match `textures.FRAMES` warns and is ignored), Tribes `textures.register_all` (a
recolouring per tribe), Shardbound `art.piece` (one PNG per team and kind).
`WARBAND_ART=procedural`, `TRIBES_ART=procedural`, `SHARDBOUND_ART=procedural` play with
the stand-ins for comparison. Painted PNGs live next to their JSON; the game repos ignore
`*.png` by default, so the asset folder is exempted in `.gitignore`.

## Adding things

- **A new frame or pose** (Warband): add it to `textures.FRAMES`/`WALK_FRAMES`/
  `ATTACK_FRAMES` and the pose tables; describe it in `FRAME_NAMES` in the tool; every
  painted sheet is then stale and `refresh` re-renders them all.
- **A new race**: write its `SUBJECTS` descriptions and `FIXES` in the tool, run
  `--race orc refresh DIR`.
- **A new subject in another game**: render the frames the game's own way into cell-sized
  RGBA images with a shared anchor, `Sheet.layout` them, write a `prompt()`, and reuse
  `render_with_codex` / `cut` / `save_frames`. Shardbound shows how vector art is
  rasterised through a recorder that implements the few `draw_*` calls it uses.
- **A style anchor**: both providers accept several input images; passing an approved
  sheet alongside a new one should tighten drift between subjects (not done yet).

## Looking at the result

`preview` writes GIF strips (original above, painted below, every facing, the walk then the
blow) and `showcase` records a clip through the real renderer. Mock tests prove the
frames load and place; only the strips and the clip show whether a game developer would
ship them. The display must be awake for the clip (`caffeinate -u`).
