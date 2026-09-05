"""Compose a WAV and play it through Saga2D; pass --audible to hear it."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audible", action="store_true", help="use the default audio driver at 25%% master volume")
    args = parser.parse_args()
    os.environ["SAGA2D_SILENT"] = "0" if args.audible else "1"
    from saga2d import Game, Scene
    from saga2d.synth import BELL, level, mix, noise, pan, thump, tone, write_wav

    with TemporaryDirectory(prefix="saga2d-synth-") as directory:
        root = Path(directory)
        cue = level(mix(thump(170, 55, .2), noise(.09, 500, 4000, seed=5) * .3,
                        (.08, pan(tone("D4", .45, partials=BELL, tau=.18), -.3)),
                        (.22, pan(tone("A4", .45, partials=BELL, tau=.18), .3))), .65)
        write_wav(root / "sounds" / "arrival.wav", cue)
        game = Game("Synthesis example", visible=False, resolution=(64, 64), asset_path=root)

        class Playback(Scene):
            def on_enter(self):
                self.started = time.monotonic()
                self.game.audio.set_volume("master", .25)
                self.game.audio.play_sound("arrival")

            def update(self, dt):
                if time.monotonic() - self.started > .85:
                    self.game.quit()

        game.run(Playback())
    print("Composed stereo WAV decoded and played through Game.audio; native run cleaned up.")


if __name__ == "__main__":
    main()
