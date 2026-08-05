#!/usr/bin/env python3
"""Example scenes for h3-consumer-bench, in the H3 prompt grammar.

The official guide wants an alignment instruction when a reference image exists,
then three fields on a timeline. integrated_multimodal_description carries the
shots. overall_soundscape and non_diegetic_music carry the audio. Subject,
framing, background, camera motion and texture keep a T2V model from collapsing
into mush. Prompts without a subject melt.
"""

INSTRUCAO_I2VA = ("For the target video, at 0.00 seconds into the target video, <Picture 1> "
                  "(from [Shot 1]) is fully referenced.")

CENAS_H3 = {
  "exemplo-t2va": {
    "image": None,
    "prompt": """integrated_multimodal_description: [Shot 1] Cinematic live-action, low resolution digital camera texture with fine grain. A person dancing alone in an empty warehouse, dramatic single-source lighting from a high window, dust drifting through the beam. The camera stays locked and wide. The dancer moves slowly at first, then faster, shadow stretching across the concrete floor. Near the end the dancer stops and the dust keeps moving.

overall_soundscape: Footsteps and fabric movement echoing in a large concrete space, breath, a distant ventilation hum.

non_diegetic_music: None.""",
  },
  "exemplo-i2va": {
    "image": "PONHA_AQUI/primeiro_frame.png",
    "prompt": f"""{INSTRUCAO_I2VA}

integrated_multimodal_description: [Shot 1] Cinematic live-action, low resolution digital camera texture. The frame opens exactly on the reference image. Describe the style, subjects, composition and scene anchors of your image here, then the next action, keeping identity, clothing, colors and spatial relations consistent. The camera pushes in very slowly.

overall_soundscape: Describe the ambient and physical sounds of your scene.

non_diegetic_music: None.""",
  },
}

if __name__ == "__main__":
  for k, v in CENAS_H3.items():
    print(f"== {k} (image={'yes' if v['image'] else 'no'})")
