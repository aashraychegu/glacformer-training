from transformers import (
    SegformerImageProcessor,
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
import pathlib as pl
test_image_processor = SegformerImageProcessor.from_pretrained("nvidia/MiT-b0")
test_image_processor.save_pretrained(pl.Path(__file__).parent / "sip")

