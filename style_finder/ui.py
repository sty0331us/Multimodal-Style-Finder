"""Gradio interface for the Style Finder pipeline."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from style_finder.pipeline import StyleFinderApp


def create_gradio_interface(app: StyleFinderApp) -> gr.Blocks:
    """Build the public-facing fashion analyzer UI."""
    example_paths = [Path(path) for path in app.settings.example_images()[:3]]
    while len(example_paths) < 3:
        example_paths.append(None)

    with gr.Blocks(theme=gr.themes.Soft(), title="Fashion Style Analyzer") as demo:
        gr.Markdown(
            """
            # Fashion Style Analyzer

            Upload an image to analyze fashion elements and get detailed information about the items.
            This application combines computer vision, vector similarity, and large language models
            to provide detailed fashion analysis.
            """
        )

        gr.Markdown("### Example Images")
        with gr.Row():
            for index, path in enumerate(example_paths, start=1):
                gr.Image(
                    value=str(path) if path else None,
                    label=f"Example {index}",
                    show_label=True,
                    scale=1,
                )

        with gr.Row():
            example_buttons = [
                gr.Button(f"Use Example {index}") for index in range(1, 4)
            ]

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Upload Fashion Image")
                submit_btn = gr.Button("Analyze Style", variant="primary")
                status = gr.Markdown("Ready to analyze.")
            with gr.Column(scale=2):
                output = gr.Markdown(label="Style Analysis Results", height=700)

        submit_btn.click(
            fn=lambda: "Analyzing image... This may take a few moments.",
            inputs=None,
            outputs=status,
        ).then(
            fn=app.process_image,
            inputs=[image_input],
            outputs=output,
        ).then(
            fn=lambda: "Analysis complete!",
            inputs=None,
            outputs=status,
        )

        for index, button in enumerate(example_buttons):
            example_path = example_paths[index]
            loaded_message = (
                f"Example {index + 1} loaded. Click 'Analyze Style' to process."
                if example_path
                else "Example image is not available."
            )
            button.click(
                fn=_load_example(example_path),
                inputs=None,
                outputs=image_input,
            ).then(
                fn=lambda message=loaded_message: message,
                inputs=None,
                outputs=status,
            )

        gr.Markdown(
            """
            ### About This Application

            This system analyzes fashion images using:

            - **Image Encoding**: Converting fashion images into numerical vectors
            - **Similarity Matching**: Finding visually similar items in a ChromaDB vector catalog
            - **Advanced AI**: Generating detailed descriptions of fashion elements

            The analyzer identifies garments, fabrics, colors, and styling details from images.
            The database includes information on outfits with brand and pricing details.
            """
        )

    return demo


def _load_example(path: Path | None):
    def _loader():
        return str(path) if path else None

    return _loader
