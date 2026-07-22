"""Lazy Gradio UI for the unified experiment runner and saved-run browser."""

from __future__ import annotations

import json
from pathlib import Path

from dashboard.service import DashboardRunRequest, DashboardService
from experiments.config import ExperimentMode


def build_app(service: DashboardService):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional UI dependency
        raise RuntimeError("The dashboard requires the optional 'gradio' package") from exc

    datasets = list(service.dataset_names())
    if not datasets:
        raise RuntimeError("No dataset adapters are registered")
    initial_dataset = datasets[0]
    initial_splits = list(service.splits(initial_dataset))
    initial_split = initial_splits[0]
    initial_runs = [str(path) for path in service.available_runs()]

    def update_dataset(dataset):
        splits = list(service.splits(dataset))
        split = splits[0]
        count = service.sample_count(dataset, split)
        image, metadata = service.load_sample(dataset, split, 0)
        return gr.update(choices=splits, value=split), gr.update(maximum=max(0, count-1), value=0), image, metadata

    def update_split(dataset, split):
        count = service.sample_count(dataset, split)
        image, metadata = service.load_sample(dataset, split, 0)
        return gr.update(maximum=max(0, count-1), value=0), image, metadata

    def load_sample(dataset, split, index):
        image, metadata = service.load_sample(dataset, split, int(index))
        return image, metadata

    def run_pipeline(dataset, split, index, mode, target, run_name, reporting, max_passes, max_sam3, max_qwen, stop_conf, tiling, expert):
        try:
            overrides = json.loads(expert or "{}")
            request = DashboardRunRequest(
                dataset_name=dataset,
                split=split,
                sample_index=int(index),
                mode=ExperimentMode(mode),
                target_concept=target or None,
                run_name=run_name or None,
                reporting_level=reporting,
                max_passes=int(max_passes) if max_passes is not None else None,
                max_sam3_calls=int(max_sam3) if max_sam3 is not None else None,
                max_qwen_calls=int(max_qwen) if max_qwen is not None else None,
                stopping_confidence=float(stop_conf) if stop_conf is not None else None,
                tiling_mode=tiling,
                expert_overrides=overrides,
            )
            view = service.run(request)
            choices = [str(path) for path in service.available_runs()]
            return (
                view.image,
                view.summary,
                list(view.pass_rows),
                list(view.node_rows),
                list(view.qwen_rows),
                list(view.information_rows),
                list(view.tiling_rows),
                str(view.run_directory),
                gr.update(choices=choices, value=str(view.run_directory)),
            )
        except Exception as exc:
            return None, {"error": f"{type(exc).__name__}: {exc}"}, [], [], [], [], [], "", gr.update()

    def replay(path):
        view = service.replay(path)
        return view.image, view.summary, list(view.pass_rows), list(view.node_rows), service.raw_json(path), gr.update(maximum=max(0, len(view.run.passes)-1), value=0)

    def show_pass(path, index):
        return service.pass_view(path, int(index))

    def compare(paths):
        return list(service.compare(paths or []))

    with gr.Blocks(title="SAM3-VLM Experiment Dashboard") as app:
        gr.Markdown("# SAM3-VLM Experiment Dashboard")
        gr.Markdown("Run unified pipelines, inspect every pass, replay complete provenance, and compare saved experiments.")
        with gr.Tab("Run experiment"):
            with gr.Row():
                with gr.Column(scale=2):
                    image = gr.Image(type="pil", label="Input / final overlay")
                    sample_metadata = gr.JSON(label="Sample metadata")
                with gr.Column(scale=1):
                    dataset = gr.Dropdown(datasets, value=initial_dataset, label="Dataset")
                    split = gr.Dropdown(initial_splits, value=initial_split, label="Split")
                    sample_index = gr.Slider(0, max(0, service.sample_count(initial_dataset, initial_split)-1), value=0, step=1, label="Sample")
                    mode = gr.Dropdown([item.value for item in ExperimentMode], value=ExperimentMode.SAM3_SINGLE_PASS.value, label="Pipeline")
                    target = gr.Textbox(label="Target concept override")
                    run_name = gr.Textbox(label="Run name")
                    reporting = gr.Radio(["standard", "full"], value="full", label="Reporting")
                    with gr.Row():
                        max_passes = gr.Number(value=8, precision=0, label="Max passes")
                        max_sam3 = gr.Number(value=32, precision=0, label="Max SAM3 calls")
                        max_qwen = gr.Number(value=16, precision=0, label="Max Qwen calls")
                    stop_conf = gr.Slider(0.50, 0.999, value=0.95, step=0.005, label="Stopping confidence")
                    tiling = gr.Radio(["off", "always", "density_adaptive", "agent_controlled"], value="off", label="Tiling")
                    with gr.Accordion("Advanced internals", open=False):
                        expert = gr.Code(value="{}", language="json", label="Expert overrides")
                    run_button = gr.Button("Run complete pipeline", variant="primary")
                    run_directory = gr.Textbox(label="Saved run directory", interactive=False)
            with gr.Tabs():
                with gr.Tab("Summary"):
                    run_summary = gr.JSON()
                with gr.Tab("Pass timeline"):
                    pass_table = gr.Dataframe(interactive=False)
                with gr.Tab("Objects"):
                    node_table = gr.Dataframe(interactive=False)
                with gr.Tab("Qwen trace"):
                    qwen_table = gr.Dataframe(interactive=False)
                with gr.Tab("Information"):
                    information_table = gr.Dataframe(interactive=False)
                with gr.Tab("Tiling"):
                    tiling_table = gr.Dataframe(interactive=False)

        with gr.Tab("Replay saved run"):
            replay_run = gr.Dropdown(initial_runs, label="Run directory")
            replay_button = gr.Button("Replay")
            with gr.Row():
                replay_image = gr.Image(type="pil", label="Saved overlay / selected pass")
                replay_summary = gr.JSON(label="Run summary")
            pass_slider = gr.Slider(0, 0, value=0, step=1, label="Pass")
            pass_button = gr.Button("Show selected pass")
            pass_detail = gr.JSON(label="Pass detail")
            replay_passes = gr.Dataframe(interactive=False)
            replay_nodes = gr.Dataframe(interactive=False)
            raw_json = gr.Code(language="json", label="run.json")

        with gr.Tab("Compare runs"):
            compare_runs = gr.Dropdown(initial_runs, multiselect=True, label="Runs")
            compare_button = gr.Button("Compare")
            compare_table = gr.Dataframe(interactive=False)

        dataset.change(update_dataset, inputs=[dataset], outputs=[split, sample_index, image, sample_metadata])
        split.change(update_split, inputs=[dataset, split], outputs=[sample_index, image, sample_metadata])
        sample_index.change(load_sample, inputs=[dataset, split, sample_index], outputs=[image, sample_metadata])
        app.load(load_sample, inputs=[dataset, split, sample_index], outputs=[image, sample_metadata])
        run_button.click(
            run_pipeline,
            inputs=[dataset, split, sample_index, mode, target, run_name, reporting, max_passes, max_sam3, max_qwen, stop_conf, tiling, expert],
            outputs=[image, run_summary, pass_table, node_table, qwen_table, information_table, tiling_table, run_directory, replay_run],
        )
        replay_button.click(replay, inputs=[replay_run], outputs=[replay_image, replay_summary, replay_passes, replay_nodes, raw_json, pass_slider])
        pass_button.click(show_pass, inputs=[replay_run, pass_slider], outputs=[replay_image, pass_detail])
        compare_button.click(compare, inputs=[compare_runs], outputs=[compare_table])
    return app


__all__ = ["build_app"]
