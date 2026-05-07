from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import traceback
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tkinter import StringVar, Tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import tkinter as tk
import tkinter.font as tkfont

from PIL import Image, ImageTk

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
TREE_DIR = ROOT_DIR / "tree_generation"
IMAGE_DIR = ROOT_DIR / "image_generation"
for import_dir in (SCRIPT_DIR, TREE_DIR, IMAGE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from NSGA2 import Config, NSGA2
from layout_preview import render_tree_preview_from_dict

APP_NAME = "EndfieldBeltSplitter"
APP_TITLE = "明日方舟：终末地 - 传送带分流计算器"
APP_VERSION = "1.2.0"

TARGET_RATE_DIVISOR = Fraction(30, 1)
DEFAULT_TARGET = Fraction(325, 799)
DEFAULT_TARGET_RATE = DEFAULT_TARGET * TARGET_RATE_DIVISOR
DEFAULT_TARGET_FRACTION_PLACEHOLDER = "325/799"
DEFAULT_TARGET_RATE_PLACEHOLDER = f"{float(DEFAULT_TARGET_RATE):.1f}"
RESULT_LIMIT = 20
FAST_PRESET_LABEL = "Fast"
SLOW_PRESET_LABEL = "Slow"
BASE_WINDOW_WIDTH = 1600
BASE_WINDOW_HEIGHT = 900
DEFAULT_SIDEBAR_WIDTH = 300
DEFAULT_TREE_HEIGHT = 180
GA_ENTRY_WIDTH = 5
TARGET_MODE_FRACTION = "fraction"
TARGET_MODE_RATE = "rate"


def configure_windows_dpi_awareness() -> None:
    if sys.platform != "win32":
        return

    try:
        # Prefer system DPI awareness for more stable IME/candidate window sizing on high-DPI displays.
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass

    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class SearchSettings:
    target_input_mode: str
    target_value: Fraction
    target_rate_per_min: Fraction
    population_size: int
    max_generations: int
    max_depth: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int


@dataclass(frozen=True)
class PresetSettings:
    population_size: int
    max_generations: int
    max_depth: int
    crossover_rate: float
    mutation_rate: float
    tournament_size: int


PRESET_SETTINGS = {
    FAST_PRESET_LABEL: PresetSettings(
        population_size=50,
        max_generations=100,
        max_depth=5,
        crossover_rate=0.8,
        mutation_rate=0.8,
        tournament_size=5,
    ),
    SLOW_PRESET_LABEL: PresetSettings(
        population_size=200,
        max_generations=400,
        max_depth=5,
        crossover_rate=0.8,
        mutation_rate=0.8,
        tournament_size=5,
    ),
}

DEFAULT_PRESET = PRESET_SETTINGS[FAST_PRESET_LABEL]


@dataclass
class SolutionSummary:
    index: int
    error: float
    cost: float
    output: float
    tree_text: str
    tree_data: dict
    preview_image: Image.Image


def format_error_percentage(error: float, target_value: Fraction | float | str) -> str:
    target = float(Fraction(target_value))
    if abs(target) < 1e-12:
        return "0.00%" if abs(error) < 1e-12 else "N/A"
    return f"{error / abs(target) * 100:.2f}%"


def format_target_rate(rate_per_min: Fraction | float | str) -> str:
    rate = float(Fraction(rate_per_min))
    return f"{rate:.3f}".rstrip("0").rstrip(".")


def format_target_expression(rate_per_min: Fraction | float | str) -> str:
    rate_fraction = Fraction(rate_per_min)
    target_value = rate_fraction / TARGET_RATE_DIVISOR
    return f"{format_target_rate(rate_fraction)}/30 ≈ {float(target_value):.6f}"


def format_fraction_target_expression(target_value: Fraction | float | str) -> str:
    target_fraction = Fraction(target_value)
    return f"{target_fraction} ≈ {float(target_fraction):.6f}"


class TopologySearchApp:
    def __init__(self) -> None:
        configure_windows_dpi_awareness()
        self.root = Tk()
        self.root.title(f"{APP_TITLE} - v{APP_VERSION}")
        self._configure_ui_font()
        self.ui_scale = 1.0
        self.sidebar_width = DEFAULT_SIDEBAR_WIDTH
        self.tree_height = DEFAULT_TREE_HEIGHT
        self._configure_initial_window_metrics()
        self.root.state("zoomed")
        self.default_entry_foreground = "black"
        self.placeholder_entry_foreground = "#888888"

        self.target_var = StringVar(value="")
        self.population_var = StringVar(value=str(DEFAULT_PRESET.population_size))
        self.generations_var = StringVar(value=str(DEFAULT_PRESET.max_generations))
        self.max_depth_var = StringVar(value=str(DEFAULT_PRESET.max_depth))
        self.crossover_var = StringVar(value=str(DEFAULT_PRESET.crossover_rate))
        self.mutation_var = StringVar(value=str(DEFAULT_PRESET.mutation_rate))
        self.tournament_var = StringVar(value=str(DEFAULT_PRESET.tournament_size))
        self.status_var = StringVar(value="就绪")
        self.solution_var = StringVar(value="未选择结果")

        self.solutions: list[SolutionSummary] = []
        self.search_thread: threading.Thread | None = None
        self.stop_search_event = threading.Event()
        self.stop_requested = False
        self.current_preview: ImageTk.PhotoImage | None = None
        self.current_settings: SearchSettings | None = None
        self.preview_source_image: Image.Image | None = None
        self.preview_scale = 1.0
        self.preview_manual_zoom = False
        self.content: ttk.Panedwindow | None = None
        self._initial_layout_applied = False
        self.ga_params_collapsed = True
        self.ga_params_container: ttk.Frame | None = None
        self.ga_toggle_button: ttk.Button | None = None
        self.target_mode = TARGET_MODE_FRACTION
        self.target_mode_button: ttk.Button | None = None
        self.target_suffix_label: ttk.Label | None = None
        self.target_placeholder_active = False
        self.editable_entries: list[ttk.Entry] = []

        self._build_ui()
        self._refresh_target_mode_widgets()
        self._activate_target_placeholder()
        self.root.bind_all("<Button-1>", self._clear_entry_selection_on_click, add="+")
        self.root.after(100, self._apply_initial_layout)

    def _configure_initial_window_metrics(self) -> None:
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        scale_x = screen_width / BASE_WINDOW_WIDTH
        scale_y = screen_height / BASE_WINDOW_HEIGHT
        self.ui_scale = min(max(min(scale_x, scale_y), 0.85), 1.5)

        window_width = min(int(screen_width * 0.9), int(BASE_WINDOW_WIDTH * self.ui_scale))
        window_height = min(int(screen_height * 0.9), int(BASE_WINDOW_HEIGHT * self.ui_scale))
        self.root.geometry(
            f"{window_width}x{window_height}+{max((screen_width - window_width) // 2, 0)}+{max((screen_height - window_height) // 2, 0)}"
        )

        self.sidebar_width = int(DEFAULT_SIDEBAR_WIDTH * self.ui_scale)
        self.tree_height = int(DEFAULT_TREE_HEIGHT * self.ui_scale)

    def _configure_ui_font(self) -> None:
        preferred_families = [
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "微软雅黑",
            "PingFang SC",
            "Noto Sans CJK SC",
            "SimHei",
            "黑体",
            "DengXian",
            "等线",
        ]
        available_families = set(tkfont.families(self.root))
        selected_family = next((name for name in preferred_families if name in available_families), None)
        if selected_family is None:
            return

        named_fonts = [
            "TkDefaultFont",
            "TkTextFont",
            "TkHeadingFont",
            "TkMenuFont",
            "TkCaptionFont",
            "TkSmallCaptionFont",
            "TkIconFont",
            "TkTooltipFont",
            "TkFixedFont",
        ]
        for font_name in named_fonts:
            try:
                named_font = tkfont.nametofont(font_name)
            except tk.TclError:
                continue
            named_font.configure(family=selected_family)

        self._configure_treeview_style()

    def _configure_treeview_style(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        rowheight = max(default_font.metrics("linespace") + 8, 24)
        heading_font = tkfont.nametofont("TkHeadingFont")
        style = ttk.Style(self.root)
        style.configure("Treeview", rowheight=rowheight, font=default_font)
        style.configure("Treeview.Heading", font=heading_font)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        content = ttk.Panedwindow(self.root, orient="horizontal")
        content.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.content = content

        sidebar = ttk.Frame(content, padding=8)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(0, weight=0)
        sidebar.rowconfigure(1, weight=0)
        sidebar.rowconfigure(2, weight=0)
        sidebar.rowconfigure(3, weight=1)
        content.add(sidebar, weight=0)

        search_frame = ttk.LabelFrame(sidebar, text="搜索", padding=10)
        search_frame.grid(row=0, column=0, sticky="nsew")
        search_frame.columnconfigure(0, minsize=52)
        search_frame.columnconfigure(1, minsize=0)
        search_frame.columnconfigure(2, minsize=40)
        search_frame.columnconfigure(3, weight=1)
        search_frame.columnconfigure(4, minsize=0)
        search_frame.rowconfigure(3, weight=1)
        search_frame.bind("<Configure>", self._on_search_frame_resize)

        ttk.Label(search_frame, text="目标值").grid(row=0, column=0, padx=(0, 8), pady=4, sticky="w")
        self.target_entry = ttk.Entry(search_frame, textvariable=self.target_var, width=7)
        self.target_entry.grid(row=0, column=1, padx=(0, 4), pady=4, sticky="w")
        self.target_entry.bind("<FocusIn>", self._on_target_focus_in)
        self.target_entry.bind("<FocusOut>", self._on_target_focus_out)
        self.target_suffix_label = ttk.Label(search_frame, text="/min")
        self.target_suffix_label.grid(row=0, column=2, padx=(0, 8), pady=4, sticky="w")
        button_frame = ttk.Frame(search_frame)
        button_frame.grid(row=0, column=4, padx=(0, 0), pady=4, sticky="e")
        self.target_mode_button = ttk.Button(
            button_frame,
            text="分数",
            width=5,
            command=self._toggle_target_mode,
            takefocus=False,
        )
        self.target_mode_button.grid(row=0, column=0, padx=(0, 16), pady=0, sticky="e")

        self.run_button = ttk.Button(button_frame, text="搜索", width=5, command=self.start_search, takefocus=False)
        self.run_button.grid(row=0, column=1, padx=(0, 0), pady=0, sticky="e")

        self.status_label = tk.Label(
            search_frame,
            textvariable=self.status_var,
            anchor="nw",
            justify="left",
            height=2,
            background=self.root.cget("background"),
        )
        self.status_label.grid(row=1, column=0, columnspan=5, pady=(4, 8), sticky="ew")

        ga_frame = ttk.LabelFrame(sidebar, text="遗传算法参数", padding=10)
        ga_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        ga_frame.columnconfigure(0, weight=1)

        ga_header = ttk.Frame(ga_frame)
        ga_header.grid(row=0, column=0, sticky="ew")
        ga_header.columnconfigure(4, weight=1)

        ttk.Label(ga_header, text="预设").grid(row=0, column=0, padx=(0, 8), pady=4, sticky="w")
        self.fast_preset_button = ttk.Button(
            ga_header,
            text="Fast",
            width=GA_ENTRY_WIDTH,
            takefocus=False,
            command=lambda: self._apply_preset(FAST_PRESET_LABEL),
        )
        self.fast_preset_button.grid(row=0, column=1, padx=(8, 12), pady=4, sticky="w")
        self.slow_preset_button = ttk.Button(
            ga_header,
            text="Slow",
            width=GA_ENTRY_WIDTH,
            takefocus=False,
            command=lambda: self._apply_preset(SLOW_PRESET_LABEL),
        )
        self.slow_preset_button.grid(row=0, column=2, padx=(8, 8), pady=4, sticky="w")
        self.ga_toggle_button = ttk.Button(
            ga_header,
            text="展开参数",
            width=8,
            takefocus=False,
            command=self._toggle_ga_params,
        )
        self.ga_toggle_button.grid(row=0, column=5, padx=(8, 0), pady=4, sticky="e")

        self.ga_params_container = ttk.Frame(ga_frame)
        self.ga_params_container.grid(row=1, column=0, sticky="ew")

        ttk.Label(self.ga_params_container, text="Population").grid(row=0, column=0, padx=(0, 8), pady=4, sticky="w")
        self.population_entry = ttk.Entry(
            self.ga_params_container, textvariable=self.population_var, width=GA_ENTRY_WIDTH
        )
        self.population_entry.grid(row=0, column=1, padx=(0, 12), pady=4, sticky="w")
        ttk.Label(self.ga_params_container, text="Generations").grid(row=0, column=2, padx=(8, 8), pady=4, sticky="w")
        self.generations_entry = ttk.Entry(
            self.ga_params_container, textvariable=self.generations_var, width=GA_ENTRY_WIDTH
        )
        self.generations_entry.grid(row=0, column=3, padx=(0, 12), pady=4, sticky="w")
        ttk.Label(self.ga_params_container, text="Crossover").grid(row=1, column=0, padx=(0, 8), pady=4, sticky="w")
        self.crossover_entry = ttk.Entry(
            self.ga_params_container, textvariable=self.crossover_var, width=GA_ENTRY_WIDTH
        )
        self.crossover_entry.grid(row=1, column=1, padx=(0, 12), pady=4, sticky="w")
        ttk.Label(self.ga_params_container, text="Mutation").grid(row=1, column=2, padx=(8, 8), pady=4, sticky="w")
        self.mutation_entry = ttk.Entry(self.ga_params_container, textvariable=self.mutation_var, width=GA_ENTRY_WIDTH)
        self.mutation_entry.grid(row=1, column=3, padx=(0, 12), pady=4, sticky="w")
        ttk.Label(self.ga_params_container, text="Max depth").grid(row=2, column=0, padx=(0, 8), pady=4, sticky="w")
        self.max_depth_entry = ttk.Entry(
            self.ga_params_container, textvariable=self.max_depth_var, width=GA_ENTRY_WIDTH
        )
        self.max_depth_entry.grid(row=2, column=1, padx=(0, 12), pady=4, sticky="w")
        ttk.Label(self.ga_params_container, text="Tournament").grid(row=2, column=2, padx=(8, 8), pady=4, sticky="w")
        self.tournament_entry = ttk.Entry(
            self.ga_params_container, textvariable=self.tournament_var, width=GA_ENTRY_WIDTH
        )
        self.tournament_entry.grid(row=2, column=3, padx=(0, 12), pady=4, sticky="w")
        self.editable_entries = [
            self.target_entry,
            self.population_entry,
            self.generations_entry,
            self.max_depth_entry,
            self.crossover_entry,
            self.mutation_entry,
            self.tournament_entry,
        ]
        self._set_ga_params_collapsed(True)

        results_frame = ttk.LabelFrame(sidebar, text="结果", padding=8)
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        columns = ("rank", "output", "error", "cost")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)
        self.results_tree.heading("rank", text="#")
        self.results_tree.heading("output", text="产出")
        self.results_tree.heading("error", text="误差")
        self.results_tree.heading("cost", text="成本")
        self.results_tree.column("rank", width=44, anchor="center", stretch=False)
        self.results_tree.column("output", width=96, anchor="center")
        self.results_tree.column("error", width=96, anchor="center")
        self.results_tree.column("cost", width=72, anchor="center")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self.on_solution_selected)

        results_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_tree.yview)
        results_scroll.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=results_scroll.set)

        log_frame = ttk.LabelFrame(sidebar, text="运行日志", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(log_frame, wrap="word", height=14, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        preview_side = ttk.Frame(content, padding=8)
        preview_side.columnconfigure(0, weight=1)
        preview_side.rowconfigure(0, minsize=self.tree_height, weight=0)
        preview_side.rowconfigure(1, weight=1)
        content.add(preview_side, weight=5)

        tree_frame = ttk.LabelFrame(preview_side, text="树结构", padding=8)
        tree_frame.grid(row=0, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.configure(height=self.tree_height)
        tree_frame.grid_propagate(False)

        self.tree_text = ScrolledText(tree_frame, wrap="none", height=8, state="disabled")
        self.tree_text.grid(row=0, column=0, sticky="nsew")

        preview_frame = ttk.LabelFrame(preview_side, text="布局预览", padding=8)
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)

        ttk.Label(preview_frame, textvariable=self.solution_var).grid(row=0, column=0, sticky="w", pady=(0, 8))

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(canvas_frame, background="#f6f4ef", highlightthickness=0)
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")

        preview_y = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.preview_canvas.yview)
        preview_y.grid(row=0, column=1, sticky="ns")
        preview_x = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.preview_canvas.xview)
        preview_x.grid(row=1, column=0, sticky="ew")
        self.preview_canvas.configure(yscrollcommand=preview_y.set, xscrollcommand=preview_x.set)
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        self.preview_canvas.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self.preview_canvas.bind("<B1-Motion>", self._on_preview_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_preview_drag_end)

    def start_search(self) -> None:
        if self.search_thread and self.search_thread.is_alive():
            self.request_stop_search()
            return

        try:
            settings = self._collect_settings()
        except ValueError as exc:
            messagebox.showerror("参数无效", str(exc))
            return

        self.stop_search_event.clear()
        self.stop_requested = False
        self._clear_results()
        self._set_busy(True)
        self.current_settings = settings
        self._append_log(f"Starting search for target={self._format_target_display(settings)}")
        self._append_log(
            "Using "
            f"population={settings.population_size}, generations={settings.max_generations}, "
            f"max_depth={settings.max_depth}, crossover={settings.crossover_rate}, "
            f"mutation={settings.mutation_rate}, tournament={settings.tournament_size}"
        )

        self.search_thread = threading.Thread(target=self._run_search, args=(settings,), daemon=True)
        self.search_thread.start()

    def _run_search(self, settings: SearchSettings) -> None:
        try:
            Config.TARGET_VAL = float(settings.target_value)
            Config.MAX_DEPTH = settings.max_depth
            Config.CROSSOVER_RATE = settings.crossover_rate
            Config.MUTATION_RATE = settings.mutation_rate
            Config.TOURNAMENT_SIZE = settings.tournament_size
            solver = NSGA2(
                settings.population_size,
                settings.max_generations,
                settings.crossover_rate,
                settings.mutation_rate,
            )
            completed = solver.run(
                progress_callback=self._queue_progress_update,
                progress_interval=10,
                should_stop=self.stop_search_event.is_set,
            )

            solutions: list[SolutionSummary] = []
            for index, individual in enumerate(solver.get_sorted_unique_front()[:RESULT_LIMIT], start=1):
                tree_data = individual.chromosome.to_dict()
                preview = render_tree_preview_from_dict(tree_data)
                solutions.append(
                    SolutionSummary(
                        index=index,
                        error=float(individual.objectives[0]),
                        cost=float(individual.objectives[1]),
                        output=float(individual.chromosome.get_output()),
                        tree_text="\n".join(
                            line.lstrip() for line in individual.chromosome.format(indent=10).splitlines()
                        ),
                        tree_data=tree_data,
                        preview_image=preview,
                    )
                )

            self.root.after(0, lambda: self._finish_search(solutions, settings, completed))
        except Exception as exc:
            details = traceback.format_exc()
            self.root.after(0, lambda: self._handle_search_error(exc, details))

    def _queue_progress_update(self, snapshot: dict) -> None:
        self.root.after(0, lambda: self._apply_progress_update(snapshot))

    def _apply_progress_update(self, snapshot: dict) -> None:
        generation = int(snapshot["generation"])
        pareto_size = int(snapshot["pareto_front_size"])
        front_count = int(snapshot["front_count"])
        avg_error = snapshot["avg_error"]
        avg_cost = snapshot["avg_cost"]

        total_generations = self.current_settings.max_generations if self.current_settings else 0
        status = f"Generation {generation}/{total_generations} | Pareto {pareto_size} | Fronts {front_count}"
        if avg_error is not None and avg_cost is not None:
            target_value = self.current_settings.target_value if self.current_settings is not None else DEFAULT_TARGET
            status += f" | Avg error {format_error_percentage(avg_error, target_value)} | Avg cost {avg_cost:.2f}"
        self.status_var.set(status)
        self._append_log(status)

    def _finish_search(self, solutions: list[SolutionSummary], settings: SearchSettings, completed: bool) -> None:
        self._set_busy(False)
        self.stop_requested = False
        self.stop_search_event.clear()
        self.solutions = solutions

        if not solutions:
            self.status_var.set(
                "搜索已中止，但没有可返回的 Pareto 结果。"
                if not completed
                else "搜索完成，但没有找到有效的 Pareto 结果。"
            )
            self._append_log(
                "Search stopped with no valid Pareto solutions."
                if not completed
                else "Search completed with no valid Pareto solutions."
            )
            return

        for solution in solutions:
            iid = self._solution_iid(solution.index)
            self.results_tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    solution.index,
                    f"{solution.output:.6f}",
                    format_error_percentage(solution.error, settings.target_value),
                    f"{solution.cost:.0f}",
                ),
            )

        if completed:
            self.status_var.set(
                f"target={self._format_target_display(settings)} 搜索完成。点击任意结果可查看布局预览。"
            )
            self._append_log(f"Search completed. Displaying {len(solutions)} solutions.")
        else:
            self.status_var.set(
                f"target={self._format_target_display(settings)} 搜索已中止，已返回当前结果。"
            )
            self._append_log(f"Search stopped. Displaying {len(solutions)} partial solutions.")
        first_iid = self._solution_iid(1)
        self.results_tree.selection_set(first_iid)
        self.results_tree.focus(first_iid)
        self.results_tree.see(first_iid)
        self._display_solution(self.solutions[0])

    def _handle_search_error(self, exc: Exception, details: str) -> None:
        self._set_busy(False)
        self.stop_requested = False
        self.stop_search_event.clear()
        self.status_var.set("搜索失败")
        self._append_log("Search failed:")
        self._append_log(details)
        messagebox.showerror("搜索失败", str(exc))

    def on_solution_selected(self, _event: object) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        index = int(selection[0].split("-")[-1]) - 1
        if 0 <= index < len(self.solutions):
            self._display_solution(self.solutions[index])

    def _display_solution(self, solution: SolutionSummary) -> None:
        target_value = self.current_settings.target_value if self.current_settings is not None else DEFAULT_TARGET
        self.solution_var.set(
            f"结果 {solution.index} | 误差={format_error_percentage(solution.error, target_value)} | 成本={solution.cost:.0f} | 产出={solution.output:.6f}"
        )

        self.preview_source_image = solution.preview_image
        self.preview_manual_zoom = False
        self._fit_preview_to_canvas()
        self._render_preview()

        self.tree_text.configure(state="normal")
        self.tree_text.delete("1.0", "end")
        self.tree_text.insert("1.0", solution.tree_text)
        self.tree_text.configure(state="disabled")

    def _clear_results(self) -> None:
        self.solutions = []
        self.current_settings = None
        self.current_preview = None
        self.preview_source_image = None
        self.preview_scale = 1.0
        self.preview_manual_zoom = False
        self.status_var.set("运行中...")
        self.solution_var.set("未选择结果")
        self.results_tree.delete(*self.results_tree.get_children())
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(scrollregion=(0, 0, 0, 0))
        self.tree_text.configure(state="normal")
        self.tree_text.delete("1.0", "end")
        self.tree_text.configure(state="disabled")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _set_busy(self, busy: bool) -> None:
        editable_entries = [
            self.target_entry,
            self.population_entry,
            self.generations_entry,
            self.max_depth_entry,
            self.crossover_entry,
            self.mutation_entry,
            self.tournament_entry,
        ]
        if busy:
            self.run_button.configure(state="normal", text="中止")
            self.fast_preset_button.configure(state="disabled")
            self.slow_preset_button.configure(state="disabled")
            if self.target_mode_button is not None:
                self.target_mode_button.configure(state="disabled")
            for entry in editable_entries:
                entry.configure(state="disabled")
        else:
            self.run_button.configure(state="normal", text="搜索")
            self.fast_preset_button.configure(state="normal")
            self.slow_preset_button.configure(state="normal")
            if self.target_mode_button is not None:
                self.target_mode_button.configure(state="normal")
            for entry in editable_entries:
                entry.configure(state="normal")

    def request_stop_search(self) -> None:
        if not self.search_thread or not self.search_thread.is_alive():
            return
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_search_event.set()
        self.status_var.set("正在中止搜索并整理当前结果...")
        self._append_log("Stop requested. Finalizing current Pareto results.")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _toggle_target_mode(self) -> None:
        self.target_mode = TARGET_MODE_RATE if self.target_mode == TARGET_MODE_FRACTION else TARGET_MODE_FRACTION
        self._activate_target_placeholder()
        self._refresh_target_mode_widgets()

    def _refresh_target_mode_widgets(self) -> None:
        if self.target_mode_button is not None:
            self.target_mode_button.configure(text="分数" if self.target_mode == TARGET_MODE_RATE else "小数")
        if self.target_suffix_label is not None:
            if self.target_mode == TARGET_MODE_RATE:
                self.target_suffix_label.grid()
            else:
                self.target_suffix_label.grid_remove()

    def _format_target_display(self, settings: SearchSettings) -> str:
        if settings.target_input_mode == TARGET_MODE_RATE:
            return format_target_expression(settings.target_rate_per_min)
        return format_fraction_target_expression(settings.target_value)

    def _toggle_ga_params(self) -> None:
        self._set_ga_params_collapsed(not self.ga_params_collapsed)

    def _set_ga_params_collapsed(self, collapsed: bool) -> None:
        self.ga_params_collapsed = collapsed
        if self.ga_params_container is not None:
            if collapsed:
                self.ga_params_container.grid_remove()
            else:
                self.ga_params_container.grid()
        if self.ga_toggle_button is not None:
            self.ga_toggle_button.configure(text="展开参数" if collapsed else "折叠参数")

    def _apply_initial_layout(self) -> None:
        if self.content is None or self._initial_layout_applied:
            return

        try:
            self.root.update_idletasks()
            self.content.sashpos(0, self.sidebar_width)
            self._initial_layout_applied = True
        except tk.TclError:
            self.root.after(100, self._apply_initial_layout)

    def _on_search_frame_resize(self, event: tk.Event) -> None:
        wraplength = max(event.width - 24, 40)
        self.status_label.configure(wraplength=wraplength)

    def _activate_target_placeholder(self) -> None:
        self.target_placeholder_active = True
        if self.target_mode == TARGET_MODE_RATE:
            self.target_var.set(DEFAULT_TARGET_RATE_PLACEHOLDER)
        else:
            self.target_var.set(DEFAULT_TARGET_FRACTION_PLACEHOLDER)
        self.target_entry.configure(foreground=self.placeholder_entry_foreground)

    def _deactivate_target_placeholder(self) -> None:
        if not self.target_placeholder_active:
            return
        self.target_placeholder_active = False
        self.target_var.set("")
        self.target_entry.configure(foreground=self.default_entry_foreground)

    def _on_target_focus_in(self, _event: tk.Event) -> None:
        self._deactivate_target_placeholder()

    def _on_target_focus_out(self, _event: tk.Event) -> None:
        if self.target_var.get().strip():
            return
        self._activate_target_placeholder()

    def _clear_entry_selection_on_click(self, event: tk.Event) -> None:
        clicked_widget = event.widget
        clicked_entry = clicked_widget if isinstance(clicked_widget, ttk.Entry) else None

        for entry in self.editable_entries:
            if entry is clicked_entry:
                continue
            try:
                entry.selection_clear()
            except tk.TclError:
                continue

        if clicked_entry is not None:
            return

        focused_widget = self.root.focus_get()
        if focused_widget in self.editable_entries:
            self.root.focus_set()

    def _on_preview_resize(self, _event: tk.Event) -> None:
        if self.preview_source_image is None or self.preview_manual_zoom:
            return
        self._fit_preview_to_canvas()
        self._render_preview()

    def _on_preview_mousewheel(self, event: tk.Event) -> str:
        if self.preview_source_image is None:
            return "break"

        zoom_factor = 1.1 if event.delta > 0 else 1 / 1.1
        fit_scale = self._get_fit_preview_scale()
        old_scale = self.preview_scale
        old_metrics = self._get_preview_metrics(old_scale)
        old_image_x = self.preview_canvas.canvasx(event.x) - old_metrics["x_offset"]
        old_image_y = self.preview_canvas.canvasy(event.y) - old_metrics["y_offset"]
        focus_rel_x = min(max(old_image_x / max(old_metrics["scaled_width"], 1), 0.0), 1.0)
        focus_rel_y = min(max(old_image_y / max(old_metrics["scaled_height"], 1), 0.0), 1.0)

        new_scale = min(max(old_scale * zoom_factor, fit_scale), 8.0)
        if abs(new_scale - self.preview_scale) < 1e-9:
            return "break"

        self.preview_scale = new_scale
        self.preview_manual_zoom = new_scale > fit_scale + 1e-9
        self._render_preview(
            focus_x=event.x,
            focus_y=event.y,
            focus_rel_x=focus_rel_x,
            focus_rel_y=focus_rel_y,
        )
        return "break"

    def _on_preview_drag_start(self, event: tk.Event) -> str:
        if self.preview_source_image is None:
            return "break"

        self.preview_canvas.scan_mark(event.x, event.y)
        self.preview_canvas.configure(cursor="fleur")
        return "break"

    def _on_preview_drag(self, event: tk.Event) -> str:
        if self.preview_source_image is None:
            return "break"

        self.preview_manual_zoom = True
        self.preview_canvas.scan_dragto(event.x, event.y, gain=1)
        return "break"

    def _on_preview_drag_end(self, _event: tk.Event) -> str:
        self.preview_canvas.configure(cursor="")
        return "break"

    def _fit_preview_to_canvas(self) -> None:
        if self.preview_source_image is None:
            return

        self.preview_scale = self._get_fit_preview_scale()

    def _get_fit_preview_scale(self) -> float:
        if self.preview_source_image is None:
            return 1.0

        canvas_width = max(self.preview_canvas.winfo_width(), 1)
        canvas_height = max(self.preview_canvas.winfo_height(), 1)
        image_width, image_height = self.preview_source_image.size

        if image_width <= 0 or image_height <= 0:
            return 1.0

        return min(canvas_width / image_width, canvas_height / image_height, 1.0)

    def _get_preview_metrics(self, scale: float) -> dict[str, int]:
        if self.preview_source_image is None:
            return {
                "scaled_width": 0,
                "scaled_height": 0,
                "canvas_width": max(self.preview_canvas.winfo_width(), 1),
                "canvas_height": max(self.preview_canvas.winfo_height(), 1),
                "x_offset": 0,
                "y_offset": 0,
                "content_width": 0,
                "content_height": 0,
            }

        scaled_width = max(1, int(round(self.preview_source_image.width * scale)))
        scaled_height = max(1, int(round(self.preview_source_image.height * scale)))
        canvas_width = max(self.preview_canvas.winfo_width(), 1)
        canvas_height = max(self.preview_canvas.winfo_height(), 1)
        x_offset = max((canvas_width - scaled_width) // 2, 0)
        y_offset = max((canvas_height - scaled_height) // 2, 0)
        content_width = max(scaled_width, canvas_width)
        content_height = max(scaled_height, canvas_height)
        return {
            "scaled_width": scaled_width,
            "scaled_height": scaled_height,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "content_width": content_width,
            "content_height": content_height,
        }

    def _render_preview(
        self,
        focus_x: int | None = None,
        focus_y: int | None = None,
        focus_rel_x: float | None = None,
        focus_rel_y: float | None = None,
    ) -> None:
        if self.preview_source_image is None:
            self.preview_canvas.delete("all")
            return

        metrics = self._get_preview_metrics(self.preview_scale)
        scaled_width = metrics["scaled_width"]
        scaled_height = metrics["scaled_height"]
        resized_image = self.preview_source_image.resize((scaled_width, scaled_height), Image.LANCZOS)

        self.current_preview = ImageTk.PhotoImage(resized_image)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(
            metrics["x_offset"], metrics["y_offset"], anchor="nw", image=self.current_preview
        )
        self.preview_canvas.configure(scrollregion=(0, 0, metrics["content_width"], metrics["content_height"]))

        if focus_x is not None and focus_y is not None and focus_rel_x is not None and focus_rel_y is not None:
            new_canvas_x = metrics["x_offset"] + focus_rel_x * metrics["scaled_width"]
            new_canvas_y = metrics["y_offset"] + focus_rel_y * metrics["scaled_height"]
            self._move_preview_viewport_to(new_canvas_x - focus_x, new_canvas_y - focus_y, metrics)
        else:
            self.preview_canvas.xview_moveto(0)
            self.preview_canvas.yview_moveto(0)

    def _move_preview_viewport_to(self, left: float, top: float, metrics: dict[str, int]) -> None:
        max_left = max(metrics["content_width"] - metrics["canvas_width"], 0)
        max_top = max(metrics["content_height"] - metrics["canvas_height"], 0)
        clamped_left = min(max(left, 0), max_left)
        clamped_top = min(max(top, 0), max_top)

        if max_left > 0:
            self.preview_canvas.xview_moveto(clamped_left / metrics["content_width"])
        else:
            self.preview_canvas.xview_moveto(0)

        if max_top > 0:
            self.preview_canvas.yview_moveto(clamped_top / metrics["content_height"])
        else:
            self.preview_canvas.yview_moveto(0)

    @staticmethod
    def _solution_iid(index: int) -> str:
        return f"solution-{index}"

    def _apply_preset(self, preset_name: str) -> None:
        preset = PRESET_SETTINGS[preset_name]
        self.population_var.set(str(preset.population_size))
        self.generations_var.set(str(preset.max_generations))
        self.max_depth_var.set(str(preset.max_depth))
        self.crossover_var.set(str(preset.crossover_rate))
        self.mutation_var.set(str(preset.mutation_rate))
        self.tournament_var.set(str(preset.tournament_size))
        self.status_var.set(f"已加载 {preset_name} preset")
        self._append_log(
            f"Loaded {preset_name} preset: population={preset.population_size}, "
            f"generations={preset.max_generations}, max_depth={preset.max_depth}, "
            f"crossover={preset.crossover_rate}, mutation={preset.mutation_rate}, "
            f"tournament={preset.tournament_size}"
        )

    def _collect_settings(self) -> SearchSettings:
        target_text = "" if self.target_placeholder_active else self.target_var.get().strip()
        if not target_text:
            target_text = (
                DEFAULT_TARGET_RATE_PLACEHOLDER
                if self.target_mode == TARGET_MODE_RATE
                else DEFAULT_TARGET_FRACTION_PLACEHOLDER
            )
        try:
            if self.target_mode == TARGET_MODE_RATE:
                target_rate_per_min = Fraction(target_text)
                target_value = target_rate_per_min / TARGET_RATE_DIVISOR
            else:
                target_value = Fraction(target_text)
                target_rate_per_min = target_value * TARGET_RATE_DIVISOR
        except Exception as exc:
            if self.target_mode == TARGET_MODE_RATE:
                raise ValueError("目标值必须是数字，例如 13.5。") from exc
            raise ValueError("目标值必须是类似 2/5、0.4 或 1/3 这样的格式。") from exc

        population_size = self._parse_int(self.population_var.get(), "Population", minimum=2)
        max_generations = self._parse_int(self.generations_var.get(), "Generations", minimum=1)
        max_depth = self._parse_int(self.max_depth_var.get(), "Max depth", minimum=1)
        crossover_rate = self._parse_float(self.crossover_var.get(), "Crossover", minimum=0.0, maximum=1.0)
        mutation_rate = self._parse_float(self.mutation_var.get(), "Mutation", minimum=0.0, maximum=1.0)
        tournament_size = self._parse_int(self.tournament_var.get(), "Tournament", minimum=2)

        if tournament_size > population_size:
            raise ValueError("Tournament 不能大于 Population。")

        return SearchSettings(
            target_input_mode=self.target_mode,
            target_value=target_value,
            target_rate_per_min=target_rate_per_min,
            population_size=population_size,
            max_generations=max_generations,
            max_depth=max_depth,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
        )

    @staticmethod
    def _parse_int(raw_value: str, label: str, minimum: int) -> int:
        try:
            value = int(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} 必须是整数。") from exc
        if value < minimum:
            raise ValueError(f"{label} 必须不小于 {minimum}。")
        return value

    @staticmethod
    def _parse_float(raw_value: str, label: str, minimum: float, maximum: float) -> float:
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"{label} 必须是数字。") from exc
        if not (minimum <= value <= maximum):
            raise ValueError(f"{label} 必须在 {minimum} 到 {maximum} 之间。")
        return value

    def run(self) -> None:
        self.root.mainloop()


def run_smoke_test(output_path: Path | None = None) -> Path:
    target_value = DEFAULT_TARGET
    Config.TARGET_VAL = float(target_value)
    solver = NSGA2(40, 20, Config.CROSSOVER_RATE, Config.MUTATION_RATE)
    solver.run(progress_callback=None, progress_interval=5)
    best = solver.get_sorted_unique_front()
    if not best:
        raise RuntimeError("Smoke test produced no valid solutions.")

    image = render_tree_preview_from_dict(best[0].chromosome.to_dict())
    resolved_output = output_path or Path(__file__).resolve().parent.parent / "smoke_preview.png"
    image.save(resolved_output)
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description="终末地传送带拓扑搜索桌面界面。")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="运行一个简短的非 GUI smoke test，并输出一张预览图。",
    )
    args = parser.parse_args()

    if args.smoke_test:
        output_path = run_smoke_test()
        print(f"Smoke test 预览图已写入 {output_path}")
        return

    app = TopologySearchApp()
    app.run()


if __name__ == "__main__":
    main()
