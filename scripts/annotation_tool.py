#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import hashlib
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from PIL import Image, ImageTk

from annotation_assist import ImageLoader, predict_boxes, propagate_boxes
from annotation_session import DatasetConfig, Session
from annotation_store import (
    Box,
    LabelError,
    ReviewStore,
    clamp,
    collect_images,
    copy_boxes,
    label_path_for,
    read_classes,
    read_yolo_boxes,
    write_yolo_boxes,
)
from annotation_workflow import export_verified, reference_crops


DATASET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSES_FILE = DATASET_ROOT / "classes.txt"
DEFAULT_GEOMETRY = "1500x920"
HANDLE_RADIUS_PX = 6
MIN_BOX_SIZE_PX = 4.0
MAX_UNDO_STATES = 100

CLASS_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
]

CLASS_ZH = {
    "high voltage insulator": "高壓礙子",
    "transformer box": "變壓器箱",
    "primary cutout switch": "一次側熔斷開關",
    "utility pole": "電桿",
    "low voltage insulator": "低壓礙子",
    "overhead ground wire": "架空地線",
    "protective cover": "防護罩",
    "overhead switch": "架空開關",
    "surge arrester": "避雷器",
    "low voltage wire rack": "低壓線架",
    "HDPE protective cover": "HDPE 防護罩",
    "dead end terminal": "終端線夾",
    "arm cross": "橫擔",
    "fuse link cutout": "熔絲鏈式熔斷器",
    "elbow plug of power pole": "電桿肘型插頭",
}
FILTERS = ("All", "Unreviewed", "Edited", "Confirmed empty", "Confirmed")
STATUS_TEXT = {"pending": "未審閱", "edited": "已修改／待確認", "positive": "已確認有物件", "negative": "已確認無物件"}


class AnnotationApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = DatasetConfig(
            args.images_dir.resolve(), args.labels_dir.resolve(), args.classes_file.resolve(), args.pass_state.resolve()
        )
        self.images_dir = self.config.images_dir
        self.labels_dir = self.config.labels_dir
        self.classes_file = self.config.classes_file
        self.pass_state_path = self.config.pass_state
        self.class_names = read_classes(self.classes_file)
        self.images = collect_images(self.images_dir)
        if not self.images:
            raise LabelError(self.images_dir, "No images found")
        self.store = ReviewStore(self.pass_state_path, self.images, self.class_names, labels_dir=self.labels_dir)
        self.session = Session(self.config, self.class_names)
        restored = self.session.load() if args.resume else {}
        names = [image.name for image in self.images]
        self.index = names.index(restored["image"]) if restored.get("image") in names else 0
        if args.start is not None:
            self.index = int(clamp(args.start, 0, len(self.images) - 1))
        cls = args.start_class if args.start_class is not None else restored.get("class_id", 0)
        self.current_class = int(clamp(cls, 0, len(self.class_names) - 1))
        self.image: Image.Image | None = None
        self.image_path: Path | None = None
        self.boxes: list[Box] = []
        self._saved_boxes: list[Box] = []
        self._saved_label_signature = None
        self.selected_box: int | None = None
        self.dirty = False
        self.undo_stack: list[tuple[list[Box], int | None]] = []
        self.redo_stack: list[tuple[list[Box], int | None]] = []
        self.candidates = []
        self.fit_scale = 1.0
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.drag_start = None
        self.drag_current = None
        self.drag_mode = None
        self.drag_box_index = None
        self.drag_start_image = None
        self.drag_original_box = None
        self.drag_original_snapshot = None
        self.resize_handle = None
        self.pan_start = None
        self.pan_offset_start = None
        self.photo = None
        self.render_cache_key = None
        self.render_cache_photo = None
        self.refreshing_classes = False
        self.is_interacting = False
        self._hq_render_id = None
        self._render_id = None
        self._session_id = None
        self._image_item = None
        self._loading = False
        self._label_error = False
        self._closing = False
        self._load_token = 0
        self._hover = None
        self._lens_photo = None
        self._restore_view = restored
        self._jobs = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotation-assist")
        self._load_timings = deque(maxlen=100)
        self._render_timings = deque(maxlen=100)
        self._notices = list(self.store.notices)
        if self.session.notice:
            self._notices.append(self.session.notice)
        self.root = tk.Tk()
        self.loader = ImageLoader(max_bytes=args.cache_mb * 1024 * 1024)
        self.show_other_classes = tk.BooleanVar(value=args.show_other_classes)
        self.image_mode = tk.BooleanVar(value=restored.get("image_mode", False))
        self.filter_var = tk.StringVar(value=restored.get("filter", "All"))
        if self.filter_var.get() not in FILTERS:
            self.filter_var.set("All")
        self.lens_enabled = tk.BooleanVar(value=restored.get("lens", True))
        self.jump_var = tk.StringVar()
        self.confidence_var = tk.DoubleVar(value=args.confidence)
        self.model_path = args.model
        self.model_mapping = None
        if args.model_mapping:
            mapping = json.loads(args.model_mapping.read_text(encoding="utf-8"))
            if not isinstance(mapping, dict):
                raise LabelError(
                    args.model_mapping, "Model mapping must be a JSON object of model name → dataset name/id."
                )
            self.model_mapping = mapping
        self.root.title(f"Label Power — {self.images_dir.name}")
        self.root.geometry(restored.get("geometry", args.geometry) if args.resume else args.geometry)
        self.root.minsize(900, 620)
        self._configure_style()
        self._build_ui()
        self._bind_events()
        self.load_image(self.index)
        self.root.after(20, self._drain_background)

    def _configure_style(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")
        families = set(tkfont.families(self.root))
        self.font_family = next(
            (
                family
                for family in ("Noto Sans CJK TC", "Noto Sans TC", "Microsoft JhengHei", "song ti")
                if family in families
            ),
            default_font.actual("family"),
        )
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(name).configure(family=self.font_family, size=12)
        self.root.option_add("*Font", "TkDefaultFont")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TButton", padding=(8, 5))
        style.configure("TLabel", padding=(2, 2))
        style.configure("Status.TLabel", foreground="#333333")

    def _build_ui(self) -> None:
        menu = tk.Menu(self.root)
        dataset_menu = tk.Menu(menu, tearoff=False)
        dataset_menu.add_command(label="開啟資料集設定…", command=self.open_profile)
        dataset_menu.add_command(label="選擇圖片／標籤目錄…", command=self.choose_dataset)
        dataset_menu.add_command(label="儲存資料集設定…", command=self.save_profile)
        dataset_menu.add_command(label="重新載入目前圖片", command=self.reload_image)
        dataset_menu.add_command(label="匯出已完整審閱資料…", command=self.export_verified)
        dataset_menu.add_command(label="資料集變更通知", command=self.show_notices)
        menu.add_cascade(label="資料集", menu=dataset_menu)
        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="快捷鍵與審閱規則", command=self.show_help)
        help_menu.add_command(label="效能量測", command=self.show_metrics)
        menu.add_cascade(label="說明", menu=help_menu)
        self.root.configure(menu=menu)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")
        canvas_frame = ttk.Frame(self.paned)
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#101114", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.paned.add(canvas_frame, weight=1)
        side = ttk.Frame(self.paned, padding=8, width=380)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(3, weight=1)
        self.paned.add(side, weight=0)
        header = ttk.Frame(side)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        self.pass_label = ttk.Label(header, text="", wraplength=370, font=(self.font_family, 12, "bold"))
        self.pass_label.grid(row=0, column=0, sticky="ew")
        self.progress_label = ttk.Label(header, text="", wraplength=370)
        self.progress_label.grid(row=1, column=0, sticky="ew")
        mode = ttk.Frame(side)
        mode.grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Checkbutton(mode, text="逐圖片總檢查", variable=self.image_mode, command=self.on_mode_changed).pack(
            side="left"
        )
        ttk.Checkbutton(
            mode, text="顯示其他類別", variable=self.show_other_classes, command=self.on_show_other_classes
        ).pack(side="left")
        navigation = ttk.Frame(side)
        navigation.grid(row=2, column=0, sticky="ew", pady=(0, 5))
        for col in range(3):
            navigation.columnconfigure(col, weight=1)
        self._button(navigation, "上一類", self.prev_class_pass, 0, 0)
        self._button(navigation, "下一類", self.next_class_pass, 0, 1)
        self._button(navigation, "下一未審", self.next_unreviewed_current_class, 0, 2)
        self.filter_combo = ttk.Combobox(
            navigation, textvariable=self.filter_var, values=FILTERS, state="readonly", width=17
        )
        self.filter_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_changed)
        jump = ttk.Frame(navigation)
        jump.grid(row=1, column=2, sticky="ew")
        entry = ttk.Entry(jump, textvariable=self.jump_var, width=5)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _event: self.jump_to_image())
        ttk.Button(jump, text="跳轉", command=self.jump_to_image).pack(side="right")
        self.notebook = ttk.Notebook(side)
        self.notebook.grid(row=3, column=0, sticky="nsew")
        classes_tab = ttk.Frame(self.notebook)
        classes_tab.columnconfigure(0, weight=1)
        classes_tab.rowconfigure(0, weight=1)
        self.class_list = tk.Listbox(classes_tab, height=5, width=40, exportselection=False)
        class_scroll = ttk.Scrollbar(classes_tab, orient=tk.VERTICAL, command=self.class_list.yview)
        class_horizontal = ttk.Scrollbar(classes_tab, orient=tk.HORIZONTAL, command=self.class_list.xview)
        self.class_list.configure(yscrollcommand=class_scroll.set, xscrollcommand=class_horizontal.set)
        self.class_list.grid(row=0, column=0, sticky="nsew")
        class_scroll.grid(row=0, column=1, sticky="ns")
        class_horizontal.grid(row=1, column=0, sticky="ew")
        ttk.Button(classes_tab, text="目前類別：已審閱範例", command=self.show_references).grid(
            row=2, column=0, sticky="ew"
        )
        self.notebook.add(classes_tab, text="類別")
        boxes_tab = ttk.Frame(self.notebook)
        boxes_tab.columnconfigure(0, weight=1)
        boxes_tab.rowconfigure(0, weight=1)
        self.box_tree = ttk.Treeview(boxes_tab, columns=("class", "xywh"), show="headings", height=5)
        self.box_tree.heading("class", text="類別")
        self.box_tree.heading("xywh", text="x y w h（像素）")
        self.box_tree.column("class", width=200, minwidth=90, anchor="w")
        self.box_tree.column("xywh", width=120, minwidth=70, anchor="w")
        self.box_tree.grid(row=0, column=0, sticky="nsew")
        ttk.Scrollbar(boxes_tab, orient="vertical", command=self.box_tree.yview).grid(row=0, column=1, sticky="ns")
        self.box_tree.configure(yscrollcommand=boxes_tab.winfo_children()[-1].set)
        ttk.Button(boxes_tab, text="選取框置中放大", command=self.focus_selected_box).grid(row=1, column=0, sticky="ew")
        self.notebook.add(boxes_tab, text="已接受框")
        candidates_tab = ttk.Frame(self.notebook)
        candidates_tab.columnconfigure(0, weight=1)
        candidates_tab.rowconfigure(1, weight=1)
        ttk.Label(candidates_tab, text="虛線候選框不會儲存，也不計入完成率。", wraplength=350).grid(
            row=0, column=0, sticky="w"
        )
        self.candidate_tree = ttk.Treeview(candidates_tab, columns=("class", "confidence"), show="headings", height=5)
        self.candidate_tree.heading("class", text="來源／類別")
        self.candidate_tree.heading("confidence", text="信心")
        self.candidate_tree.column("class", width=260, minwidth=120)
        self.candidate_tree.column("confidence", width=60, minwidth=40)
        self.candidate_tree.grid(row=1, column=0, sticky="nsew")
        candidate_scroll = ttk.Scrollbar(candidates_tab, orient="vertical", command=self.candidate_tree.yview)
        candidate_scroll.grid(row=1, column=1, sticky="ns")
        self.candidate_tree.configure(yscrollcommand=candidate_scroll.set)
        accept_controls = ttk.Frame(candidates_tab)
        accept_controls.grid(row=2, column=0, sticky="ew")
        for col in range(3):
            accept_controls.columnconfigure(col, weight=1)
        self._button(accept_controls, "接受選取", self.accept_candidates, 0, 0)
        self._button(accept_controls, "拒絕選取", self.reject_candidates, 0, 1)
        self._button(accept_controls, "拒絕全部", self.clear_candidates, 0, 2)
        self.notebook.add(candidates_tab, text="待確認候選")
        tools = ttk.Frame(self.notebook, padding=4)
        for col in range(2):
            tools.columnconfigure(col, weight=1)
        self._button(tools, "選擇 YOLO 權重", self.choose_model, 0, 0)
        self._button(tools, "模型預標", self.predict_current, 0, 1)
        self._button(tools, "上一影格傳播", self.propagate_previous, 1, 0)
        self._button(tools, "設定類別映射", self.choose_model_mapping, 1, 1)
        ttk.Label(tools, text="模型信心門檻").grid(row=2, column=0, sticky="w")
        ttk.Spinbox(tools, from_=0.01, to=1.0, increment=0.05, textvariable=self.confidence_var, width=8).grid(
            row=2, column=1, sticky="ew"
        )
        ttk.Checkbutton(
            tools, text="局部放大鏡 + 十字準星", variable=self.lens_enabled, command=self.on_lens_changed
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        self._button(tools, "清除目前類別", self.clear_current_class, 4, 0)
        self._button(tools, "清除整張圖片", self.clear_boxes, 4, 1)
        ttk.Label(
            tools, text="候選框須人工接受；接受後仍須明確確認審閱。\n影格傳播只允許同一影片序列。", wraplength=350
        ).grid(row=5, column=0, columnspan=2, sticky="w")
        self.notebook.add(tools, text="輔助工具")
        self.primary_controls = ttk.Frame(side)
        self.primary_controls.grid(row=4, column=0, sticky="ew", pady=(6, 0))
        for col in range(2):
            self.primary_controls.columnconfigure(col, weight=1)
        self.confirm_button = ttk.Button(self.primary_controls, text="確認本類 + 下一張", command=self.next_image)
        self.confirm_button.grid(row=0, column=0, columnspan=2, sticky="ew", padx=3, pady=3)
        self._button(self.primary_controls, "確認本類無物件", self.reviewed_and_next, 1, 0)
        self._button(self.primary_controls, "略過／下一張", self.next_image_only, 1, 1)
        self._button(self.primary_controls, "上一張", self.prev_image, 2, 0)
        self._button(self.primary_controls, "儲存（不確認）", self.save_current, 2, 1)
        self._button(self.primary_controls, "復原 Ctrl+Z", self.undo_last, 3, 0)
        self._button(self.primary_controls, "重做 Ctrl+Y", self.redo_last, 3, 1)
        self._button(self.primary_controls, "刪除選取框", self.delete_selected, 4, 0)
        self._button(self.primary_controls, "符合視窗 F", self.fit_to_view, 4, 1)
        self.status = ttk.Label(self.root, text="", style="Status.TLabel", anchor="w")
        self.status.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.refresh_class_list()
        self.on_mode_changed()

    def _button(self, parent: ttk.Frame, label: str, command, row: int, col: int) -> None:
        ttk.Button(parent, text=label, command=command).grid(row=row, column=col, sticky="ew", padx=3, pady=3)

    def _bind_events(self) -> None:
        self.class_list.bind("<<ListboxSelect>>", self.on_class_select)
        self.box_tree.bind("<<TreeviewSelect>>", self.on_box_select)
        self.candidate_tree.bind("<<TreeviewSelect>>", lambda _event: self.render())
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<ButtonPress-3>", self.on_right_down)
        self.canvas.bind("<B3-Motion>", self.on_right_drag)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", self.on_hover_leave)
        self.canvas.bind("<Double-Button-1>", lambda _event: self.focus_selected_box())
        self.root.bind("<Key>", self.on_key)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def run(self) -> None:
        self.root.mainloop()

    def current_label_path(self) -> Path:
        assert self.image_path is not None
        return label_path_for(self.image_path, self.labels_dir)

    def current_image_name(self) -> str:
        assert self.image_path is not None
        return self.image_path.name

    def is_reviewed(self, image_name: str, cls: int) -> bool:
        return self.store.status(image_name, cls) in {"positive", "negative"}

    def reviewed_count(self, cls: int) -> int:
        return sum(self.is_reviewed(image.name, cls) for image in self.images)

    def _editable(self) -> bool:
        return self.image is not None and not self._loading and not self._label_error

    def reload_image(self) -> None:
        if (self.dirty or self.candidates) and not messagebox.askyesno(
            "放棄未儲存修改", "重新載入會放棄尚未儲存的修改與候選框。已寫入磁碟的標註不會刪除。繼續？", parent=self.root
        ):
            return
        try:
            store = ReviewStore(self.pass_state_path, self.images, self.class_names, labels_dir=self.labels_dir)
        except (OSError, LabelError) as exc:
            messagebox.showerror("無法重新載入審閱狀態", str(exc), parent=self.root)
            return
        self.store = store
        self._notices.extend(notice for notice in store.notices if notice not in self._notices)
        self.dirty = False
        self.candidates.clear()
        self.load_image(self.index)

    @staticmethod
    def _label_signature(path: Path):
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None

    def load_image(self, index: int) -> None:
        if self.dirty and not self.save_current(silent=True):
            return
        if self.candidates and not messagebox.askyesno(
            "未確認候選框", "離開會丟棄尚未接受的候選框。繼續？", parent=self.root
        ):
            return
        if self.image is not None:
            self._save_session()
        self._load_token += 1
        token = self._load_token
        self.index = int(clamp(index, 0, len(self.images) - 1))
        self.image_path = self.images[self.index]
        self.image = None
        self.boxes = []
        self._saved_boxes = []
        self.candidates.clear()
        self.selected_box = None
        self.dirty = False
        self._loading = True
        self._label_error = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.invalidate_render_cache()
        self._image_item = None
        self.canvas.delete("all")
        self.canvas.create_text(20, 20, text="載入圖片…", fill="white", anchor="nw")
        self.reset_drag_state()
        self.refresh_box_tree()
        self.refresh_candidates()
        self.update_status()
        requested = time.perf_counter()

        def loaded(path, image, error):
            if self._closing or token != self._load_token:
                return
            self._loading = False
            if error:
                self._label_error = True
                self.canvas.delete("all")
                self.canvas.create_text(20, 20, text=f"圖片載入失敗：{error}", fill="#ff7777", anchor="nw", width=600)
                self.update_status(str(error))
                return
            self.image = image
            label = self.current_label_path()
            try:
                signature = self._label_signature(label)
                self.boxes = (
                    read_yolo_boxes(label, image.width, image.height, len(self.class_names)) if label.exists() else []
                )
                if self._label_signature(label) != signature:
                    raise LabelError(label, "Label changed while loading; reload before editing.")
                self._saved_label_signature = signature
            except (OSError, LabelError) as exc:
                self._label_error = True
                self._notices.append(str(exc))
                messagebox.showerror(
                    "標籤錯誤：禁止覆寫", str(exc) + "\n修正原始檔後，使用資料集 → 重新載入。", parent=self.root
                )
            self._saved_boxes = copy_boxes(self.boxes)
            self.refresh_box_tree()
            self.fit_to_view()
            restored = self._restore_view
            self._restore_view = {}
            if restored.get("image") == path.name and isinstance(restored.get("view"), dict):
                view = restored["view"]
                if all(
                    type(view.get(key)) in (int, float) and math.isfinite(view[key])
                    for key in ("zoom", "center_x", "center_y")
                ):
                    self.zoom = clamp(view["zoom"], 0.2, 20.0)
                    self.offset_x = self.canvas.winfo_width() / 2 - view["center_x"] * self.display_scale
                    self.offset_y = self.canvas.winfo_height() / 2 - view["center_y"] * self.display_scale
                    self.render()

            def painted():
                if token == self._load_token and not self._closing:
                    self._load_timings.append((time.perf_counter() - requested) * 1000)

            self.root.after_idle(painted)
            neighbors = self.images[max(0, self.index - 2) : self.index] + self.images[self.index + 1 : self.index + 3]
            self.loader.prefetch(neighbors)
            self.update_status()
            self.schedule_session_save()

        self.loader.request(self.image_path, loaded)

    def save_current(self, silent: bool = False) -> bool:
        if not self._editable():
            if not silent:
                self.update_status("圖片未就緒或標籤有錯誤；禁止儲存。")
            return False
        try:
            if self._label_signature(self.current_label_path()) != self._saved_label_signature:
                raise LabelError(
                    self.current_label_path(), "Label was changed by another editor; reload before saving."
                )
            if self.dirty:
                changed = []
                for cls in range(len(self.class_names)):
                    before = sorted((box.x1, box.y1, box.x2, box.y2) for box in self._saved_boxes if box.cls == cls)
                    after = sorted((box.x1, box.y1, box.x2, box.y2) for box in self.boxes if box.cls == cls)
                    if before != after:
                        changed.append(cls)
                # Invalidate confirmation first: a failed label write must never leave
                # a previous confirmation attached to newly edited annotations.
                if changed:
                    self.store.mark_edited(self.current_image_name(), changed)
                write_yolo_boxes(
                    self.current_label_path(),
                    self.boxes,
                    self.image.width,
                    self.image.height,
                    len(self.class_names),
                    expected_signature=self._saved_label_signature,
                )
                self._saved_boxes = copy_boxes(self.boxes)
                self._saved_label_signature = self._label_signature(self.current_label_path())
                self.dirty = False
            elif not silent and not self.current_label_path().exists():
                write_yolo_boxes(
                    self.current_label_path(),
                    self.boxes,
                    self.image.width,
                    self.image.height,
                    len(self.class_names),
                    expected_signature=self._saved_label_signature,
                )
                self._saved_label_signature = self._label_signature(self.current_label_path())
        except (OSError, LabelError) as exc:
            messagebox.showerror("儲存失敗：保留未儲存修改", str(exc), parent=self.root)
            self.update_status("儲存失敗；不要關閉或切換資料集。")
            return False
        self.refresh_class_list()
        if not silent:
            self.update_status("已儲存；尚未代表已審閱。")
        return True

    def _confirm(self, empty: bool = False) -> bool:
        if not self._editable():
            return False
        classes = list(range(len(self.class_names))) if self.image_mode.get() and not empty else [self.current_class]
        if any(candidate.box.cls in classes for candidate in self.candidates):
            messagebox.showwarning("候選尚未處理", "請先接受或拒絕本次確認範圍內的候選框。", parent=self.root)
            return False
        if empty and any(box.cls == self.current_class for box in self.boxes):
            messagebox.showwarning("已有物件", "目前類別仍有框，不能確認無物件。請先檢查或刪除框。", parent=self.root)
            return False
        if not self.save_current():
            return False
        try:
            for cls in classes:
                self.store.confirm(self.current_image_name(), cls, any(box.cls == cls for box in self.boxes))
        except (OSError, LabelError) as exc:
            messagebox.showerror("審閱狀態未儲存", str(exc), parent=self.root)
            return False
        self.refresh_class_list()
        self.update_status("審閱已明確確認。")
        return True

    def _matches_filter(self, image_name: str, value: str | None = None) -> bool:
        statuses = (
            [self.store.status(image_name, cls) for cls in range(len(self.class_names))]
            if self.image_mode.get()
            else [self.store.status(image_name, self.current_class)]
        )
        value = value or self.filter_var.get()
        if value == "Unreviewed":
            return any(status in {"pending", "edited"} for status in statuses)
        if value == "Edited":
            return "edited" in statuses
        if value == "Confirmed empty":
            return all(status == "negative" for status in statuses)
        if value == "Confirmed":
            return all(status in {"positive", "negative"} for status in statuses)
        return True

    def _navigate(self, direction: int, filter_value: str | None = None) -> None:
        if self._loading:
            return
        for step in range(1, len(self.images)):
            index = (self.index + direction * step) % len(self.images)
            if self._matches_filter(self.images[index].name, filter_value):
                self.load_image(index)
                return
        self.update_status("沒有其他符合篩選的圖片；目前圖片保持不變。")

    def prev_image(self) -> None:
        self._navigate(-1)

    def next_image_only(self) -> None:
        self._navigate(1)

    def next_image(self) -> None:
        if self._confirm():
            self._navigate(1)

    def reviewed_and_next(self) -> None:
        if self._confirm(empty=True):
            self._navigate(1)

    def next_unreviewed_current_class(self) -> None:
        self._navigate(1, "Unreviewed")

    def next_class_pass(self) -> None:
        self._change_pass(1)

    def prev_class_pass(self) -> None:
        self._change_pass(-1)

    def _change_pass(self, delta: int) -> None:
        if self._loading or (self.dirty and not self.save_current(silent=True)):
            return
        self.select_class((self.current_class + delta) % len(self.class_names))
        if not self.image_mode.get():
            self.load_image(self.first_unreviewed_index(self.current_class))

    def first_unreviewed_index(self, cls: int) -> int:
        return next((idx for idx, image in enumerate(self.images) if not self.is_reviewed(image.name, cls)), self.index)

    def on_mode_changed(self) -> None:
        self.confirm_button.configure(text="確認整張 + 下一張" if self.image_mode.get() else "確認本類 + 下一張")
        self.refresh_box_tree()
        self.render()
        self.update_status()
        self.schedule_session_save()

    def on_filter_changed(self, _event=None) -> None:
        self.schedule_session_save()
        if self.image_path and not self._matches_filter(self.current_image_name()):
            self._navigate(1)
        else:
            self.update_status()

    def jump_to_image(self) -> None:
        value = self.jump_var.get().strip()
        names = [image.name for image in self.images]
        if value in names:
            self.load_image(names.index(value))
        elif value.isdigit() and 1 <= int(value) <= len(self.images):
            self.load_image(int(value) - 1)
        else:
            messagebox.showwarning("圖片跳轉", f"輸入 1–{len(self.images)} 或完整圖片檔名。", parent=self.root)

    def fit_to_view(self) -> None:
        if self.image is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if cw < 30 or ch < 30:
            self.root.after(50, self.fit_to_view)
            return
        self.fit_scale = min((cw - 20) / self.image.width, (ch - 20) / self.image.height)
        self.fit_scale = max(self.fit_scale, 0.01)
        self.zoom = 1.0
        scale = self.display_scale
        self.offset_x = (cw - self.image.width * scale) / 2.0
        self.offset_y = (ch - self.image.height * scale) / 2.0
        self.render()

    @property
    def display_scale(self) -> float:
        return self.fit_scale * self.zoom

    def on_canvas_configure(self, _event) -> None:
        self.render()

    def on_class_select(self, _event=None) -> None:
        if self.refreshing_classes:
            return
        selection = self.class_list.curselection()
        if selection:
            self.select_class(int(selection[0]))

    def on_show_other_classes(self) -> None:
        self.refresh_box_tree()
        self.render()
        self.update_status()

    def on_box_select(self, _event=None) -> None:
        selection = self.box_tree.selection()
        if not selection:
            return
        try:
            self.selected_box = int(selection[0])
        except ValueError:
            return
        if 0 <= self.selected_box < len(self.boxes):
            box = self.boxes[self.selected_box]
            self.current_class = box.cls
            self.class_list.selection_clear(0, tk.END)
            self.class_list.selection_set(box.cls)
            self.class_list.see(box.cls)
        self.render()
        self.update_status()

    def on_left_down(self, event) -> None:
        if not self._editable():
            return
        self.canvas.focus_set()
        self.drag_start = (event.x, event.y)
        self.drag_current = (event.x, event.y)
        self.drag_start_image = self.canvas_to_image(event.x, event.y)
        self.drag_original_snapshot = self.snapshot_state()

        hit_handle = self.hit_resize_handle(event.x, event.y)
        if hit_handle is not None:
            idx, handle = hit_handle
            self.drag_mode = "resize"
            self.drag_box_index = idx
            self.resize_handle = handle
            self.selected_box = idx
            self.drag_original_box = Box(
                self.boxes[idx].cls, self.boxes[idx].x1, self.boxes[idx].y1, self.boxes[idx].x2, self.boxes[idx].y2
            )
            self.select_tree_item(idx)
            self.render()
            return

        hit_body = self.hit_box_body(event.x, event.y)
        if hit_body is not None:
            self.drag_mode = "move"
            self.drag_box_index = hit_body
            self.selected_box = hit_body
            self.drag_original_box = Box(
                self.boxes[hit_body].cls,
                self.boxes[hit_body].x1,
                self.boxes[hit_body].y1,
                self.boxes[hit_body].x2,
                self.boxes[hit_body].y2,
            )
            self.select_tree_item(hit_body)
            self.render()
            return

        self.drag_mode = "draw"

    def on_left_drag(self, event) -> None:
        if self.drag_start is None:
            return
        self.drag_current = (event.x, event.y)
        if self.drag_mode == "move":
            self.apply_move_drag(event.x, event.y)
        elif self.drag_mode == "resize":
            self.apply_resize_drag(event.x, event.y)
        self.schedule_high_quality_render()
        self.render()

    def on_left_up(self, event) -> None:
        if self.image is None or self.drag_start is None:
            return
        sx, sy = self.drag_start
        moved = abs(event.x - sx) + abs(event.y - sy)
        if self.drag_mode == "move" and self.drag_box_index is not None and self.drag_original_box is not None:
            self.apply_move_drag(event.x, event.y)
            changed = self.box_changed(self.drag_original_box, self.boxes[self.drag_box_index])
            if changed and self.drag_original_snapshot is not None:
                self.push_undo_state(self.drag_original_snapshot)
                self.dirty = True
                self.save_current(silent=True)
            self.refresh_box_tree()
            self.render()
            self.reset_drag_state()
            self.update_status("moved" if changed else None)
            return
        if self.drag_mode == "resize" and self.drag_box_index is not None and self.drag_original_box is not None:
            self.apply_resize_drag(event.x, event.y)
            changed = self.box_changed(self.drag_original_box, self.boxes[self.drag_box_index])
            if changed and self.drag_original_snapshot is not None:
                self.push_undo_state(self.drag_original_snapshot)
                self.dirty = True
                self.save_current(silent=True)
            self.refresh_box_tree()
            self.render()
            self.reset_drag_state()
            self.update_status("resized" if changed else None)
            return
        if moved < 5:
            self.select_box_at(event.x, event.y)
        else:
            x1, y1 = self.canvas_to_image(sx, sy)
            x2, y2 = self.canvas_to_image(event.x, event.y)
            box = Box(self.current_class, x1, y1, x2, y2).clamp(self.image.width, self.image.height)
            if box.width >= MIN_BOX_SIZE_PX and box.height >= MIN_BOX_SIZE_PX:
                if self.drag_original_snapshot is not None:
                    self.push_undo_state(self.drag_original_snapshot)
                self.boxes.append(box)
                self.selected_box = len(self.boxes) - 1
                self.dirty = True
                self.refresh_box_tree()
                self.select_tree_item(self.selected_box)
                self.save_current(silent=True)
            else:
                self.update_status("box too small")
        self.drag_start = None
        self.drag_current = None
        self.reset_drag_state()
        self.render()

    def on_right_down(self, event) -> None:
        self.canvas.focus_set()
        self.pan_start = (event.x, event.y)
        self.pan_offset_start = (self.offset_x, self.offset_y)

    def on_right_drag(self, event) -> None:
        if self.pan_start is None or self.pan_offset_start is None:
            return
        sx, sy = self.pan_start
        ox, oy = self.pan_offset_start
        self.offset_x = ox + event.x - sx
        self.offset_y = oy + event.y - sy
        self.schedule_high_quality_render()
        self.render()

    def on_mousewheel(self, event) -> None:
        if getattr(event, "num", None) == 5 or getattr(event, "delta", 0) < 0:
            factor = 0.9
        else:
            factor = 1.1
        self.zoom_at(event.x, event.y, factor)

    def on_key(self, event) -> str | None:
        if self._loading or event.widget.winfo_class() in {"Entry", "TEntry", "Text", "TSpinbox", "TCombobox"}:
            return None
        char = (event.char or "").lower()
        key = event.keysym
        if key in {"Left", "KP_Left"}:
            self.prev_image()
            return "break"
        if key in {"Right", "KP_Right"}:
            self.next_image_only()
            return "break"
        if key.lower() == "z" and (event.state & 0x4):
            self.restore_previous_state()
            return "break"
        if key.lower() == "y" and (event.state & 0x4):
            self.redo_last()
            return "break"
        if char == "s":
            self.save_current()
            return "break"
        if char == "n":
            self.next_image()
            return "break"
        if char == "m":
            self.reviewed_and_next()
            return "break"
        if char == "p":
            self.prev_image()
            return "break"
        if char == "j":
            self.next_class_pass()
            return "break"
        if char == "k":
            self.prev_class_pass()
            return "break"
        if char == "u":
            self.restore_previous_state()
            return "break"
        if char == "f":
            self.fit_to_view()
            return "break"
        if char == "[":
            self.select_class_delta(-1)
            return "break"
        if char == "]":
            self.select_class_delta(1)
            return "break"
        if key in {"Delete", "BackSpace"}:
            self.delete_selected()
            return "break"
        if key in {"plus", "equal"}:
            self.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, 1.1)
            return "break"
        if key in {"minus", "underscore"}:
            self.zoom_at(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2, 0.9)
            return "break"
        if char.isdigit():
            cls = int(char)
            if cls < len(self.class_names):
                self.select_class(cls)
                return "break"
        return None

    def select_class_delta(self, delta: int) -> None:
        self.select_class((self.current_class + delta) % len(self.class_names))

    def select_class(self, cls: int) -> None:
        self.current_class = int(clamp(cls, 0, len(self.class_names) - 1))
        self.class_list.selection_clear(0, tk.END)
        self.class_list.selection_set(self.current_class)
        self.class_list.see(self.current_class)
        self.selected_box = None
        self.refresh_class_list()
        self.refresh_box_tree()
        self.render()
        self.update_status()
        self.schedule_session_save()

    def zoom_at(self, cx: float, cy: float, factor: float) -> None:
        if self.image is None:
            return
        old_scale = self.display_scale
        if old_scale <= 0.0:
            return
        image_x = (cx - self.offset_x) / old_scale
        image_y = (cy - self.offset_y) / old_scale
        self.zoom = clamp(self.zoom * factor, 0.2, 20.0)
        new_scale = self.display_scale
        self.offset_x = cx - image_x * new_scale
        self.offset_y = cy - image_y * new_scale
        self.schedule_high_quality_render()
        self.render()
        self.update_status()
        self.schedule_session_save()

    def reset_drag_state(self) -> None:
        self.drag_start = None
        self.drag_current = None
        self.drag_mode = None
        self.drag_box_index = None
        self.drag_start_image = None
        self.drag_original_box = None
        self.drag_original_snapshot = None
        self.resize_handle = None

    def invalidate_render_cache(self) -> None:
        self.render_cache_key = None
        self.render_cache_photo = None
        self.photo = None

    def snapshot_state(self) -> tuple[list[Box], int | None]:
        return copy_boxes(self.boxes), self.selected_box

    def push_undo_state(self, snapshot: tuple[list[Box], int | None]) -> None:
        self.undo_stack.append(snapshot)
        self.redo_stack.clear()
        if len(self.undo_stack) > MAX_UNDO_STATES:
            self.undo_stack.pop(0)

    def _restore_snapshot(self, snapshot) -> None:
        boxes, selected = snapshot
        self.boxes = copy_boxes(boxes)
        self.selected_box = selected if selected is not None and 0 <= selected < len(self.boxes) else None
        self.dirty = True
        self.refresh_box_tree()
        if self.selected_box is not None:
            self.select_tree_item(self.selected_box)
        self.save_current(silent=True)
        self.render()

    def restore_previous_state(self) -> None:
        if not self._editable() or not self.undo_stack:
            self.update_status("沒有可復原操作。")
            return
        self.redo_stack.append(self.snapshot_state())
        self._restore_snapshot(self.undo_stack.pop())
        self.update_status("已復原；需重新確認審閱。")

    def redo_last(self) -> None:
        if not self._editable() or not self.redo_stack:
            self.update_status("沒有可重做操作。")
            return
        self.undo_stack.append(self.snapshot_state())
        self._restore_snapshot(self.redo_stack.pop())
        self.update_status("已重做；需重新確認審閱。")

    def _shows_class(self, cls: int) -> bool:
        return self.image_mode.get() or self.show_other_classes.get() or cls == self.current_class

    def visible_box_indices(self) -> list[int]:
        visible = []
        for idx, box in enumerate(self.boxes):
            if self.image is None:
                continue
            clean = box.clamp(self.image.width, self.image.height)
            if self._shows_class(clean.cls):
                visible.append(idx)
        return visible

    def box_handle_points(self, box: Box) -> dict[str, tuple[float, float]]:
        x1, y1 = self.image_to_canvas(box.x1, box.y1)
        x2, y2 = self.image_to_canvas(box.x2, box.y2)
        xm = (x1 + x2) / 2.0
        ym = (y1 + y2) / 2.0
        return {
            "nw": (x1, y1),
            "n": (xm, y1),
            "ne": (x2, y1),
            "e": (x2, ym),
            "se": (x2, y2),
            "s": (xm, y2),
            "sw": (x1, y2),
            "w": (x1, ym),
        }

    def hit_resize_handle(self, cx: float, cy: float) -> tuple[int, str] | None:
        if self.image is None:
            return None
        candidates = [self.selected_box] if self.selected_box is not None else []
        candidates += [idx for idx in reversed(self.visible_box_indices()) if idx != self.selected_box]
        for idx in candidates:
            if idx is None or not (0 <= idx < len(self.boxes)):
                continue
            box = self.boxes[idx].clamp(self.image.width, self.image.height)
            for handle, (hx, hy) in self.box_handle_points(box).items():
                if abs(cx - hx) <= HANDLE_RADIUS_PX and abs(cy - hy) <= HANDLE_RADIUS_PX:
                    return idx, handle
        return None

    def hit_box_body(self, cx: float, cy: float) -> int | None:
        if self.image is None:
            return None
        x, y = self.canvas_to_image(cx, cy)
        for idx in reversed(self.visible_box_indices()):
            box = self.boxes[idx].clamp(self.image.width, self.image.height)
            if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
                return idx
        return None

    def apply_move_drag(self, event_x: float, event_y: float) -> None:
        if (
            self.image is None
            or self.drag_box_index is None
            or self.drag_start_image is None
            or self.drag_original_box is None
        ):
            return
        ix, iy = self.canvas_to_image(event_x, event_y)
        sx, sy = self.drag_start_image
        dx = ix - sx
        dy = iy - sy
        original = self.drag_original_box
        width = original.width
        height = original.height
        x1 = clamp(original.x1 + dx, 0.0, max(0.0, self.image.width - width))
        y1 = clamp(original.y1 + dy, 0.0, max(0.0, self.image.height - height))
        self.boxes[self.drag_box_index] = Box(original.cls, x1, y1, x1 + width, y1 + height).clamp(
            self.image.width, self.image.height
        )

    def apply_resize_drag(self, event_x: float, event_y: float) -> None:
        if (
            self.image is None
            or self.drag_box_index is None
            or self.drag_original_box is None
            or self.resize_handle is None
        ):
            return
        ix, iy = self.canvas_to_image(event_x, event_y)
        original = self.drag_original_box
        x1, y1, x2, y2 = original.x1, original.y1, original.x2, original.y2
        handle = self.resize_handle
        if "w" in handle:
            x1 = clamp(ix, 0.0, x2 - MIN_BOX_SIZE_PX)
        if "e" in handle:
            x2 = clamp(ix, x1 + MIN_BOX_SIZE_PX, self.image.width)
        if "n" in handle:
            y1 = clamp(iy, 0.0, y2 - MIN_BOX_SIZE_PX)
        if "s" in handle:
            y2 = clamp(iy, y1 + MIN_BOX_SIZE_PX, self.image.height)
        self.boxes[self.drag_box_index] = Box(original.cls, x1, y1, x2, y2).clamp(self.image.width, self.image.height)

    def box_changed(self, before: Box, after: Box) -> bool:
        return any(
            abs(a - b) > 0.5
            for a, b in [(before.x1, after.x1), (before.y1, after.y1), (before.x2, after.x2), (before.y2, after.y2)]
        )

    def select_box_at(self, cx: float, cy: float) -> None:
        x, y = self.canvas_to_image(cx, cy)
        selected = None
        for idx in reversed(range(len(self.boxes))):
            box = self.boxes[idx].clamp(self.image.width, self.image.height)  # type: ignore[union-attr]
            if not self._shows_class(box.cls):
                continue
            if box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2:
                selected = idx
                break
        self.selected_box = selected
        if selected is not None:
            self.select_tree_item(selected)
        else:
            self.box_tree.selection_remove(self.box_tree.selection())
        self.render()
        self.update_status()

    def undo_last(self) -> None:
        self.restore_previous_state()

    def delete_selected(self) -> None:
        if not self._editable():
            return
        if self.selected_box is None or not (0 <= self.selected_box < len(self.boxes)):
            self.update_status("no selected box")
            return
        self.push_undo_state(self.snapshot_state())
        del self.boxes[self.selected_box]
        self.selected_box = None
        self.dirty = True
        self.refresh_box_tree()
        self.save_current(silent=True)
        self.render()

    def clear_current_class(self) -> None:
        if not self._editable():
            return
        current_count = sum(1 for box in self.boxes if box.cls == self.current_class)
        if current_count == 0:
            return
        class_name = self.class_names[self.current_class]
        if not messagebox.askyesno(
            "Clear current class", f"Clear {current_count} boxes for '{class_name}' on this image?"
        ):
            return
        self.push_undo_state(self.snapshot_state())
        self.boxes = [box for box in self.boxes if box.cls != self.current_class]
        self.selected_box = None
        self.dirty = True
        self.refresh_box_tree()
        self.save_current(silent=True)
        self.render()

    def clear_boxes(self) -> None:
        if not self._editable():
            return
        if not self.boxes:
            return
        if not messagebox.askyesno("Clear boxes", "Clear all boxes on this image?"):
            return
        self.push_undo_state(self.snapshot_state())
        self.boxes.clear()
        self.selected_box = None
        self.dirty = True
        self.refresh_box_tree()
        self.save_current(silent=True)
        self.render()

    def image_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        scale = self.display_scale
        return self.offset_x + x * scale, self.offset_y + y * scale

    def canvas_to_image(self, x: float, y: float) -> tuple[float, float]:
        scale = max(self.display_scale, 1e-9)
        return (x - self.offset_x) / scale, (y - self.offset_y) / scale

    def refresh_class_list(self) -> None:
        if not hasattr(self, "class_list"):
            return
        self.refreshing_classes = True
        self.class_list.delete(0, tk.END)
        total = len(self.images)
        for idx, name in enumerate(self.class_names):
            done = self.reviewed_count(idx)
            zh = CLASS_ZH.get(name, "")
            self.class_list.insert(tk.END, f"{idx:02d}  {zh} | {name}  [{done}/{total}]")
            if idx == self.current_class:
                self.class_list.itemconfig(idx, background="#1f4e79", foreground="#ffffff")
        self.class_list.selection_clear(0, tk.END)
        self.class_list.selection_set(self.current_class)
        self.class_list.see(self.current_class)
        self.refreshing_classes = False
        self.update_pass_labels()

    def update_pass_labels(self) -> None:
        if not hasattr(self, "pass_label"):
            return
        name = self.class_names[self.current_class]
        done = self.reviewed_count(self.current_class)
        self.pass_label.configure(text=f"{self.current_class:02d}  {CLASS_ZH.get(name, '')}\n{name}")
        status = (
            STATUS_TEXT[self.store.status(self.current_image_name(), self.current_class)]
            if self.image_path
            else "未載入"
        )
        current_count = sum(box.cls == self.current_class for box in self.boxes)
        self.progress_label.configure(
            text=f"本類已確認 {done}/{len(self.images)} · {status}\n本類 {current_count} 框 · 候選 {len(self.candidates)} 框"
        )

    def refresh_box_tree(self) -> None:
        self.box_tree.delete(*self.box_tree.get_children())
        if self.image is None:
            return
        for idx, box in enumerate(self.boxes):
            clean = box.clamp(self.image.width, self.image.height)
            if not self._shows_class(clean.cls):
                continue
            name = self.class_label(clean.cls)
            xywh = f"{clean.x1:.0f} {clean.y1:.0f} {clean.width:.0f} {clean.height:.0f}"
            self.box_tree.insert("", tk.END, iid=str(idx), values=(f"{clean.cls:02d} {name}", xywh))

    def select_tree_item(self, idx: int) -> None:
        iid = str(idx)
        if self.box_tree.exists(iid):
            self.box_tree.selection_set(iid)
            self.box_tree.see(iid)

    def schedule_high_quality_render(self) -> None:
        if hasattr(self, "_hq_render_id") and self._hq_render_id is not None:
            try:
                self.root.after_cancel(self._hq_render_id)
            except tk.TclError:
                pass

        def hq_callback():
            self.is_interacting = False
            self._hq_render_id = None
            self.render()

        self.is_interacting = True
        self._hq_render_id = self.root.after(200, hq_callback)

    def render(self) -> None:
        if not self._closing and self._render_id is None:
            self._render_id = self.root.after(16 if self.is_interacting else 0, self._render_now)

    def _render_now(self) -> None:
        self._render_id = None
        if self.image is None or self._closing:
            return
        started = time.perf_counter()

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        scale = self.display_scale
        disp_w = max(1, int(round(self.image.width * scale)))
        disp_h = max(1, int(round(self.image.height * scale)))

        cx1 = max(0.0, self.offset_x)
        cy1 = max(0.0, self.offset_y)
        cx2 = min(float(cw), self.offset_x + disp_w)
        cy2 = min(float(ch), self.offset_y + disp_h)

        if cx2 <= cx1 or cy2 <= cy1:
            self.canvas.delete("all")
            self._image_item = None
            return

        sx1 = cx1 - self.offset_x
        sy1 = cy1 - self.offset_y
        sx2 = cx2 - self.offset_x
        sy2 = cy2 - self.offset_y

        orig_x1 = max(0.0, min(self.image.width, sx1 / scale))
        orig_y1 = max(0.0, min(self.image.height, sy1 / scale))
        orig_x2 = max(0.0, min(self.image.width, sx2 / scale))
        orig_y2 = max(0.0, min(self.image.height, sy2 / scale))

        crop_x1 = int(math.floor(orig_x1))
        crop_y1 = int(math.floor(orig_y1))
        crop_x2 = int(math.ceil(orig_x2))
        crop_y2 = int(math.ceil(orig_y2))

        crop_x1 = max(0, min(self.image.width - 1, crop_x1))
        crop_y1 = max(0, min(self.image.height - 1, crop_y1))
        crop_x2 = max(crop_x1 + 1, min(self.image.width, crop_x2))
        crop_y2 = max(crop_y1 + 1, min(self.image.height, crop_y2))

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1

        target_w = max(1, int(round(crop_w * scale)))
        target_h = max(1, int(round(crop_h * scale)))

        if getattr(self, "is_interacting", False):
            resample = Image.Resampling.NEAREST
        else:
            resample = Image.Resampling.BILINEAR if scale < 1.0 else Image.Resampling.LANCZOS

        cache_key = (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
            scale,
            target_w,
            target_h,
            getattr(self, "is_interacting", False),
        )
        if self.render_cache_key != cache_key or self.render_cache_photo is None:
            cropped = self.image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            view = cropped.resize((target_w, target_h), resample)
            self.render_cache_photo = ImageTk.PhotoImage(view)
            self.render_cache_key = cache_key
        self.photo = self.render_cache_photo

        pos_x = self.offset_x + crop_x1 * scale
        pos_y = self.offset_y + crop_y1 * scale

        if self._image_item is None:
            self.canvas.delete("all")
            self._image_item = self.canvas.create_image(pos_x, pos_y, image=self.photo, anchor="nw")
        else:
            self.canvas.coords(self._image_item, pos_x, pos_y)
            self.canvas.itemconfigure(self._image_item, image=self.photo)
        self.canvas.delete("overlay")
        self.canvas.tag_lower(self._image_item)
        self.draw_boxes()
        self.draw_selection_handles()
        self.draw_drag_box()
        self.draw_candidates()
        self.draw_lens()
        self._render_timings.append((time.perf_counter() - started) * 1000)

    def class_label(self, cls: int) -> str:
        if 0 <= cls < len(self.class_names):
            return self.class_names[cls]
        return "unknown"

    def draw_boxes(self) -> None:
        if self.image is None:
            return
        for idx, box in enumerate(self.boxes):
            clean = box.clamp(self.image.width, self.image.height)
            is_current = clean.cls == self.current_class
            if not self._shows_class(clean.cls):
                continue
            color = CLASS_COLORS[clean.cls % len(CLASS_COLORS)] if is_current or self.image_mode.get() else "#8a8f98"
            width = 3 if idx == self.selected_box else 2
            x1, y1 = self.image_to_canvas(clean.x1, clean.y1)
            x2, y2 = self.image_to_canvas(clean.x2, clean.y2)
            if x2 < 0 or y2 < 0 or x1 > self.canvas.winfo_width() or y1 > self.canvas.winfo_height():
                continue
            dash = () if is_current or self.image_mode.get() else (4, 3)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, dash=dash, tags="overlay")
            label = f"{clean.cls:02d} {self.class_label(clean.cls)}" if idx == self.selected_box else f"{clean.cls:02d}"
            text_id = self.canvas.create_text(
                x1 + 3,
                y1 + 3,
                text=label,
                anchor="nw",
                fill="#ffffff",
                font=(self.font_family, 9, "bold"),
                tags="overlay",
            )
            bbox = self.canvas.bbox(text_id)
            if bbox:
                bg_id = self.canvas.create_rectangle(
                    bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1, fill=color, outline=color, tags="overlay"
                )
                self.canvas.tag_lower(bg_id, text_id)

    def draw_drag_box(self) -> None:
        if self.drag_mode != "draw" or self.drag_start is None or self.drag_current is None:
            return
        sx, sy = self.drag_start
        cx, cy = self.drag_current
        color = CLASS_COLORS[self.current_class % len(CLASS_COLORS)]
        self.canvas.create_rectangle(sx, sy, cx, cy, outline=color, width=2, dash=(4, 2), tags="overlay")

    def draw_selection_handles(self) -> None:
        if self.image is None or self.selected_box is None or not (0 <= self.selected_box < len(self.boxes)):
            return
        box = self.boxes[self.selected_box].clamp(self.image.width, self.image.height)
        if not self._shows_class(box.cls):
            return
        color = CLASS_COLORS[box.cls % len(CLASS_COLORS)]
        for hx, hy in self.box_handle_points(box).values():
            self.canvas.create_rectangle(
                hx - HANDLE_RADIUS_PX,
                hy - HANDLE_RADIUS_PX,
                hx + HANDLE_RADIUS_PX,
                hy + HANDLE_RADIUS_PX,
                fill="#ffffff",
                outline=color,
                width=2,
                tags="overlay",
            )

    def update_status(self, extra: str | None = None) -> None:
        if not hasattr(self, "status"):
            return
        self.update_pass_labels()
        name = self.image_path.name if self.image_path else "未載入"
        state = (
            "載入中"
            if self._loading
            else "未儲存"
            if self.dirty
            else "唯讀：標籤錯誤"
            if self._label_error
            else "已儲存"
            if self.image and self.current_label_path().exists()
            else "尚無標籤檔"
        )
        resolution = f"{self.image.width}×{self.image.height}" if self.image else ""
        msg = f"{self.index + 1}/{len(self.images)}  {name}  {resolution}  {state}  zoom={self.zoom:.2f}x"
        if self._notices:
            msg += f"  · {len(self._notices)} 則資料集通知（資料集選單）"
        if extra:
            msg += f"  |  {extra}"
        self.status.configure(text=msg)

    def _drain_background(self) -> None:
        if self._closing:
            return
        self.loader.drain()
        remaining = []
        for future, token, callback in self._jobs:
            if not future.done():
                remaining.append((future, token, callback))
                continue
            if token is not None and token != self._load_token:
                continue
            try:
                callback(future.result())
            except Exception as exc:
                messagebox.showerror("背景工作失敗", str(exc), parent=self.root)
                self.update_status(str(exc))
        self._jobs = remaining
        self.root.after(20, self._drain_background)

    def _submit_job(self, operation, callback, *, bound_to_image: bool = True) -> None:
        if self._jobs:
            self.update_status("背景工作尚未完成，請稍候。")
            return
        self._jobs.append((self._executor.submit(operation), self._load_token if bound_to_image else None, callback))
        self.update_status("背景工作執行中；圖片操作不會被運算阻塞。")

    def schedule_session_save(self) -> None:
        if self._closing:
            return
        if self._session_id is not None:
            self.root.after_cancel(self._session_id)

        def save():
            self._session_id = None
            self._save_session()

        self._session_id = self.root.after(500, save)

    def _save_session(self) -> bool:
        if self.image_path is None or self.image is None:
            return True
        center = self.canvas_to_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2)
        try:
            self.session.save(
                image=self.current_image_name(),
                class_id=self.current_class,
                image_mode=self.image_mode.get(),
                filter=self.filter_var.get(),
                lens=self.lens_enabled.get(),
                geometry=self.root.geometry(),
                view={"zoom": self.zoom, "center_x": center[0], "center_y": center[1]},
            )
        except OSError as exc:
            self.update_status(f"導覽狀態儲存失敗：{exc}")
            return False
        return True

    def _switch_dataset(self, config: DatasetConfig) -> None:
        if self._jobs:
            messagebox.showwarning("背景工作尚未完成", "請等目前背景工作完成再切換資料集。", parent=self.root)
            return
        if self.dirty and not self.save_current(silent=True):
            return
        if self.candidates and not messagebox.askyesno(
            "未確認候選框", "切換資料集會丟棄候選框。繼續？", parent=self.root
        ):
            return
        try:
            classes = read_classes(config.classes_file)
            images = collect_images(config.images_dir)
            if not images:
                raise LabelError(config.images_dir, "No images found")
            store = ReviewStore(config.pass_state, images, classes, labels_dir=config.labels_dir)
        except (OSError, LabelError) as exc:
            messagebox.showerror("資料集無法開啟", str(exc), parent=self.root)
            return
        if not self._save_session():
            return
        self.candidates.clear()
        self.image = None
        self.image_path = None
        self.boxes = []
        self._saved_boxes = []
        self._saved_label_signature = None
        self.selected_box = None
        self.config = config
        self.images_dir, self.labels_dir = config.images_dir, config.labels_dir
        self.classes_file, self.pass_state_path = config.classes_file, config.pass_state
        self.class_names, self.images, self.store = classes, images, store
        self.session = Session(config, classes)
        restored = self.session.load()
        self._restore_view = restored
        self.current_class = int(clamp(restored.get("class_id", 0), 0, len(classes) - 1))
        self.image_mode.set(restored.get("image_mode", False))
        self.lens_enabled.set(restored.get("lens", True))
        self.filter_var.set(restored.get("filter", "All") if restored.get("filter", "All") in FILTERS else "All")
        self._notices = list(store.notices) + ([self.session.notice] if self.session.notice else [])
        names = [image.name for image in images]
        index = names.index(restored["image"]) if restored.get("image") in names else 0
        self.loader.close()
        self.loader = ImageLoader(max_bytes=self.args.cache_mb * 1024 * 1024)
        self.refresh_class_list()
        self.on_mode_changed()
        self.load_image(index)
        self.root.title(f"Label Power — {config.images_dir}")

    def open_profile(self) -> None:
        name = filedialog.askopenfilename(
            parent=self.root, title="開啟資料集設定", filetypes=[("Dataset JSON", "*.json")]
        )
        if name:
            try:
                self._switch_dataset(DatasetConfig.from_profile(Path(name).resolve()))
            except (OSError, LabelError) as exc:
                messagebox.showerror("設定檔錯誤", str(exc), parent=self.root)

    def choose_dataset(self) -> None:
        config = choose_dataset_dialog(self.root, self.classes_file)
        if config:
            self._switch_dataset(config)

    def save_profile(self) -> None:
        name = filedialog.asksaveasfilename(
            parent=self.root, title="儲存資料集設定", defaultextension=".json", filetypes=[("Dataset JSON", "*.json")]
        )
        if name:
            try:
                self.config.save_profile(Path(name))
                self.update_status("資料集設定已儲存。")
            except OSError as exc:
                messagebox.showerror("設定檔無法儲存", str(exc), parent=self.root)

    def export_verified(self) -> None:
        if self.dirty and not self.save_current(silent=True):
            return
        name = filedialog.asksaveasfilename(
            parent=self.root, title="指定新的匯出目錄名稱（不可已存在）", confirmoverwrite=False
        )
        if name:
            config = self.config
            destination = Path(name)

            def done(result):
                messagebox.showinfo(
                    "已匯出",
                    f"{result['images']} 張圖片、{result['boxes']} 個框。\n{destination}\n訓練前仍須建立獨立 train/val 分割。",
                    parent=self.root,
                )
                self.update_status("完整審閱資料已匯出。")

            self._submit_job(lambda: export_verified(config, destination), done, bound_to_image=False)

    def show_notices(self) -> None:
        messagebox.showinfo("資料集通知", "\n\n".join(self._notices) or "沒有資料集變更或遷移通知。", parent=self.root)

    def show_help(self) -> None:
        messagebox.showinfo(
            "快捷鍵與審閱規則",
            "左拖空白：畫框；拖框：移動；拖控制點：縮放\n"
            "右拖：平移；滾輪：縮放；F：符合視窗\n"
            "←/→、P：上一張／略過下一張（不確認）\n"
            "N：確認本類／整張並下一張；M：確認本類無物件\n"
            "J/K：下一類／上一類（不確認）；[ / ]：切類別\n"
            "0–9：選類別；其餘類別從列表選取\n"
            "S：儲存；Ctrl+Z：復原；Ctrl+Y：重做；Delete：刪框\n"
            "雙擊選取框：置中放大\n\n"
            "儲存 ≠ 審閱完成。修改、復原與接受候選框皆需重新確認。\n"
            "無框 ≠ 已確認無物件。候選框只在人工接受後才寫入。\n"
            "中文為操作提示；classes.txt 英文類別及順序是訓練依據。",
            parent=self.root,
        )

    def show_metrics(self) -> None:
        def summary(values):
            if not values:
                return "尚無樣本"
            ordered = sorted(values)
            return f"n={len(values)} · median={ordered[len(ordered) // 2]:.2f} ms · p95={ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]:.2f} ms"

        messagebox.showinfo(
            "效能量測",
            f"圖片請求 → 解碼／標籤載入／首次重繪：\n{summary(self._load_timings)}\n\n"
            f"實際重繪（不含事件排程等待）：\n{summary(self._render_timings)}\n\n"
            f"影像快取 {self.loader.cache_bytes / (1024 * 1024):.1f} MiB\n命中次數 {self.loader.cache_hits}",
            parent=self.root,
        )

    def on_hover(self, event) -> None:
        self._hover = (event.x, event.y)
        if self.lens_enabled.get() and not getattr(self, "_lens_pending", False):
            self._lens_pending = True

            def update_lens():
                self._lens_pending = False
                if not self._closing:
                    self.draw_lens()

            self.root.after(16, update_lens)

    def on_hover_leave(self, _event=None) -> None:
        self._hover = None
        self.canvas.delete("lens")

    def on_lens_changed(self) -> None:
        self.render()
        self.schedule_session_save()

    def draw_lens(self) -> None:
        self.canvas.delete("lens")
        if not self.lens_enabled.get() or self._hover is None or self.image is None:
            return
        cx, cy = self._hover
        ix, iy = self.canvas_to_image(cx, cy)
        if not (0 <= ix < self.image.width and 0 <= iy < self.image.height):
            return
        self.canvas.create_line(cx, 0, cx, self.canvas.winfo_height(), fill="#cccccc", dash=(2, 4), tags="lens")
        self.canvas.create_line(0, cy, self.canvas.winfo_width(), cy, fill="#cccccc", dash=(2, 4), tags="lens")
        half = 32
        crop = self.image.crop((int(ix) - half, int(iy) - half, int(ix) + half, int(iy) + half))
        self._lens_photo = ImageTk.PhotoImage(crop.resize((192, 192), Image.Resampling.NEAREST))
        lx = max(0, self.canvas.winfo_width() - 204)
        self.canvas.create_image(lx, 12, image=self._lens_photo, anchor="nw", tags="lens")
        self.canvas.create_rectangle(lx, 12, lx + 192, 204, outline="white", tags="lens")
        self.canvas.create_line(lx + 96, 12, lx + 96, 204, fill="#ff5555", tags="lens")
        self.canvas.create_line(lx, 108, lx + 192, 108, fill="#ff5555", tags="lens")
        self.canvas.create_text(lx, 208, anchor="nw", text=f"原圖 3× · {ix:.1f}, {iy:.1f}", fill="white", tags="lens")

    def focus_selected_box(self) -> None:
        if self.image is None or self.selected_box is None:
            return
        box = self.boxes[self.selected_box]
        scale = min(
            self.canvas.winfo_width() * 0.65 / max(box.width, 1), self.canvas.winfo_height() * 0.65 / max(box.height, 1)
        )
        self.zoom = clamp(scale / self.fit_scale, 0.2, 20.0)
        self.offset_x = self.canvas.winfo_width() / 2 - (box.x1 + box.x2) / 2 * self.display_scale
        self.offset_y = self.canvas.winfo_height() / 2 - (box.y1 + box.y2) / 2 * self.display_scale
        self.render()
        self.schedule_session_save()

    def show_references(self) -> None:
        config, cls = self.config, self.current_class

        def done(result):
            crops, errors = result
            window = tk.Toplevel(self.root)
            window.title(f"已確認範例 — {self.class_names[cls]}")
            if not crops:
                ttk.Label(
                    window,
                    text="此類別尚無已明確確認的範例。\n先標註並確認一張圖片；不以未審候選框充當範例。",
                    padding=15,
                ).pack()
            window._photos = []
            for index, (name, crop) in enumerate(crops):
                photo = ImageTk.PhotoImage(crop)
                window._photos.append(photo)
                frame = ttk.Frame(window, padding=5)
                frame.grid(row=index // 3, column=index % 3, sticky="n")
                ttk.Label(frame, image=photo).pack()
                ttk.Label(frame, text=name, wraplength=280).pack()
            if errors:
                messagebox.showwarning("部分範例無法讀取", "\n".join(errors[:5]), parent=window)

        self._submit_job(lambda: reference_crops(config, cls), done, bound_to_image=False)

    def choose_model(self) -> None:
        name = filedialog.askopenfilename(
            parent=self.root, title="選擇可信任的 YOLO 權重（模型可能執行程式碼）", filetypes=[("YOLO weights", "*.pt")]
        )
        if name:
            if messagebox.askyesno(
                "模型信任", "PyTorch 權重可能包含可執行內容。只載入自己訓練或可信來源的模型。繼續？", parent=self.root
            ):
                self.model_path = Path(name)
                self.model_mapping = None
                self.update_status(f"模型：{self.model_path.name}；須精確類別名稱或明確映射。")

    def choose_model_mapping(self) -> None:
        name = filedialog.askopenfilename(
            parent=self.root, title="model name → dataset name/id JSON", filetypes=[("JSON", "*.json")]
        )
        if name:
            try:
                mapping = json.loads(Path(name).read_text(encoding="utf-8"))
                if not isinstance(mapping, dict):
                    raise ValueError("Mapping must be a JSON object.")
                self.model_mapping = mapping
                self.update_status("模型類別映射已載入。")
            except (OSError, ValueError) as exc:
                messagebox.showerror("類別映射錯誤", str(exc), parent=self.root)

    def _receive_candidates(self, candidates) -> None:
        self.candidates.extend(candidates)
        self.refresh_candidates()
        self.render()
        self.notebook.select(2)
        self.update_status(f"新增 {len(candidates)} 個候選；請逐一接受或拒絕。")

    def predict_current(self) -> None:
        if not self._editable():
            return
        if self.model_path is None:
            self.choose_model()
        if self.model_path is None:
            return
        try:
            confidence = float(self.confidence_var.get())
            if not math.isfinite(confidence) or not 0 < confidence <= 1:
                raise ValueError("Confidence must be within (0, 1].")
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("信心門檻錯誤", str(exc), parent=self.root)
            return
        image, model = self.image_path, self.model_path
        classes, mapping = list(self.class_names), self.model_mapping
        self._submit_job(lambda: predict_boxes(image, model, classes, confidence, mapping), self._receive_candidates)

    def propagate_previous(self) -> None:
        if not self._editable() or self.index == 0:
            self.update_status("目前沒有上一張影格。")
            return
        previous, current = self.images[self.index - 1], self.image_path

        def sequence(path):
            parts = path.stem.rsplit("_", 2)
            return (
                parts[0]
                if len(parts) == 3 and parts[1].isdigit() and parts[2].endswith("ms") and parts[2][:-2].isdigit()
                else None
            )

        if sequence(previous) is None or sequence(previous) != sequence(current):
            messagebox.showwarning(
                "不同影片序列", "傳播只適用於 prefix_frame_timestampms 檔名且同一影片序列的相鄰圖片。", parent=self.root
            )
            return
        try:
            with Image.open(previous) as source:
                boxes = read_yolo_boxes(label_path_for(previous, self.labels_dir), *source.size, len(self.class_names))
            boxes = [box for box in boxes if self.image_mode.get() or box.cls == self.current_class]
            if not boxes:
                raise LabelError(previous, "上一張圖片沒有此範圍的已接受框；不會複製假候選。")
        except (OSError, LabelError) as exc:
            messagebox.showwarning("無法傳播", str(exc), parent=self.root)
            return
        self._submit_job(lambda: propagate_boxes(previous, current, boxes), self._receive_candidates)

    def refresh_candidates(self) -> None:
        if not hasattr(self, "candidate_tree"):
            return
        self.candidate_tree.delete(*self.candidate_tree.get_children())
        for index, candidate in enumerate(self.candidates):
            score = f"{candidate.confidence:.2f}" if candidate.confidence is not None else "—"
            self.candidate_tree.insert(
                "",
                "end",
                iid=str(index),
                values=(f"{candidate.box.cls:02d} {self.class_label(candidate.box.cls)} · {candidate.source}", score),
            )

    def accept_candidates(self) -> None:
        if not self._editable():
            return
        selected = sorted(int(value) for value in self.candidate_tree.selection())
        if not selected:
            self.update_status("先選取要接受的候選框。")
            return
        self.push_undo_state(self.snapshot_state())
        for index in selected:
            self.boxes.append(self.candidates[index].box)
        self.candidates = [candidate for index, candidate in enumerate(self.candidates) if index not in selected]
        self.selected_box = len(self.boxes) - 1
        self.dirty = True
        self.refresh_box_tree()
        self.refresh_candidates()
        self.save_current(silent=True)
        self.render()
        self.update_status("已接受候選；仍需確認審閱。")

    def reject_candidates(self) -> None:
        selected = {int(value) for value in self.candidate_tree.selection()}
        self.candidates = [candidate for index, candidate in enumerate(self.candidates) if index not in selected]
        self.refresh_candidates()
        self.render()
        self.update_status()

    def clear_candidates(self) -> None:
        self.candidates.clear()
        self.refresh_candidates()
        self.render()
        self.update_status()

    def draw_candidates(self) -> None:
        selected = {int(value) for value in self.candidate_tree.selection()}
        for index, candidate in enumerate(self.candidates):
            box = candidate.box
            if not self._shows_class(box.cls):
                continue
            x1, y1 = self.image_to_canvas(box.x1, box.y1)
            x2, y2 = self.image_to_canvas(box.x2, box.y2)
            self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="#ffcc00", dash=(5, 3), width=3 if index in selected else 1, tags="overlay"
            )
            self.canvas.create_text(x1, y1 - 3, text=f"? {box.cls:02d}", anchor="sw", fill="#ffcc00", tags="overlay")

    def close(self) -> None:
        if self.dirty and not self.save_current(silent=True):
            return
        if not self._save_session():
            if not messagebox.askyesno("工作階段未儲存", "導覽狀態儲存失敗。標籤已保留；仍要關閉？", parent=self.root):
                return
        self._closing = True
        self.loader.close()
        self._executor.shutdown(wait=False, cancel_futures=True)
        for after_id in self.root.tk.call("after", "info"):
            self.root.after_cancel(after_id)
        self.root.destroy()


def choose_dataset_dialog(parent, classes_file: Path = DEFAULT_CLASSES_FILE) -> DatasetConfig | None:
    images = filedialog.askdirectory(parent=parent, title="選擇圖片目錄（直接包含圖片的目錄）")
    if not images:
        return None
    labels = filedialog.askdirectory(parent=parent, title="選擇標籤目錄（可選新建的空目錄）", mustexist=False)
    if not labels:
        return None
    classes = filedialog.askopenfilename(
        parent=parent,
        title="選擇類別檔（每行一類，順序固定）",
        initialdir=classes_file.parent,
        initialfile=classes_file.name,
        filetypes=[("Class catalog", "*.txt")],
    )
    if not classes:
        return None
    labels_path = Path(labels).resolve()
    return DatasetConfig(
        Path(images).resolve(), labels_path, Path(classes).resolve(), labels_path / ".annotation" / "review.json"
    )


def validate_dataset(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(DATASET_ROOT / "scripts" / "verify_yolo_labels.py"),
        "--images-dir",
        str(args.images_dir),
        "--labels-dir",
        str(args.labels_dir),
        "--classes-file",
        str(args.classes_file),
        "--pass-state",
        str(args.pass_state),
    ]
    if args.require_reviewed:
        command.append("--require-reviewed")
    raise SystemExit(subprocess.call(command))


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label Power: explicit-review desktop YOLO annotation.")
    parser.add_argument("--dataset", type=Path, help="Portable dataset profile JSON.")
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--labels-dir", type=Path)
    parser.add_argument("--classes-file", type=Path, default=DEFAULT_CLASSES_FILE)
    parser.add_argument("--pass-state", type=Path)
    parser.add_argument("--start", type=int, help="Zero-based image index; overrides session.")
    parser.add_argument("--start-class", type=int, help="Class id; overrides session.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-other-classes", action="store_true")
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    parser.add_argument("--cache-mb", type=int, default=128)
    parser.add_argument("--model", type=Path, help="Trusted local YOLO .pt weights; never downloaded automatically.")
    parser.add_argument("--model-mapping", type=Path, help="JSON: model class name -> dataset class name/id.")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--selftest", action="store_true", help="Read-only strict dataset validation without opening UI."
    )
    parser.add_argument(
        "--require-reviewed", action="store_true", help="With --selftest, require all image/class reviews."
    )
    args = parser.parse_args(argv)
    if not 16 <= args.cache_mb <= 4096:
        parser.error("--cache-mb must be within 16..4096.")
    if not math.isfinite(args.confidence) or not 0 < args.confidence <= 1:
        parser.error("--confidence must be within (0, 1].")
    if args.dataset and (args.images_dir or args.labels_dir or args.pass_state):
        parser.error("--dataset cannot be combined with image/label/pass-state paths.")
    if args.dataset:
        config = DatasetConfig.from_profile(args.dataset.expanduser().resolve())
        args.images_dir, args.labels_dir = config.images_dir, config.labels_dir
        args.classes_file, args.pass_state = config.classes_file, config.pass_state
    elif args.images_dir or args.labels_dir:
        if not args.images_dir or not args.labels_dir:
            parser.error("--images-dir and --labels-dir must be supplied together.")
        args.images_dir = args.images_dir.expanduser().resolve()
        args.labels_dir = args.labels_dir.expanduser().resolve()
        args.classes_file = args.classes_file.expanduser().resolve()
        args.pass_state = (
            args.pass_state.expanduser().resolve()
            if args.pass_state
            else args.labels_dir / ".annotation" / "review.json"
        )
    elif args.selftest:
        parser.error("--selftest requires --dataset or --images-dir/--labels-dir.")
    return args


def main() -> None:
    try:
        args = parse_args()
        if args.images_dir is None:
            selector = tk.Tk()
            selector.withdraw()
            config = choose_dataset_dialog(selector, args.classes_file)
            selector.destroy()
            if config is None:
                return
            args.images_dir, args.labels_dir = config.images_dir, config.labels_dir
            args.classes_file, args.pass_state = config.classes_file, config.pass_state
        if args.selftest:
            validate_dataset(args)
        app = AnnotationApp(args)
        app.run()
    except (OSError, LabelError, ValueError, tk.TclError) as exc:
        raise SystemExit(f"Cannot start Label Power: {exc}") from exc


if __name__ == "__main__":
    main()
