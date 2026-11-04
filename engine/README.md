# engine/ — twilight-fable 的运行时

`engine/` 是把 `styles/twilight-fable/` 的多场景 HTML 与 Canvas2D 世界合成为确定性画面的最小运行时（HTML-in-Canvas → WebGL2 合成 → 后处理 pass）。本仓库只保留这条风格实际用到的模块。

| 文件 | 职责 |
| :-- | :-- |
| [`html_in_canvas.js`](./html_in_canvas.js) | HTML-in-Canvas 兼容层：`ensureCanvasDrawable()` / `markDrawable()`、特性探测、`texElementSubImage2D`（Chrome 157+）与旧版 `texElementImage2D`（151–154）回退、`updateElementGeometry()` |
| [`stage.js`](./stage.js) | 每帧管线：可选 `world.prepareDom(frame)` → `world.render(...)` 上传 Canvas2D 背景 → 把当前 `.frame.scene` 与 `[data-layer]` 上传为纹理 → 合成 → WebGL2 pass 链 → 屏幕 |
| [`motion.js`](./motion.js) | 确定性 `data-*` 动效（`springStep`、`smootherstep`…）、`[data-cam]` 2D 相机变换、多场景序列器（`prepareScenes` / `applyScene`）、声音 cue 提取（`extractCues` → `window.CUES`）与运行时 cue 注册（`registerRuntimeCue`） |
| [`studio2d.js`](./studio2d.js) | 2D 绘本舞台：全幅 1.0x 光学基线、每场一次单向微镜（`static` / `push` / `pull` / `pan` / `rise` / `drop` / `drift`）、多层视差（`beginLayer` / `endLayer`）、分镜导出（`is2D: true`） |
| [`critter2d.js`](./critter2d.js) | 2D 生灵几何与运动学：参数化物种 × 体型 × 视角、120Hz 弹簧骨架、步态与表情 |
| [`passes/`](./passes/index.js) | 全屏 WebGL2 pass：只保留 [`grain.js`](./passes/grain.js)（颗粒 + 暗角 + 暖调） |
| [`base.css`](./base.css) | 排版与舞台覆盖层契约（`.frame.scene`、`[data-layer]`…），按 `--u`（画布短边 1%）缩放，适配 16:9、9:16、1:1 |
| [`serve.mjs`](./serve.mjs) / [`server.mjs`](./server.mjs) | 本地静态服务（`--open` 打开已探测到的 Chrome） |
| [`browser.mjs`](./browser.mjs) | 无头 Chromium 启动器（确定性 sRGB），读取 `CHROME_BIN` / `PLAYWRIGHT_CHROME` |
| [`render.mjs`](./render.mjs) | 转发到 [`tools/render.mjs`](../tools/render.mjs) 的薄封装 |

## 多场景与声音 cue

`styles/twilight-fable/scenes.html` 在 `<canvas id="stage" content="drawable" layoutsubtree>` 里定义多个场景，纯画面场景不放文字卡，用 `data-cues` 驱动拟音：

```html
<section class="frame scene"
         data-scene="fog-wood"
         data-title="01 · 雾林 · Fog Wood"
         data-dur="3.2"
         data-look="fogwood"
         data-cues="0.9:layer_float:far-window, 1.7:count_tick:ears-perk"></section>
```

- **kernel 与 world**：可复用的材质、地形/水/室内生成与生灵在 `styles/twilight-fable/kernel.js`；一个具体故事在 `styles/twilight-fable/world.js`（或经 `?world=` / `--world` 传入的外部模块）。
- **`extractCues(root)`**：解析每个 `.frame.scene` 上的 `data-cues`（以及 `[data-enter]` / `[data-draw]` / `[data-count]` / `[data-layer]`）成 `window.CUES`，由 `tools/sound.py` 消费；`critter2d` 在运行时通过 `registerRuntimeCue` 追加 `paw_step` / `critter_hop`。
