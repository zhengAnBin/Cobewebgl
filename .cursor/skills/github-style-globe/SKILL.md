---
name: github-style-globe
description: Renders a Three.js-based interactive globe with circle-based land regions, arcs/spikes for data visualization, halo effect, and performance optimizations. Use when building GitHub-style homepage globes, Three.js globe with instanced circles, PR/event arcs on a sphere, graceful degradation for WebGL, or when the user mentions GitHub globe, interactive globe visualization, or circle-based world map.
---

# GitHub 风格地球（Three.js 实现）

基于 [GitHub 官方博客《How we built the GitHub globe》](https://github.blog/engineering/how-we-built-the-github-globe/) 的实践总结。使用 Three.js、无纹理、用大量小圆片（CircleBufferGeometry + InstancedMesh）表示陆地，配合光晕、弧线动画与多档位性能降级。

## 场景组成（五层）

1. **Halo**：比球略大的背面球体 + 自定义 shader 渐变，用于弱化锯齿边。
2. **Globe**：主球体，打四盏灯，无纹理。
3. **Earth regions**：约 1.2 万个小圆（五边形 CircleBufferGeometry），仅陆地处绘制。
4. **Blue spikes**：开放中的 PR（可选用线条/尖刺）。
5. **Pink arcs**：已合并的 PR，两点间贝塞尔弧线 + 管状几何。

## 1. 陆地圆片：经纬循环 + 密度

- 陆地/海洋用**小 PNG 世界图**判定：`getImageData()` 取像素，经度纬度映射到 UV，若该像素 **alpha ≥ 90** 则视为陆地，才加入圆片。
- 圆片位置用经纬循环生成，**从南极往北**遍历纬度；每纬度的圆数由该纬度的「周长 × 密度」决定，保证沿纬线均匀分布。

核心循环逻辑：

- `lat`：从 -90 到 90，步进 `180/rows`。
- 该纬度在球面上的半径：`radius = cos(|lat| * DEG2RAD) * GLOBE_RADIUS`。
- 该纬线周长：`circumference = radius * 2 * PI`。
- 该纬线圆片数：`dotsForLat = circumference * dotDensity`。
- 经度：`long = -180 + x * 360 / dotsForLat`（x 从 0 到 dotsForLat）。
- 对每个 (long, lat) 调用 `visibilityForCoordinate(long, lat)`；为陆地则写入该圆片的矩阵等数据。
- 最后用 **CircleBufferGeometry + InstancedMesh** 一次性绘制所有圆片。

详见 [reference.md](reference.md) 中的循环与 land/water 判定。

## 2. 初始旋转：让用户“看到自己所在时区”

- 不做 IP 定位，用 **时区偏移** 近似：`getTimezoneOffset()`，换算成绕 Y 轴的弧度。
- `rotationOffset.y = ROTATION_OFFSET.y + PI * (timeZoneOffset / timeZoneMaxOffset)`，其中 `timeZoneMaxOffset = 60*12`（分钟）。
- 首帧即可用，无请求延迟。

## 3. 弧线（如 PR 合并弧）

- 每条弧有两个端点：起点（如 open 地）、终点（如 merge 地）。
- 用 **CubicBezierCurve3(start, ctrl1, ctrl2, end)** 生成曲线；可设多档「轨道」高度，两点距离越远弧线拉得越高。
- 用 **TubeBufferGeometry** 沿曲线生成管状几何；用 **setDrawRange()** 控制绘制长度，实现弧线生长/消失动画。

## 4. 落点动效（圆点 + 扩散环）

- 弧线到达终点时：一个**实心圆**常驻，一个**圆环**做“放大并淡出”。
- 缓动：每帧向目标逼近约 6%，自然缓出。  
  `scale += (1 - scale) * 0.06`，圆环的 `opacity = 1 - scaleUpFade`。

## 5. 性能取舍与光晕（Halo）

- **关闭 antialias** 以保帧率 → 球体边缘会有硬锯齿。
- **Halo**：在球体后方放一个稍大的球（如 `scale.multiplyScalar(1.15)`），略旋转（如 `rotateX(PI*0.03)`、`rotateY(PI*0.03)`），用**自定义 shader 在背面画渐变**，视觉上柔化边缘、替代抗锯齿。

## 6. 摩尔纹与“大气”感

- 圆片在球体边缘密集时易出现**摩尔纹**。
- 在圆片的 **fragment shader** 里按**与相机距离**调 alpha：超过 `fadeThreshold` 时按 `alphaFallOff` 线性减弱 alpha，模拟大气与边缘虚化，减轻摩尔纹。

详见 [reference.md](reference.md) 中的片段 shader 片段。

## 7. 首屏体感速度：SVG 占位 + 交叠动画

- 用 Figma 等做一版**纯渐变的静态地球 SVG**，直接内嵌在 HTML，首屏即可见。
- WebGL 首帧就绪后，用 **Web Animations API** 做 canvas 与 SVG 的**交叉淡入淡出 + 缩放**（如 600ms、ease、scale 0.8→1），动画结束后移除占位 SVG，避免首屏白屏。

## 8. 优雅降级（Quality Tiers）

- **持续监控 FPS**（如最近 50 帧）；若持续低于约 **55.5 FPS** 则进入下一档质量。
- 多档位可依次：
  - 降低 **pixel ratio**（如 2.0 → 1.5）。
  - 降低 **raycast 频率**（如每多等几帧再检测一次 hover）。
  - 减少**同时显示的弧/点数量**。
  - 降低 **陆地圆片密度**（如 `worldDotDensity *= 0.65`），然后 **resetWorldMap()** 再 **buildWorldGeometry()** 重建地球几何。

这样在低端设备上自动减少圆片数与渲染负担，保持可玩帧率。

## 技术要点小结

| 项目 | 做法 |
|------|------|
| 陆地表示 | PNG 世界图 getImageData，alpha≥90 为陆地；经纬循环按周长密度分布圆片 |
| 几何 | CircleBufferGeometry + InstancedMesh，约 1.2 万圆片 |
| 弧线 | CubicBezierCurve3 + TubeBufferGeometry + setDrawRange 动画 |
| 边缘/摩尔纹 | Halo 球 + 背面渐变 shader；圆片 fragment 按深度淡出 alpha |
| 首屏 | SVG 占位 + Web Animations API 与 canvas 交叉淡入淡出 |
| 性能 | 关 antialias、FPS 监控、多档位降密度与 pixel ratio |

## 参考资料

- 原文：[How we built the GitHub globe](https://github.blog/engineering/how-we-built-the-github-globe/)
- Three.js：[CircleBufferGeometry](https://threejs.org/docs/#api/en/geometries/CircleBufferGeometry)、[InstancedMesh](https://threejs.org/docs/#api/en/objects/InstancedMesh)、[TubeBufferGeometry](https://threejs.org/docs/#api/en/geometries/TubeBufferGeometry)、[BufferGeometry.setDrawRange](https://threejs.org/docs/#api/en/core/BufferGeometry.setDrawRange)

更多代码片段与参数见 [reference.md](reference.md)。
