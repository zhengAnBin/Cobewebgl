---
name: cobe-dotted-globe
description: Implements a high-performance dotted globe (Cobe-style) with WebGL and GLSL. Uses pure WebGL (no Three.js), math-driven shaders for dot distribution, and optional Fibonacci sphere lattice. Use when building dotted/sphere globes, Cobe-like effects, spherical point distribution in fragment shaders, or when the user mentions Cobe, point-matrix Earth, or WebGL globe.
---

# Cobe 风格高性能点阵地球

基于标准 WebGL + GLSL，用数学公式在 shader 中实现点阵球体，无高清纹理、无 CPU 循环，体积可控制在约 5KB 量级，移动端友好。

## 实现流程概览

按顺序完成以下步骤，每步都在前一步基础上扩展：

1. **从圆开始** → 2. **球面贴纹理** → 3. **经纬均分点阵** → 4. **点阵 × 纹理 = 点阵地球** → 5. **斐波那契点阵（可选）** → 6. **菲涅尔 + 光照优化**

## 1. 从圆开始

在片元 shader 中：

- 将 `gl_FragCoord.xy / u_resolution.xy` 转为 NDC：`p = uv*2.-1.`，`p.x *= aspect`。
- 用 `dot(p, p)` 得到 x²+y²，与 r² 比较：`d2 < r*r` 为圆内，否则为背景。
- 圆内输出一种颜色，圆外输出背景色。

## 2. 给圆形贴上纹理（等距柱状投影）

- 圆内升维到 3D 球面：`z = sqrt(r*r - d2)`，`nor = normalize(vec3(p.x, p.y, z))`。
- 可选旋转：`nor = rotateY(time)*rotateX(time)*nor`。
- 等距柱状投影到 [0,1] UV：
  - 经度 U：`u_lon = atan(nor.x, nor.z) / (2.0*PI) + 0.5`
  - 纬度 V：`v_lat = asin(nor.y) / PI + 0.5`
- 用 `texture(u_texture, vec2(u_lon, v_lat)).r` 采样，输出到 `outColor`。

纹理只需 0/1 二值（陆/海），可用约 1KB 小图替代大图，效果接近。

## 3. 均分点阵（经纬网格）

- 球面法线 `nor` 映射到二维参数：`theta = acos(nor.y)`，`phi = atan(nor.z, nor.x)`，`u = (phi+PI)/(2.*PI)`，`v = theta/PI`，得到 `st = vec2(u,v)`。
- 网格：`grid = vec2(60., 30.)`，`ipos = floor(st * grid)`，`fpos = fract(st * grid)`。
- 格内画点：`dist = distance(fpos, vec2(0.5))`，`dotMask = smoothstep(0.1, 0.09, dist)`。
- 缺点：两极密集、赤道稀疏。

## 4. 点阵地球

- 点阵遮罩 × 纹理二值：`outColor = vec4(vec3(dotMask * mapValue), 1.0)`，其中 `mapValue = texture(u_texture, st).r`（或 step 二值化）。
- `dotMask` 与 `mapValue` 均为 0 或 1，相乘即得到“只在陆地上的点”的效果。

## 5. 斐波那契点阵（推荐）

用球面斐波那契分布替代经纬均分，避免极区过密、赤道过疏。

- 思路：高度 [-1,1] 均分 N 份；经度方向每点旋转黄金角 ≈ 137.5°（2π·(1-φ⁻¹)）。
- 在 shader 中无简单逆公式，采用 **Spherical Fibonacci Mapping**：先估计当前点所在维度 k，再算网格坐标 c，最后在**周围 4 个候选点**中找最近邻。
- 实现一个 `nearestFibonacciLattice(vec3 p, out float m)`：返回最近斐波那契点的 3D 方向，`m` 为距离。
- 用该最近点的经纬度做 UV：`gPhi = asin(gP.y)`，`gTheta = atan(gP.z, gP.x)`，`dotUV = vec2(0.5 + gTheta/kTau, 0.5 + gPhi/PI)`，再 `texture(u_texture, dotUV).r` 判断陆海。
- 点显隐：`v = smoothstep(.008, .0, dis)`，`outColor = vec4(vec3(v * isOnLand), 1.0)`。

详见 [reference.md](reference.md) 中的常数定义与 `nearestFibonacciLattice` 结构。

## 6. 优化整体效果

- **菲涅尔/边缘光**：在球体边缘用 `pow(1. - dotNL, 4.) * glowColor` 或基于到圆边距离的 `smoothstep` 减弱摩尔纹感。
- **光照**：`dotNL = dot(rawNor, light)`，用 `pow(dotNL, 3.)` 或 `pow(dotNL, .4)` 调制点阵亮度与基础色；`colorFactor = mix((1.-sample2)*pow(dotNL,.4), sample2, dark) + .1`。
- **外发光**：圆外也可加 `glowFactor`，与到圆的距离相关，再与 `glowColor` 混合到最终 `outColor`。

## 技术要点小结

| 项目 | 做法 |
|------|------|
| 圆/球数学 | 圆内 d²=x²+y²，球面 z=√(r²-d²)，nor 归一化 |
| UV 映射 | 经度 atan(x,z)，纬度 asin(y)，归一化到 [0,1] |
| 点阵 | 先 grid+floor/fract 做均分点；进阶用斐波那契最近邻 |
| 性能 | 纯 WebGL、小纹理(0/1)、GPU 数学，无 CPU 递归 |

## 参考资料

- [Cobe](https://github.com/shuding/cobe)
- [Spherical Fibonacci Lattice](https://observablehq.com/@mbostock/spherical-fibonacci-lattice)
- [Spherical fibonacci mapping (ACM)](https://dl.acm.org/doi/10.1145/2816795.2818131)

更多公式与完整代码片段见 [reference.md](reference.md)。
