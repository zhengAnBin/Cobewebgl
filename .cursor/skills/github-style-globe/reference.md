# GitHub 风格地球 - 代码与参数参考

## 1. 陆地圆片：经纬循环与陆地判定

```javascript
const DEG2RAD = Math.PI / 180;
const GLOBE_RADIUS = 100; // 示例

for (let lat = -90; lat <= 90; lat += 180 / rows) {
  const radius = Math.cos(Math.abs(lat) * DEG2RAD) * GLOBE_RADIUS;
  const circumference = radius * Math.PI * 2;
  const dotsForLat = circumference * dotDensity;
  for (let x = 0; x < dotsForLat; x++) {
    const long = -180 + (x * 360) / dotsForLat;
    if (!this.visibilityForCoordinate(long, lat)) continue;
    // 写入该圆片的 matrix 等 instance 数据
  }
}
```

陆地判定：加载小尺寸世界图 PNG，用 canvas `getImageData()` 取像素；将 (long, lat) 映射到图像 (u, v)，读取该像素 alpha，`alpha >= 90` 则视为陆地并绘制圆片。

## 2. 初始旋转（时区近似）

```javascript
const date = new Date();
const timeZoneOffset = date.getTimezoneOffset() || 0;
const timeZoneMaxOffset = 60 * 12;
rotationOffset.y = ROTATION_OFFSET.y + Math.PI * (timeZoneOffset / timeZoneMaxOffset);
```

## 3. PR 弧线（贝塞尔 + 管状几何）

```javascript
const curve = new THREE.CubicBezierCurve3(startLocation, ctrl1, ctrl2, endLocation);
// 使用 TubeBufferGeometry 沿 curve 生成几何
// 用 geometry.setDrawRange(0, count) 控制绘制长度以做生长/消失动画
```

## 4. 落点动画（6% 缓出）

```javascript
// 实心圆：每帧向 scale=1 逼近
const scale = animated.dot.scale.x + (1 - animated.dot.scale.x) * 0.06;
animated.dot.scale.set(scale, scale, 1);

// 扩散环：放大并淡出
const scaleUpFade = animated.dotFade.scale.x + (1 - animated.dotFade.scale.x) * 0.06;
animated.dotFade.scale.set(scaleUpFade, scaleUpFade, 1);
animated.dotFade.material.opacity = 1 - scaleUpFade;
```

## 5. Halo 球（柔化锯齿边）

```javascript
const halo = new THREE.Mesh(haloGeometry, haloMaterial);
halo.scale.multiplyScalar(1.15);
halo.rotateX(Math.PI * 0.03);
halo.rotateY(Math.PI * 0.03);
this.haloContainer.add(halo);
```

haloMaterial 使用自定义 shader，在球体**背面**绘制渐变（如从中心到边缘变暗），置于主球后方。

## 6. 圆片按深度淡出（减轻摩尔纹）

Fragment shader 中：

```glsl
if (gl_FragCoord.z > fadeThreshold) {
  gl_FragColor.a = 1.0 + (fadeThreshold - gl_FragCoord.z) * alphaFallOff;
}
```

根据设备与效果调节 `fadeThreshold`、`alphaFallOff`。

## 7. 首屏：SVG 占位与 Web Animations API 切换

```javascript
const keyframesIn = [
  { opacity: 0, transform: 'scale(0.8)' },
  { opacity: 1, transform: 'scale(1)' }
];
const keyframesOut = [
  { opacity: 1, transform: 'scale(0.8)' },
  { opacity: 0, transform: 'scale(1)' }
];
const options = { fill: 'both', duration: 600, easing: 'ease' };

this.renderer.domElement.animate(keyframesIn, options);
const placeHolderAnim = placeholder.animate(keyframesOut, options);
placeHolderAnim.addEventListener('finish', () => {
  placeholder.remove();
});
```

## 8. 优雅降级（质量档位示例）

```javascript
// 降低 pixel ratio（如从 2.0 到 1.5）
this.renderer.setPixelRatio(Math.min(AppProps.pixelRatio, 1.5));
// 减少同时显示的 PR 数量
this.indexIncrementSpeed = VISIBLE_INCREMENT_SPEED / 3 * 2;
// 降低 raycast 频率（每多等几帧再检测一次）
this.raycastTrigger = RAYCAST_TRIGGER + 4;
// 降低陆地圆片密度并重建
this.worldDotDensity = WORLD_DOT_DENSITY * 0.65;
this.resetWorldMap();
this.buildWorldGeometry();
```

FPS 监控：统计最近 50 帧平均帧率，若低于约 55.5 则触发降级并进入下一档位。
