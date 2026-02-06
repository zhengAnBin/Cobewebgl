# Cobe 点阵地球 - 公式与代码参考

## 常用常数（GLSL）

```glsl
#define PI 3.14159265359
#define kTau 6.28318530718
#define kPhi 1.618033988749
#define phiMinusOne 0.618033988749
#define twoPiOnPhi 3.88322207745
#define sqrt5 2.2360679775
#define byLogPhiPlusOne 2.07808692124
#define byDots (1.0 / dots)
// dots: 点阵总点数，如 3000.0
```

## 1. 画圆（片元）

```glsl
vec2 uv = gl_FragCoord.xy / u_resolution.xy;
float aspect = u_resolution.x / u_resolution.y;
vec2 p = uv * 2. - 1.;
p.x *= aspect;
float r = 0.5;
float d2 = dot(p, p);
if (d2 < r * r) {
    outColor = vec4(vec3(0.0), 1.0);
} else {
    outColor = vec4(vec3(0.141), 1.0);
}
```

## 2. 球面法线 + 等距柱状 UV

```glsl
float z = sqrt(r * r - d2);
vec3 nor = normalize(vec3(p.x, p.y, z));
// 旋转（可选）
nor = rotateY(u_time * 0.5) * rotateX(u_time * 0.3) * nor;

float u_lon = atan(nor.x, nor.z) / (2.0 * PI) + 0.5;
float v_lat = asin(nor.y) / PI + 0.5;
float mapValue = texture(u_texture, vec2(u_lon, v_lat)).r;
```

## 3. 经纬映射函数 mappingUV

```glsl
vec2 mappingUV(vec3 nor) {
    float theta = acos(nor.y);
    float phi = atan(nor.z, nor.x);
    float u = (phi + PI) / (2.0 * PI);
    float v = theta / PI;
    return vec2(u, v);
}
```

## 4. 均分点阵

```glsl
vec2 st = mappingUV(nor);
vec2 grid = vec2(60.0, 30.0);
vec2 ipos = floor(st * grid);
vec2 fpos = fract(st * grid);
float dist = distance(fpos, vec2(0.5));
float dotMask = smoothstep(0.1, 0.09, dist);
float mapValue = texture(u_texture, st).r;
outColor = vec4(vec3(dotMask * mapValue), 1.0);
```

## 5. 斐波那契最近邻（用法）

```glsl
float dis;
vec3 gP = nearestFibonacciLattice(nor, dis);
float gPhi = asin(gP.y);
float gTheta = atan(gP.z, gP.x);
vec2 dotUV = vec2(0.5 + gTheta / kTau, 0.5 + gPhi / PI);
float isOnLand = step(0.45, texture(u_texture, dotUV).r);
float v = smoothstep(.008, .0, dis);
outColor = vec4(vec3(v * isOnLand), 1.0);
```

`nearestFibonacciLattice` 内部：坐标系转换 (p.xzy) → 按纬度估计 k → 构造斐波那契基向量 br1、br2 → 球面转极坐标 sp → 逆矩阵得格子坐标 c → 对 c 周围 4 个点算 idx、theta、3D 点，取距离最小者。完整实现见 Cobe 源码或博客中的 GLSL 块。

## 6. 光照与菲涅尔（片段）

```glsl
vec3 light = vec3(0, 0, 1);
float dotNL = dot(rawNor, light);
float lighting = pow(dotNL, 3.0) * dotsBrightness;
float sample2 = mapColor * v * lighting;
float colorFactor = mix((1. - sample2) * pow(dotNL, .4), sample2, dark) + .1;
layer += vec4(baseColor * colorFactor, 1.);
layer.xyz += pow(1. - dotNL, 4.) * glowColor;
// 边缘发光
glowFactor = pow(dot(normalize(vec3(-uv, sqrt(1.-l))), vec3(0,0,1)), 4.) * smoothstep(0., 1., .2/(l-rSquared));
```

## 旋转矩阵（rotateX / rotateY）

需在 shader 中定义 3×3 旋转矩阵，或由 JS 传入 uniform。绕 Y 轴、绕 X 轴旋转法线后再做 UV 与点阵计算。
