/**
 * Cobe 风格高性能点阵地球 (v2)
 * 支持可调参数：点密集度 (u_dots) 与 点大小 (u_dotSize)
 *
 * 参考：
 * - https://github.com/shuding/cobe
 * - Spherical Fibonacci Mapping (ACM)
 * - 博客：前端实现一个高性能的点阵地球-Cobe
 */

// ───────────────────────────────────────────────
//  Vertex Shader
// ───────────────────────────────────────────────
const VERTEX_SHADER = `
  attribute vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

// ───────────────────────────────────────────────
//  Fragment Shader
// ───────────────────────────────────────────────
const FRAGMENT_SHADER = `
  precision highp float;

  uniform vec2  u_resolution;
  uniform float u_time;
  uniform sampler2D u_texture;

  // 旋转角度
  uniform float u_phi;
  uniform float u_theta;

  // 外观
  uniform vec3  u_baseColor;   // 地球底色（海洋/非陆地区域）
  uniform vec3  u_glowColor;   // 边缘发光色
  uniform vec3  u_dotColor;    // 点阵颜色（陆地点）
  uniform float u_opacity;
  uniform float u_glowOn;     // 边缘发光开关（0.0 关 / 1.0 开）
  uniform float u_debug;      // 诊断模式: 0=正常, 1=所有点(跳过纹理), 2=显示UV, 3=显示纹理采样值

  // ★ 用户可调参数 ★
  uniform float u_dots;     // 斐波那契点阵总点数（密集度）
  uniform float u_dotSize;  // 每个点的半径（step 阈值）

  // 常量
  const float PI   = 3.14159265359;
  const float kPhi = 1.618033988749895;
  const float phiMinusOne = 0.618033988749895;
  const float sqrt5 = 2.23606797749979;
  const float kTau  = 6.283185307179586;
  const float twoPiOnPhi = 3.883222077450933;
  const float byLogPhiPlusOne = 1.0 / log2(1.618033988749895 + 1.0);

  // ── 旋转矩阵 ──
  mat3 rotateX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1., 0., 0.,
                0., c, -s,
                0., s,  c);
  }
  mat3 rotateY(float a) {
    float c = cos(a), s = sin(a);
    return mat3( c, 0., s,
                 0., 1., 0.,
                -s, 0., c);
  }

  // ── 斐波那契最近邻搜索 ──
  vec3 nearestFibonacciLattice(vec3 p, out float m) {
    float dots = u_dots;
    float byDots = 1.0 / dots;

    // 坐标系转换 y-up → z-up
    p = p.xzy;

    // 自适应层级
    float k = max(2., floor(
      log2(sqrt5 * dots * PI * (1. - p.z * p.z)) * byLogPhiPlusOne
    ));

    // 斐波那契网格基向量
    vec2 f   = floor(pow(kPhi, k) / sqrt5 * vec2(1., kPhi) + .5);
    vec2 br1 = fract((f + 1.) * phiMinusOne) * kTau - twoPiOnPhi;
    vec2 br2 = -2. * f;

    // 球面极坐标
    vec2 sp = vec2(atan(p.y, p.x), p.z - 1.);

    // 逆矩阵求格子坐标
    float denom = br1.x * br2.y - br2.x * br1.y;
    vec2 c = floor(vec2(
      br2.y * sp.x - br1.y * (sp.y * dots + 1.),
     -br2.x * sp.x + br1.x * (sp.y * dots + 1.)
    ) / denom);

    float mindist = PI;
    vec3  minip   = vec3(0., 0., 1.);

    // 4 邻域搜索
    for (float s = 0.; s < 4.; s += 1.) {
      vec2 o = vec2(mod(s, 2.), floor(s * .5));
      float idx = dot(f, c + o);
      if (idx > dots || idx < 0.) continue;

      // 快速计算经度 theta（黄金序列分数）
      float fracV = fract(idx * phiMinusOne);
      float theta = fract(fracV) * kTau;

      // 还原候选点 3D 坐标
      float cosphi = 1. - 2. * idx * byDots;
      float sinphi = sqrt(1. - cosphi * cosphi);
      vec3 sample2 = vec3(cos(theta) * sinphi, sin(theta) * sinphi, cosphi);

      float dist = length(p - sample2);
      if (dist < mindist) {
        mindist = dist;
        minip   = sample2;
      }
    }

    m = mindist;
    return minip.xzy; // 转回 y-up
  }

  // ── 主函数 ──
  void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution.xy;
    float aspect = u_resolution.x / u_resolution.y;
    vec2 p = uv * 2. - 1.;
    p.x *= aspect;

    float r = 0.5;
    float d2 = dot(p, p);
    float rSquared = r * r;
    float l = d2;

    float glowFactor = 0.;
    vec4 color = vec4(0.);

    if (d2 < rSquared) {
      // 球面法线
      float z = sqrt(rSquared - d2);
      vec3 nor = normalize(vec3(p.x, p.y, z));

      // 旋转
      nor = rotateY(u_phi) * rotateX(u_theta) * nor;

      // 斐波那契最近邻
      float dis;
      vec3 gP = nearestFibonacciLattice(nor, dis);

      // 最近点 → 地理经纬度 → 标准等距柱状投影 UV
      float gLat = asin(gP.y);                           // 纬度 [-π/2, π/2]
      float gLng = PI * 0.5 - atan(gP.z, gP.x);         // 经度 (rad)
      // fract() 确保 u 始终在 [0,1)，避免依赖 REPEAT wrap
      vec2 dotUV = vec2(fract(gLng / kTau + 0.5), gLat / PI + 0.5);
      float texSample = texture2D(u_texture, dotUV).r;
      float mapColor = step(0.5, texSample);

      // ★ 用 u_dotSize 控制点的大小（硬边实心圆点） ★
      float v = step(dis, u_dotSize);

      // ── 诊断模式 ──
      if (u_debug > 0.5 && u_debug < 1.5) {
        // 模式1: 显示所有点（忽略纹理），确认斐波那契点阵是否正常
        float dotMask = v;
        vec3 surfaceColor = mix(u_baseColor, u_dotColor, dotMask);
        color = vec4(surfaceColor * u_opacity, 1.);
      } else if (u_debug > 1.5 && u_debug < 2.5) {
        // 模式2: 用UV坐标作为颜色，检查UV映射
        color = vec4(dotUV.x, dotUV.y, 0.0, 1.0);
      } else if (u_debug > 2.5 && u_debug < 3.5) {
        // 模式3: 显示纹理采样值（白=陆地，黑=海洋），叠加点阵
        float dotMask = v * mapColor;
        // 背景用纹理灰度显示
        vec3 surfaceColor = mix(vec3(texSample * 0.3), u_dotColor, dotMask);
        color = vec4(surfaceColor, 1.);
      } else {
        // 正常模式
        float dotMask = mapColor * v;
        vec3 surfaceColor = mix(u_baseColor, u_dotColor, dotMask);
        color = vec4(surfaceColor * u_opacity, 1.);
      }

      // 边缘柔和渐隐（保留球体立体感）
      float edgeFade = sqrt(1.0 - d2 / rSquared);
      color.rgb *= mix(0.3, 1.0, edgeFade);

      // 内发光（受开关控制）
      glowFactor = u_glowOn * pow(
        dot(normalize(vec3(-uv, sqrt(1. - l))), vec3(0., 0., 1.)),
        4.
      ) * smoothstep(0., 1., .2 / (l - rSquared));

    } else {
      // 外发光（受开关控制）
      float outD = sqrt(0.2 / (l - rSquared));
      glowFactor = u_glowOn * smoothstep(0.5, 1., outD / (outD + 1.));
    }

    gl_FragColor = color + vec4(glowFactor * u_glowColor, glowFactor);
  }
`;

// ───────────────────────────────────────────────
//  Arc Vertex Shader
// ───────────────────────────────────────────────
const ARC_VERTEX_SHADER = `
  attribute vec3 a_arcPos;
  uniform float u_phi;
  uniform float u_theta;
  uniform float u_aspect;
  varying float v_z;

  mat3 rotateX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1., 0., 0., 0., c, -s, 0., s, c);
  }
  mat3 rotateY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c, 0., s, 0., 1., 0., -s, 0., c);
  }

  void main() {
    // World -> View: inverse of globe's rotation
    vec3 viewPos = rotateX(-u_theta) * rotateY(-u_phi) * a_arcPos;
    v_z = viewPos.z;
    // Scale to match globe visual radius (0.5) and convert to clip space
    gl_Position = vec4(viewPos.x * 0.5 / u_aspect, viewPos.y * 0.5, 0.0, 1.0);
  }
`;

// ───────────────────────────────────────────────
//  Arc Fragment Shader
// ───────────────────────────────────────────────
const ARC_FRAGMENT_SHADER = `
  precision highp float;
  uniform vec3 u_arcColor;
  uniform float u_arcAlpha;
  varying float v_z;

  void main() {
    if (v_z < 0.0) discard;
    // Slight fade near the sphere surface for depth cue
    float zFade = smoothstep(0.0, 0.15, v_z);
    gl_FragColor = vec4(u_arcColor, u_arcAlpha * zFade);
  }
`;

// ───────────────────────────────────────────────
//  工具函数
// ───────────────────────────────────────────────
function createProgram(gl, vsSrc, fsSrc) {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, vsSrc);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    console.error('Vertex shader error:', gl.getShaderInfoLog(vs));
    gl.deleteShader(vs);
    return null;
  }

  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, fsSrc);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    console.error('Fragment shader error:', gl.getShaderInfoLog(fs));
    gl.deleteShader(fs);
    gl.deleteShader(vs);
    return null;
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Link error:', gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return prog;
}

/**
 * 从 GeoJSON 生成标准等距柱状投影纹理
 *
 * 新策略（最简洁可靠）：
 *  1. Canvas 上画标准等距柱状投影世界地图（北极在上）
 *  2. Shader 直接从 Fibonacci 点算出地理经纬度
 *  3. Shader UV = 标准等距柱状投影公式
 *  4. 纹理就是标准世界地图，零偏移、零镜像
 *
 * 关键推导：
 *   nearestFibonacciLattice 内部做了 p.xzy（y-up → z-up），
 *   返回时又做 minip.xzy（z-up → y-up）。
 *   gP 在 y-up 中：gP = (cosθ·sinφ, cosφ, sinθ·sinφ)
 *   地理纬度 = asin(gP.y)
 *   地理经度 = π/2 − atan(gP.z, gP.x)
 *   标准 UV: u = (lng+π)/(2π) = 0.5 + lng/(2π)
 *            v = (lat+π/2)/π  = 0.5 + lat/π
 */
function createWorldTextureFromGeoJSON(gl, geojson) {
  const W = 720, H = 360;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');

  // 画标准等距柱状投影（北极在 y=0）
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = '#fff';

  for (const feature of geojson.features) {
    const geom = feature.geometry;
    if (!geom) continue;
    let polys = [];
    if (geom.type === 'Polygon') polys = [geom.coordinates];
    else if (geom.type === 'MultiPolygon') polys = geom.coordinates;

    for (const poly of polys) {
      ctx.beginPath();
      const ring = poly[0];
      for (let i = 0; i < ring.length; i++) {
        const [lng, lat] = ring[i];
        const x = (lng + 180) / 360 * W;
        const y = (90 - lat) / 180 * H;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fill();
    }
  }

  // ── 诊断：统计像素 + 在页面显示小地图 ──
  const imgData = ctx.getImageData(0, 0, W, H).data;
  let whiteCount = 0;
  for (let i = 0; i < imgData.length; i += 4) {
    if (imgData[i] > 128) whiteCount++;
  }
  console.log(`[TEXTURE] Canvas ${W}x${H}, 陆地像素: ${whiteCount}, 海洋像素: ${W*H - whiteCount}, 陆地占比: ${(whiteCount/(W*H)*100).toFixed(1)}%`);

  // 在页面左下角显示调试小地图
  canvas.style.cssText = 'position:fixed;left:10px;bottom:10px;width:200px;height:100px;border:1px solid #0f0;z-index:999;image-rendering:pixelated;';
  document.body.appendChild(canvas);

  // 直接上传 Canvas 作为纹理
  // FLIP_Y = true：Canvas 顶部(北极 y=0) → 纹理 v=1(高纬度)
  // 对应 shader: dotUV.y = lat/π + 0.5，北极 lat=π/2 → v=1 ✓
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);  // 恢复默认
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  // WebGL1 NPOT 纹理(720x360)只能用 CLAMP_TO_EDGE，否则纹理 incomplete 返回全黑！
  // shader 中已用 fract() 处理经度环绕
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

// ───────────────────────────────────────────────
//  弧线工具函数
// ───────────────────────────────────────────────
const DEG2RAD = Math.PI / 180;

/** 经纬度 → 单位球面 3D 坐标 (y-up) */
function latLngToVec3(lat, lng) {
  const latR = lat * DEG2RAD;
  const lngR = lng * DEG2RAD;
  return [
    Math.cos(latR) * Math.sin(lngR),
    Math.sin(latR),
    Math.cos(latR) * Math.cos(lngR)
  ];
}

function vec3Dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function vec3Length(a) { return Math.sqrt(vec3Dot(a, a)); }
function vec3Normalize(a) {
  const l = vec3Length(a);
  return l > 1e-8 ? [a[0]/l, a[1]/l, a[2]/l] : [0, 0, 1];
}

/** 球面线性插值 */
function slerp(a, b, t) {
  const d = Math.max(-1, Math.min(1, vec3Dot(a, b)));
  const angle = Math.acos(d);
  if (angle < 0.001) {
    // 几乎重合，直接线性插值
    const x = a[0]*(1-t) + b[0]*t;
    const y = a[1]*(1-t) + b[1]*t;
    const z = a[2]*(1-t) + b[2]*t;
    return vec3Normalize([x, y, z]);
  }
  const sinA = Math.sin(angle);
  const wa = Math.sin((1 - t) * angle) / sinA;
  const wb = Math.sin(t * angle) / sinA;
  return [a[0]*wa + b[0]*wb, a[1]*wa + b[1]*wb, a[2]*wa + b[2]*wb];
}

/**
 * 生成弧线采样点（沿大圆路径 + 弧形隆起）
 * 返回 Float32Array [x0,y0,z0, x1,y1,z1, ...]
 */
function generateArcPoints(startVec, endVec, numSegments) {
  numSegments = numSegments || 64;
  const d = Math.max(-1, Math.min(1, vec3Dot(startVec, endVec)));
  const angularDist = Math.acos(d);
  // 两点越远弧线越高
  const arcHeight = 0.15 + 0.3 * angularDist;

  const points = [];
  for (let i = 0; i <= numSegments; i++) {
    const t = i / numSegments;
    const p = slerp(startVec, endVec, t);
    // 抛物线隆起，峰值在中点
    const elevation = 1.0 + arcHeight * Math.sin(Math.PI * t);
    points.push(p[0] * elevation, p[1] * elevation, p[2] * elevation);
  }
  return new Float32Array(points);
}

/** hex 颜色字符串 → [r, g, b] 归一化 */
function parseHexColor(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
}

// ───────────────────────────────────────────────
//  弧线动画管理器（滑动线段 + 每弧独立颜色）
// ───────────────────────────────────────────────
class ArcAnimator {
  /**
   * @param {Array} arcsData - [{order, startLat, startLng, endLat, endLng, color?}]
   * @param {WebGLRenderingContext} gl
   */
  constructor(arcsData, gl) {
    this.gl = gl;
    this.arcs = arcsData
      .slice()
      .sort((a, b) => a.order - b.order)
      .map(arc => {
        const startVec = latLngToVec3(arc.startLat, arc.startLng);
        const endVec   = latLngToVec3(arc.endLat, arc.endLng);
        const points   = generateArcPoints(startVec, endVec, 64);
        const buffer   = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, points, gl.STATIC_DRAW);

        const numPoints = points.length / 3; // 65
        const d = Math.max(-1, Math.min(1, vec3Dot(startVec, endVec)));
        const angularDist = Math.acos(d);

        // 每弧独立颜色（可选），未指定时渲染时用全局 fallback
        const color = arc.color ? parseHexColor(arc.color) : null;

        // 动画时长随距离缩放：越远越长
        const travelDuration = 1.0 + 1.2 * angularDist;
        const holdDuration   = 0.4;  // 到达终点后保持整段弧可见的时长
        const drainDuration  = 0.3 + 0.25 * angularDist;  // 消散时长随弧长增加，长弧不会“还没到就没了”
        const totalDuration  = travelDuration + holdDuration + drainDuration;

        // 尾巴长度：约 30% 的总点数
        const tailLen = Math.max(5, Math.floor(numPoints * 0.3));

        return { buffer, numPoints, color, travelDuration, holdDuration, drainDuration, totalDuration, tailLen };
      });

    this.arcDelay = 1.0; // 每条弧交错启动间隔（秒）
    const maxDur = Math.max(...this.arcs.map(a => a.totalDuration));
    this.cycleDuration = Math.max(1, this.arcs.length - 1) * this.arcDelay + maxDur + 1.0;
  }

  /**
   * 获取第 i 条弧的当前动画状态（滑动窗口）
   * @returns {null | {startIdx: number, drawCount: number, alpha: number}}
   */
  getState(arcIndex, timeSec) {
    const arc = this.arcs[arcIndex];
    const cycleTime = timeSec % this.cycleDuration;
    const arcStart  = arcIndex * this.arcDelay;
    const local     = cycleTime - arcStart;

    if (local < 0 || local > arc.totalDuration) return null;

    const N       = arc.numPoints;
    const tailLen = arc.tailLen;
    const headEnd = N - 1;

    if (local < arc.travelDuration) {
      // ── 滑行阶段：弧头从起点滑到终点 ──
      const p     = local / arc.travelDuration;
      const eased = 1 - Math.pow(1 - p, 2); // ease-out quadratic
      const head  = Math.floor(eased * headEnd);
      const tail  = Math.max(0, head - tailLen);
      const count = head - tail + 1;
      return { startIdx: tail, drawCount: Math.max(2, count), alpha: 1.0 };
    }

    const afterTravel = local - arc.travelDuration;
    if (afterTravel < arc.holdDuration) {
      // ── 到达终点后保持：整段弧（头在终点）完整显示一段时间 ──
      const tail = Math.max(0, headEnd - tailLen);
      return { startIdx: tail, drawCount: headEnd - tail + 1, alpha: 1.0 };
    }

    // ── 消散阶段：尾巴追上头 + 淡出（长弧消散更慢） ──
    const drainP = (afterTravel - arc.holdDuration) / arc.drainDuration;
    const tail   = Math.floor((headEnd - tailLen) + tailLen * drainP);
    const count  = headEnd - tail + 1;
    if (count < 2) return null;
    const alpha = Math.max(0, 1.0 - drainP);
    return { startIdx: Math.min(tail, headEnd - 1), drawCount: Math.max(2, count), alpha };
  }
}

// ───────────────────────────────────────────────
//  主入口
// ───────────────────────────────────────────────
async function main() {
  const canvas = document.getElementById('canvas');
  const gl = canvas.getContext('webgl', { antialias: false, alpha: false });
  if (!gl) {
    document.body.innerHTML = '<p style="color:#fff;text-align:center;margin-top:40vh">当前环境不支持 WebGL</p>';
    return;
  }

  const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  if (!program) return;

  // ── uniform locations ──
  const loc = {};
  ['a_position'].forEach(n => loc[n] = gl.getAttribLocation(program, n));
  [
    'u_resolution', 'u_time', 'u_texture',
    'u_phi', 'u_theta',
    'u_baseColor', 'u_glowColor', 'u_dotColor',
    'u_opacity', 'u_glowOn',
    'u_dots', 'u_dotSize', 'u_debug'
  ].forEach(n => loc[n] = gl.getUniformLocation(program, n));

  // ── 全屏四边形 ──
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1, 1,
    -1,  1,  1, -1,  1, 1
  ]), gl.STATIC_DRAW);

  // ── 纹理：从 GeoJSON 生成精确世界地图 ──
  let worldTex;
  try {
    const resp = await fetch('../globe.json');
    const geojson = await resp.json();
    worldTex = createWorldTextureFromGeoJSON(gl, geojson);
    console.log('GeoJSON texture loaded');
    console.log('[DEBUG] 在控制台输入 __GLOBE_DEBUG__=1 显示所有点(跳过纹理)');
    console.log('[DEBUG] 在控制台输入 __GLOBE_DEBUG__=2 显示UV坐标颜色');
    console.log('[DEBUG] 在控制台输入 __GLOBE_DEBUG__=3 显示纹理采样+点阵');
    console.log('[DEBUG] 在控制台输入 __GLOBE_DEBUG__=0 恢复正常模式');
  } catch (e) {
    console.error('Failed to load globe.json:', e);
    return;
  }

  // ── 弧线 program ──
  const arcProgram = createProgram(gl, ARC_VERTEX_SHADER, ARC_FRAGMENT_SHADER);
  const arcLoc = {};
  if (arcProgram) {
    arcLoc.a_arcPos  = gl.getAttribLocation(arcProgram, 'a_arcPos');
    ['u_phi', 'u_theta', 'u_aspect', 'u_arcColor', 'u_arcAlpha'].forEach(
      n => arcLoc[n] = gl.getUniformLocation(arcProgram, n)
    );
  }

  // ── 弧线数据（支持每弧独立 color，不指定则用控制面板默认色） ──
  const arcsData = [
    {
      order: 1,
      startLat: 39.9042, startLng: 116.4074,  // 北京
      endLat: 35.6762,   endLng: 139.6503,     // 东京
      color: '#ff6633',
    },
    {
      order: 2,
      startLat: 51.5074, startLng: -0.1278,    // 伦敦
      endLat: 40.7128,   endLng: -74.006,       // 纽约
      color: '#33ccff',
    },
    {
      order: 3,
      startLat: 22.3193, startLng: 114.1694,   // 香港
      endLat: 1.3521,    endLng: 103.8198,      // 新加坡
      color: '#ffcc00',
    },
    {
      order: 4,
      startLat: 48.8566, startLng: 2.3522,     // 巴黎
      endLat: 55.7558,   endLng: 37.6173,       // 莫斯科
      color: '#66ff66',
    },
    {
      order: 5,
      startLat: -33.8688, startLng: 151.2093,  // 悉尼
      endLat: -23.5505,   endLng: -46.6333,     // 圣保罗
      color: '#ff66cc',
    },
    {
      order: 6,
      startLat: 37.7749, startLng: -122.4194,  // 旧金山
      endLat: 34.0522,   endLng: -118.2437,     // 洛杉矶
    },
    {
      order: 7,
      startLat: 19.4326, startLng: -99.1332,   // 墨西哥城
      endLat: 3.1390,    endLng: 101.6869,      // 吉隆坡
    },
  ];
  const arcAnimator = arcProgram ? new ArcAnimator(arcsData, gl) : null;

  // ── 自动旋转（无拖拽） ──
  let phi = 0.3, theta = 0.15;

  // ── 颜色配置（可被 UI 修改） ──
  let baseColor = [0.06, 0.1, 0.2];
  let glowColor = [0.15, 0.4, 0.85];
  let dotColor  = [0.4, 0.8, 1.0];

  // ── 辅助：hex → [r,g,b] 归一化 ──
  function hexToRgb(hex) {
    const n = parseInt(hex.replace('#', ''), 16);
    return [(n >> 16 & 255) / 255, (n >> 8 & 255) / 255, (n & 255) / 255];
  }
  function rgbToHex(rgb) {
    return '#' + rgb.map(c => Math.round(c * 255).toString(16).padStart(2, '0')).join('');
  }

  // ── UI 控件绑定 ──
  const densitySlider  = document.getElementById('densitySlider');
  const dotSizeSlider  = document.getElementById('dotSizeSlider');
  const densityValue   = document.getElementById('densityValue');
  const dotSizeValue   = document.getElementById('dotSizeValue');

  const dotColorPicker  = document.getElementById('dotColorPicker');
  const baseColorPicker = document.getElementById('baseColorPicker');
  const glowColorPicker = document.getElementById('glowColorPicker');

  // 初始化颜色选择器的值
  dotColorPicker.value  = rgbToHex(dotColor);
  baseColorPicker.value = rgbToHex(baseColor);
  glowColorPicker.value = rgbToHex(glowColor);

  let dots    = parseFloat(densitySlider.value);
  let dotSize = parseFloat(dotSizeSlider.value);

  densitySlider.addEventListener('input', () => {
    dots = parseFloat(densitySlider.value);
    densityValue.textContent = dots;
  });
  dotSizeSlider.addEventListener('input', () => {
    dotSize = parseFloat(dotSizeSlider.value);
    dotSizeValue.textContent = dotSize.toFixed(3);
  });
  dotColorPicker.addEventListener('input', () => {
    dotColor = hexToRgb(dotColorPicker.value);
  });
  baseColorPicker.addEventListener('input', () => {
    baseColor = hexToRgb(baseColorPicker.value);
  });
  glowColorPicker.addEventListener('input', () => {
    glowColor = hexToRgb(glowColorPicker.value);
  });

  const glowToggle = document.getElementById('glowToggle');
  let glowOn = glowToggle.checked ? 1.0 : 0.0;
  glowToggle.addEventListener('change', () => {
    glowOn = glowToggle.checked ? 1.0 : 0.0;
  });

  // ── 弧线颜色 ──
  let arcColor = [1.0, 0.4, 0.2]; // 橙色
  const arcColorPicker = document.getElementById('arcColorPicker');
  if (arcColorPicker) {
    arcColorPicker.value = rgbToHex(arcColor);
    arcColorPicker.addEventListener('input', () => {
      arcColor = hexToRgb(arcColorPicker.value);
    });
  }

  // ── resize ──
  function resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
      gl.viewport(0, 0, w, h);
    }
  }

  // ── 渲染循环 ──
  function render(t) {
    resize();
    const timeSec = t * 0.001;

    // 自动旋转
    phi += 0.003;

    // ─── Pass 1: 绘制球体 ───
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc.a_position);
    gl.vertexAttribPointer(loc.a_position, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, worldTex);

    gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);
    gl.uniform1f(loc.u_time, timeSec);
    gl.uniform1i(loc.u_texture, 0);
    gl.uniform1f(loc.u_phi, phi);
    gl.uniform1f(loc.u_theta, theta);
    gl.uniform3fv(loc.u_baseColor, baseColor);
    gl.uniform3fv(loc.u_glowColor, glowColor);
    gl.uniform3fv(loc.u_dotColor, dotColor);
    gl.uniform1f(loc.u_opacity, 0.9);
    gl.uniform1f(loc.u_glowOn, glowOn);
    gl.uniform1f(loc.u_dots, dots);
    gl.uniform1f(loc.u_dotSize, dotSize);
    gl.uniform1f(loc.u_debug, window.__GLOBE_DEBUG__ || 0.0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);

    // ─── Pass 2: 绘制弧线（滑动线段 + 每弧独立颜色） ───
    if (arcProgram && arcAnimator) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(arcProgram);
      const aspect = canvas.width / canvas.height;

      gl.uniform1f(arcLoc.u_phi, phi);
      gl.uniform1f(arcLoc.u_theta, theta);
      gl.uniform1f(arcLoc.u_aspect, aspect);

      // 尝试设置线宽（macOS 通常支持）
      try { gl.lineWidth(2.0); } catch (e) { /* 忽略不支持的平台 */ }

      for (let i = 0; i < arcAnimator.arcs.length; i++) {
        const state = arcAnimator.getState(i, timeSec);
        if (!state) continue;

        // 每弧独立颜色，未指定时回退到控制面板默认色
        const color = arcAnimator.arcs[i].color || arcColor;
        gl.uniform3fv(arcLoc.u_arcColor, color);
        gl.uniform1f(arcLoc.u_arcAlpha, state.alpha);

        gl.bindBuffer(gl.ARRAY_BUFFER, arcAnimator.arcs[i].buffer);
        gl.enableVertexAttribArray(arcLoc.a_arcPos);
        gl.vertexAttribPointer(arcLoc.a_arcPos, 3, gl.FLOAT, false, 0, 0);

        gl.drawArrays(gl.LINE_STRIP, state.startIdx, state.drawCount);
      }

      gl.disable(gl.BLEND);
    }

    requestAnimationFrame(render);
  }

  resize();
  requestAnimationFrame(render);
}

main();
