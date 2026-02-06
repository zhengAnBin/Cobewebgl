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

      // 最近点的球面 UV → 纹理采样
      float gPhi   = asin(gP.y);
      float gTheta = atan(gP.z, gP.x);
      vec2 dotUV   = vec2(0.5 + gTheta / kTau, 0.5 + gPhi / PI);
      float mapColor = step(0.5, texture2D(u_texture, dotUV).r);

      // ★ 用 u_dotSize 控制点的大小（硬边实心圆点） ★
      float v = step(dis, u_dotSize);

      // 点阵遮罩：陆地上的点
      float dotMask = mapColor * v;

      // ★ 无光照：直接用颜色混合 ★
      // 点阵区域用 u_dotColor，其余用 u_baseColor
      vec3 surfaceColor = mix(u_baseColor, u_dotColor, dotMask);
      color = vec4(surfaceColor * u_opacity, 1.);

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
 * 生成简易世界地图纹理（0/1 二值：陆地 / 海洋）
 * 采用多段简易多边形近似各大洲轮廓，128×64 分辨率，约 1KB
 */
function createWorldTexture(gl) {
  const W = 128, H = 64;
  const data = new Uint8Array(W * H * 4);

  // 辅助：判断 (u,v) 是否在某个矩形区域内 [u0,u1]x[v0,v1]
  function rect(u, v, u0, v0, u1, v1) {
    return u >= u0 && u <= u1 && v >= v0 && v <= v1;
  }

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W;     // 经度 0-1
      const v = 1 - y / H; // 纬度 0-1（南→北）
      let land = 0;

      // ── 北美 ──
      if (rect(u, v, 0.08, 0.55, 0.25, 0.85)) land = 1;
      if (rect(u, v, 0.12, 0.50, 0.22, 0.55)) land = 1;
      if (rect(u, v, 0.15, 0.45, 0.21, 0.50)) land = 1;
      if (rect(u, v, 0.06, 0.60, 0.12, 0.80)) land = 1;  // 阿拉斯加
      if (rect(u, v, 0.10, 0.80, 0.22, 0.90)) land = 1;  // 加拿大北部

      // ── 中美 ──
      if (rect(u, v, 0.15, 0.42, 0.20, 0.50)) land = 1;

      // ── 南美 ──
      if (rect(u, v, 0.20, 0.18, 0.30, 0.42)) land = 1;
      if (rect(u, v, 0.22, 0.10, 0.28, 0.18)) land = 1;

      // ── 欧洲 ──
      if (rect(u, v, 0.45, 0.62, 0.55, 0.78)) land = 1;
      if (rect(u, v, 0.42, 0.65, 0.48, 0.72)) land = 1;  // 西欧
      if (rect(u, v, 0.48, 0.58, 0.53, 0.65)) land = 1;  // 地中海

      // ── 非洲 ──
      if (rect(u, v, 0.45, 0.30, 0.56, 0.58)) land = 1;
      if (rect(u, v, 0.47, 0.22, 0.54, 0.30)) land = 1;  // 南非

      // ── 亚洲 ──
      if (rect(u, v, 0.55, 0.55, 0.78, 0.80)) land = 1;  // 俄罗斯/中亚
      if (rect(u, v, 0.58, 0.45, 0.72, 0.58)) land = 1;  // 中国/印度
      if (rect(u, v, 0.55, 0.42, 0.62, 0.50)) land = 1;  // 中东
      if (rect(u, v, 0.68, 0.48, 0.75, 0.60)) land = 1;  // 日本/朝鲜
      if (rect(u, v, 0.60, 0.80, 0.78, 0.92)) land = 1;  // 西伯利亚

      // ── 东南亚 ──
      if (rect(u, v, 0.65, 0.38, 0.72, 0.48)) land = 1;

      // ── 大洋洲/澳大利亚 ──
      if (rect(u, v, 0.72, 0.18, 0.82, 0.32)) land = 1;
      if (rect(u, v, 0.84, 0.14, 0.88, 0.22)) land = 1;  // 新西兰

      // ── 格陵兰 ──
      if (rect(u, v, 0.28, 0.78, 0.35, 0.90)) land = 1;

      const g = land * 255;
      const i = (y * W + x) * 4;
      data[i]     = g;
      data[i + 1] = g;
      data[i + 2] = g;
      data[i + 3] = 255;
    }
  }

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

// ───────────────────────────────────────────────
//  主入口
// ───────────────────────────────────────────────
function main() {
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
    'u_dots', 'u_dotSize'
  ].forEach(n => loc[n] = gl.getUniformLocation(program, n));

  // ── 全屏四边形 ──
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1, 1,
    -1,  1,  1, -1,  1, 1
  ]), gl.STATIC_DRAW);

  // ── 纹理 ──
  const worldTex = createWorldTexture(gl);

  // ── 交互：拖拽旋转 ──
  let phi = 0.3, theta = 0.15;
  let autoRotate = true;
  let lastX = 0, lastY = 0, dragging = false;

  canvas.addEventListener('mousedown', e => {
    dragging = true;
    autoRotate = false;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('mousemove', e => {
    if (!dragging) return;
    phi += (e.clientX - lastX) * 0.005;
    theta = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, theta + (e.clientY - lastY) * 0.005));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });

  // 触控支持
  canvas.addEventListener('touchstart', e => {
    if (e.touches.length === 1) {
      dragging = true;
      autoRotate = false;
      lastX = e.touches[0].clientX;
      lastY = e.touches[0].clientY;
    }
  });
  canvas.addEventListener('touchmove', e => {
    if (!dragging || e.touches.length !== 1) return;
    e.preventDefault();
    phi += (e.touches[0].clientX - lastX) * 0.005;
    theta = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, theta + (e.touches[0].clientY - lastY) * 0.005));
    lastX = e.touches[0].clientX;
    lastY = e.touches[0].clientY;
  }, { passive: false });
  canvas.addEventListener('touchend', () => { dragging = false; });

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

    // 自动旋转
    if (autoRotate) {
      phi += 0.003;
    }

    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(loc.a_position);
    gl.vertexAttribPointer(loc.a_position, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, worldTex);

    gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);
    gl.uniform1f(loc.u_time, t * 0.001);
    gl.uniform1i(loc.u_texture, 0);
    gl.uniform1f(loc.u_phi, phi);
    gl.uniform1f(loc.u_theta, theta);
    gl.uniform3fv(loc.u_baseColor, baseColor);
    gl.uniform3fv(loc.u_glowColor, glowColor);
    gl.uniform3fv(loc.u_dotColor, dotColor);
    gl.uniform1f(loc.u_opacity, 0.9);
    gl.uniform1f(loc.u_glowOn, glowOn);

    // ★ 传入用户可调参数 ★
    gl.uniform1f(loc.u_dots, dots);
    gl.uniform1f(loc.u_dotSize, dotSize);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
  }

  resize();
  requestAnimationFrame(render);
}

main();
