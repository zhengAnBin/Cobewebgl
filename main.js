/**
 * Cobe 风格高性能点阵地球
 * 参考：https://juejin.cn/post/7601355703240458280
 */

const VERTEX_SHADER = `
  attribute vec2 a_position;
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
  }
`;

const FRAGMENT_SHADER = `
  precision highp float;

  uniform vec2 u_resolution;
  uniform float u_time;
  uniform sampler2D u_texture;

  // 可调参数（与 Cobe 类似）
  uniform float u_phi;       // 绕 Y 轴旋转
  uniform float u_theta;     // 绕 X 轴旋转
  uniform vec3 u_baseColor;
  uniform vec3 u_glowColor;
  uniform float u_dark;      // 陆地与海洋对比
  uniform float u_dotsBrightness;
  uniform float u_opacity;

  const float PI = 3.14159265359;
  const float kPhi = 1.618033988749895;      // 黄金分割比
  const float phiMinusOne = 0.618033988749895; // 1/phi
  const float sqrt5 = 2.23606797749979;
  const float kTau = 6.283185307179586;      // 2*PI
  const float twoPiOnPhi = 3.883222077450933; // 2*PI/phi
  const float dots = 1800.0;                  // 点数量级
  const float byDots = 1.0 / dots;
  const float byLogPhiPlusOne = 1.0 / log2(kPhi + 1.0);

  mat3 rotateX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1., 0., 0., 0., c, -s, 0., s, c);
  }
  mat3 rotateY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c, 0., s, 0., 1., 0., -s, 0., c);
  }

  vec3 nearestFibonacciLattice(vec3 p, out float m) {
    p = p.xzy;

    float k = max(2., floor(log2(sqrt5 * dots * PI * (1. - p.z * p.z)) * byLogPhiPlusOne));
    vec2 f = floor(pow(kPhi, k) / sqrt5 * vec2(1., kPhi) + .5);
    vec2 br1 = fract((f + 1.) * phiMinusOne) * kTau - twoPiOnPhi;
    vec2 br2 = -2. * f;

    vec2 sp = vec2(atan(p.y, p.x), p.z - 1.);

    float denom = br1.x * br2.y - br2.x * br1.y;
    vec2 c = floor(vec2(br2.y * sp.x - br1.y * (sp.y * dots + 1.), -br2.x * sp.x + br1.x * (sp.y * dots + 1.)) / denom);

    float mindist = PI;
    vec3 minip = vec3(0., 0., 1.);

    for (float s = 0.; s < 4.; s += 1.) {
      vec2 o = vec2(mod(s, 2.), floor(s * .5));
      float idx = dot(f, c + o);
      if (idx > dots || idx < 0.) continue;

      float fracV = fract(idx * phiMinusOne);
      float theta = fract(fracV) * kTau;

      float cosphi = 1. - 2. * idx * byDots;
      float sinphi = sqrt(1. - cosphi * cosphi);
      vec3 sample2 = vec3(cos(theta) * sinphi, sin(theta) * sinphi, cosphi);

      float dist = length(p - sample2);
      if (dist < mindist) {
        mindist = dist;
        minip = sample2;
      }
    }

    m = mindist;
    return minip.xzy;
  }

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
      float z = sqrt(rSquared - d2);
      vec3 nor = normalize(vec3(p.x, p.y, z));
      vec3 rawNor = nor;

      nor = rotateY(u_phi) * rotateX(u_theta) * nor;

      vec3 light = vec3(0., 0., 1.);
      float dotNL = dot(rawNor, light);

      float dis;
      vec3 gP = nearestFibonacciLattice(nor, dis);

      float gPhi = asin(gP.y);
      float gTheta = atan(gP.z, gP.x);
      vec2 dotUV = vec2(0.5 + gTheta / kTau, 0.5 + gPhi / PI);
      float mapColor = step(0.5, texture2D(u_texture, dotUV).r);
      float v = smoothstep(0.008, 0.0, dis);

      float lighting = pow(dotNL, 3.0) * u_dotsBrightness;
      float sample2 = mapColor * v * lighting;
      float colorFactor = mix((1. - sample2) * pow(dotNL, 0.4), sample2, u_dark) + 0.1;
      color = vec4(u_baseColor * colorFactor, 1.);

      color.xyz += pow(1. - dotNL, 4.) * u_glowColor;
      color *= (1. + u_opacity) * 0.5;

      glowFactor = pow(dot(normalize(vec3(-uv, sqrt(1. - l))), vec3(0., 0., 1.)), 4.) * smoothstep(0., 1., .2 / (l - rSquared));
    } else {
      float outD = sqrt(0.2 / (l - rSquared));
      glowFactor = smoothstep(0.5, 1., outD / (outD + 1.));
    }

    gl_FragColor = color + vec4(glowFactor * u_glowColor, glowFactor);
  }
`;

function createProgram(gl, vsSrc, fsSrc) {
  const vs = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vs, vsSrc);
  gl.compileShader(vs);
  if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(vs));
    gl.deleteShader(vs);
    return null;
  }

  const fs = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fs, fsSrc);
  gl.compileShader(fs);
  if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(fs));
    gl.deleteShader(fs);
    gl.deleteShader(vs);
    return null;
  }

  const program = gl.createProgram();
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  return program;
}

function createWorldTexture(gl) {
  const W = 128;
  const H = 64;
  const data = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const u = x / W;
      const v = y / H;
      let land = 0;
      if (v > 0.25 && v < 0.75) {
        if (u > 0.1 && u < 0.35) land = 1;
        if (u > 0.38 && u < 0.62) land = 1;
        if (u > 0.65 && u < 0.92) land = 1;
      }
      if (v > 0.35 && v < 0.65 && u > 0.72 && u < 0.88) land = 1;
      const g = land * 255;
      const i = (y * W + x) * 4;
      data[i] = g;
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

function main() {
  const canvas = document.getElementById('canvas');
  const gl = canvas.getContext('webgl', { antialias: false });
  if (!gl) {
    document.body.innerHTML = '<p>当前环境不支持 WebGL</p>';
    return;
  }

  const program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
  if (!program) return;

  const posLoc = gl.getAttribLocation(program, 'a_position');
  const resolutionLoc = gl.getUniformLocation(program, 'u_resolution');
  const timeLoc = gl.getUniformLocation(program, 'u_time');
  const textureLoc = gl.getUniformLocation(program, 'u_texture');
  const phiLoc = gl.getUniformLocation(program, 'u_phi');
  const thetaLoc = gl.getUniformLocation(program, 'u_theta');
  const baseColorLoc = gl.getUniformLocation(program, 'u_baseColor');
  const glowColorLoc = gl.getUniformLocation(program, 'u_glowColor');
  const darkLoc = gl.getUniformLocation(program, 'u_dark');
  const dotsBrightnessLoc = gl.getUniformLocation(program, 'u_dotsBrightness');
  const opacityLoc = gl.getUniformLocation(program, 'u_opacity');

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);

  const worldTex = createWorldTexture(gl);

  let phi = 0.3;
  let theta = 0.4;
  let lastX = 0, lastY = 0;
  let dragging = false;

  canvas.addEventListener('mousedown', (e) => {
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    phi += (e.clientX - lastX) * 0.005;
    theta = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, theta + (e.clientY - lastY) * 0.005));
    lastX = e.clientX;
    lastY = e.clientY;
  });
  canvas.addEventListener('mouseup', () => { dragging = false; });
  canvas.addEventListener('mouseleave', () => { dragging = false; });

  const baseColor = [0.2, 0.45, 0.7];
  const glowColor = [0.2, 0.5, 0.9];

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

  function render(t) {
    resize();
    if (!phiLoc) return;
    gl.useProgram(program);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(posLoc);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, worldTex);

    gl.uniform2f(resolutionLoc, canvas.width, canvas.height);
    gl.uniform1f(timeLoc, t * 0.001);
    gl.uniform1i(textureLoc, 0);
    gl.uniform1f(phiLoc, phi);
    gl.uniform1f(thetaLoc, theta);
    gl.uniform3fv(baseColorLoc, baseColor);
    gl.uniform3fv(glowColorLoc, glowColor);
    gl.uniform1f(darkLoc, 0.9);
    gl.uniform1f(dotsBrightnessLoc, 1.2);
    gl.uniform1f(opacityLoc, 0.8);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
    requestAnimationFrame(render);
  }

  resize();
  requestAnimationFrame(render);
}

main();
