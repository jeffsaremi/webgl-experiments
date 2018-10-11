const globalProgramCache = new Map();

function createCanvas(canvasWidth, canvasHeight) {
  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  return canvas;
}

function getContext(canvas) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    depth: false,
    antialias: false,
    stencil: false,
    preserveDrawingBuffer: false,
    premultipliedAlpha: false,
    powerPreference: 'high-performance'
  });
  if (!gl.getExtension('EXT_color_buffer_float')) {
    throw 'Floating point extension is not supported';
  }
  gl.disable(gl.DEPTH_TEST);
  gl.disable(gl.STENCIL_TEST);
  gl.disable(gl.BLEND);
  gl.disable(gl.DITHER);
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.disable(gl.SAMPLE_COVERAGE);
  gl.disable(gl.SCISSOR_TEST);
  //gl.enable(gl.CULL_FACE);
  //gl.cullFace(gl.BACK);
  return gl;
}

function getDefaultGeometry() {
  return new Float32Array([
    -1.0, 1.0,  0.0, 0.0, 1.0,  // upper left
    -1.0, -1.0, 0.0, 0.0, 0.0,  // lower left
    1.0,  1.0,  0.0, 1.0, 1.0,  // upper right
    1.0,  -1.0, 0.0, 1.0, 0.0   // lower right
  ]);
}
function setupVBO(gl) {
  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, getDefaultGeometry(), gl.STATIC_DRAW);
  const positionHandle = 0;
  const textureCoordHandle = 1;
  gl.enableVertexAttribArray(positionHandle);
  gl.enableVertexAttribArray(textureCoordHandle);
  gl.vertexAttribPointer(positionHandle, 3, gl.FLOAT, gl.FALSE, 20, 0);
  gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, gl.FALSE, 20, 12);
  return vbo;
}
function createTexture(gl, internalFormat, format, type, width, height, data) {
  // Create the texture
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(
      gl.TEXTURE_2D,
      0,  // 0 - no mipmaps
      internalFormat, width, height,
      0,  // Always 0 in OpenGL ES.
      format, type, data);
  checkError(gl);
  return texture;
}
function createFrameBuffer(gl) {
  const frameBuffer = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, frameBuffer);
  return frameBuffer;
}
function attachOutputTexture(gl, texture) {
  gl.activeTexture(
      gl.TEXTURE0 +
      (32 - 1));  // TODO: replace hardcoded 32 with MAX_TEXTURE_UNITS Value
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
      0);  // 0, we aren't using MIPMAPs
  const status = frameBufferIsComplete(gl);
  if (!status.isComplete) {
    throw status.message;
  }
}
function frameBufferIsComplete(gl) {
  let message;
  let status;
  let value;

  status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);

  switch (status) {
    case gl.FRAMEBUFFER_COMPLETE:
      message = 'gl.FRAMEBUFFER_COMPLETE';
      value = true;
      break;
    case gl.FRAMEBUFFER_UNSUPPORTED:
      message = 'gl.FRAMEBUFFER_UNSUPPORTED';
      value = false;
      break;
    case gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      message = 'gl.FRAMEBUFFER_INCOMPLETE_ATTACHMENT';
      value = false;
      break;
    case gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS:
      message = 'gl.FRAMEBUFFER_INCOMPLETE_DIMENSIONS';
      value = false;
      break;
    case gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      message = 'gl.FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT';
      value = false;
      break;
    default:
      message = 'Unknown framebuffer status: ' + status;
      value = false;
  }
  return {isComplete: value, message: message};
}
function compileShader(gl, shaderSource, shaderType) {
  let shader = gl.createShader(shaderType);
  gl.shaderSource(shader, shaderSource);
  gl.compileShader(shader);

  let success = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
  if (!success) {
    console.log(shaderSource);
    throw `Shader compilation failed: ${gl.getShaderInfoLog(shader)}`;
  }
  return shader;
}
function getDefaultVertexShader(gl) {
  const vertexShaderSource = `#version 300 es
                          layout (location = 0) in vec3 position;
                          layout (location = 1) in vec2 texCoord;
                          out vec2 TexCoord;
  
                          void main()
                          {
                           gl_Position = vec4(position, 1.0f);
                           TexCoord = texCoord;
                          }`;

  return compileShader(gl, vertexShaderSource, gl.VERTEX_SHADER);
}
function checkError(gl) {
  const error = gl.getError();
  let label = '';
  switch (error) {
    case (gl.NO_ERROR):
      return;
    case (gl.INVALID_ENUM):
      label = 'INVALID_ENUM';
      break;
    case (gl.INVALID_VALUE):
      label = 'INVALID_VALUE';
      break;
    case (gl.INVALID_OPERATION):
      label = 'INVALID_OPERATION';
      break;
    case (gl.INVALID_FRAMEBUFFER_OPERATION):
      label = 'INVALID_FRAMEBUFFER_OPERATION';
      break;
    case (gl.OUT_OF_MEMORY):
      label = 'OUT_OF_MEMORY';
      break;
    case (gl.CONTEXT_LOST_WEBGL):
      label = 'CONTEXT_LOST_WEBGL';
      break;
    default:
      label = 'Unknown WebGL Error: ' + error.toString(16);
  }
  throw label;
}
function getProgram(key) {
  return globalProgramCache.get(key);
}
function cacheProgram(key, program) {
  globalProgramCache.set(key, program);
}
function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (gl.getProgramParameter(program, gl.LINK_STATUS) === false) {
    throw `gl.getProgramInfoLog: ${gl.getProgramInfoLog(program)}`;
  }
  return program;
}
function getAttribLocation(gl, program, name) {
  const attributeLocation = gl.getAttribLocation(program, name);
  if (attributeLocation === -1) {
    throw `Can not find attribute ${name}`;
  }
  return attributeLocation;
}
function getUniformLocation(program, name) {
  const reference = gl.getUniformLocation(program, name);
  if (reference === -1) {
    throw `Can not find uniform ${name}`;
  }
  return reference;
}
function bindInputTexture(gl, program, tex, name, index) {
  const handle = gl.getUniformLocation(program, name);
  gl.activeTexture(gl.TEXTURE0 + index);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(handle, index);
}
async function waitForSync(gl) {
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
  let status = gl.clientWaitSync(sync, 0, 0);

  while (status !== gl.CONDITION_SATISFIED && status !== gl.ALREADY_SIGNALED) {
      await new Promise(resolve => setTimeout(resolve, 1));
      status = gl.clientWaitSync(sync, 0, 0);
  }
  gl.deleteSync(sync);
}
function readOutput(gl, width, height, format, type, buffer) {
  gl.readPixels(0, 0, width, height, format, type, buffer);
}
function debugPrintTexture(gl, texture, width, height, format, type) {
  createFrameBuffer(gl);
  attachOutputTexture(gl, texture);
  const buffer = new Float32Array(width*height * (format === gl.RGBA ? 4 : 1));
  readOutput(gl, width, height, format, type, buffer);
  console.debug(buffer);
}
function computeStrides(shape) {
  const rank = shape.length;
  if (rank < 2) {
    return [1];
  }

  const strides = new Array(rank);
  strides[rank - 1] = 1;
  strides[rank - 2] = shape[rank - 1];
  for (let i = rank - 3; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}
function computeTextureShape(shape, strides) {
  const maxTextureSize = 16000;
  const effectiveShape = shape.filter(i => i > 1);
  if (!strides) {
    strides = computeStrides(shape);
  }
  const totalSize = Math.ceil(shape.reduce((a, b) => a * b));

  let width = Math.floor(Math.sqrt(totalSize));

  for (; width < maxTextureSize && width < totalSize; width++) {
    if (totalSize % width === 0) {
      break;
    }
  }

  if (width >= maxTextureSize || totalSize % width !== 0) {
    throw 'The given dimensions are outside this GPU\'s boundaries';
  }
  return [width, totalSize / width];
}
function compareOutputs(actual, expected, epsilon) {
  if (actual.length !== expected.length) {
    console.error('lengths did not match');
    return false;
  }

  for (let i = actual.length - 1; i >= 0; i--) {
    const a = actual[i], b = expected[i];

    // check for NaN
    //
    if (Number.isNaN(a) && Number.isNaN(b)) {
      continue;  // 2 numbers are NaN, treat as equal
    }
    if (Number.isNaN(a) || Number.isNaN(b)) {
      console.error('NaN');
      return false;  // one is NaN and the other is not
    }
    // sign should be same if not equals to zero
    //
    if ((a > 0 && b < 0) || (a < 0 && b > 0)) {
      console.error('sign mismatch');
      return false;  // sign is different
    }
    if (Math.abs(actual[i] - expected[i]) < epsilon) {
      continue;  // absolute error check pass
    }
    if (a !== 0 && b !== 0 && a / b < epsilon && b / a < epsilon) {
      continue;  // relative error check pass
    }

    console.error(`${a} !== ${b}`);
    return false;
  }
  return true;
}
function getGlslIndicesToOffset(name, textureData) {
  let block = '';
  for (let i = textureData.dims.length - 1; i >= 0; --i) {
    block += `
      offset += indices[${i}] * ${textureData.strides[i]};
      `;
  }
  return `
    int indicesToOffset_${name}(int indices[${textureData.dims.length}]) {
      int offset = 0;
      ${block}
      return offset;
    }
    `;
}
function getGlslOffsetToCoords() {
  return `
    ivec2 offsetToCoords(int offset, int width) {
      int t = offset / width;
      int s = offset - t*width;
      return ivec2(s,t);
    }`;
}
function getGlslAccessor(name, textureData) {
  return `
    ${getGlslIndicesToOffset(name, textureData)}
    float _${name}(int m[${textureData.dims.length}]) {
      int offset = indicesToOffset_${name}(m);
      ivec2 coords = offsetToCoords(offset, ${textureData.width});
      return texelFetch(${name}, coords, 0).r;
    }
    `;
}
function getGlslAccessorVec(name, textureData) {
  return `
    ${getGlslIndicesToOffset(name, textureData)}
    vec4 _${name}Vec(int m[${textureData.dims.length}]) {
      int offset = indicesToOffset_${name}(m);
      ivec2 coords = offsetToCoords(offset, ${textureData.width});
      return texelFetch(${name}, coords, 0);
    }
    `;
}
function getGlslOffsetToIndices(name, textureData) {
  const strides = textureData.strides;
  const rank = strides.length;
  const stridesBlock = [];
  for (let i = 0; i < rank - 1; ++i) {
    stridesBlock.push(`
    indices[${i}] = offset / ${strides[i]};`);
    stridesBlock.push(`
      offset -= indices[${i}] * ${strides[i]};`);
  }
  stridesBlock.push(`
    indices[${rank - 1}] = offset;`);
  return `
    void offsetToIndices_${name}(int offset, out int indices[${rank}]) {
      ${stridesBlock.join('')}
    }
    `;
}
function glslCoordsToOutputIndices(textureData) {
  const rank = textureData.dims.length;
  const strides = textureData.strides;
  const width = textureData.width;
  const height = textureData.height;

  const stridesBlock = [];
  for (let i = 0; i < rank - 1; ++i) {
    stridesBlock.push(`
      c[${i}] = offset / ${strides[i]};`);
    stridesBlock.push(`
      offset -= c[${i}] * ${strides[i]};`);
  }
  stridesBlock.push(`
      c[${rank - 1}] = offset;`);

  return `
    void toIndices(vec2 coords, out int c[${rank}]) {
      int offset = int(coords.t * ${height}.0) * ${width} + int(coords.s * ${width}.0);
      ${stridesBlock.join('')}
    }
  `;
}
function createTextureData(gl, channels, dataType, dims, data) {
  const strides = computeStrides(dims);
  const [width, height] = computeTextureShape(dims, strides);
  const internalFormat = channels === 1 ? gl.R32F : gl.RGBA32F; // ignoring data type
  const format = channels === 1 ? gl.RED : gl.RGBA;
  const texture = createTexture(gl, internalFormat, format, gl.FLOAT, width, height, data);
  return {
    internalFormat: internalFormat,
    format: format,
    type: dataType,
    width: width,
    height: height,
    texture: texture,
    channels: channels,
    dims: dims,
    strides: strides
  };
}
function createTextureData2(gl, channels, dataType, dims, data, width, height) {
  const strides = computeStrides(dims);
  const internalFormat = channels === 1 ? gl.R32F : gl.RGBA32F; // ignoring data type
  const format = channels === 1 ? gl.RED : gl.RGBA;
  const texture = createTexture(gl, internalFormat, format, gl.FLOAT, width, height, data);
  return {
    internalFormat: internalFormat,
    format: format,
    type: dataType,
    width: width,
    height: height,
    texture: texture,
    channels: channels,
    dims: dims,
    strides: strides
  };
}
function glslTiledCoordsToOriginal() {
  return `
    vec2 tiledCoordsToOriginal(vec2 coords, int width, int height, int tileLength, int originalWidth, int originalHeight) {
      int tl = tileLength;
      int ts = tl*tl; // Tile Size

      int x = int(coords.s * float(width));
      int y = int(coords.t * float(height));
      int ti = x / tl + (y / tl) * (width / tl); // Tile Index
      int offset = ti * ts;
      int ty = y - (y/tl)*tl;
      int tx = x - (x/tl)*tl;
      offset += tx + (ty * tl);

      return offsetToCoord(offset, originalWidth, originalHeight);
    }
    `;
}
function glslOriginalCoordsToTiled() {
  return `
    vec2 originalCoordsToTiled(vec2 coords, int width, int height, int tileLength, int tiledWidth, int tiledHeight) {
      int tl = tileLength;
      int ts = tl*tl; // Tile Size

      int offset = coordsToOffset(coords, width, height);
      int ti = offset / ts; // Tile Index
      int inTileOffset = offset - (ti * ts);
      int ty = inTileOffset / tl;
      int tx = inTileOffset - ty * tl);
      int y = ti / tiledWidth + ty;
      int x = ti - y * tiledWidth + tx;
      return (vec2(x,y) + vec2(0.5,0.5)) / vec2(tileWidth, tileHeight);
    }
    `;
}
function glslGetTexel() {
  return `
    vec4 texel(sampler2D texture, ivec2 xy, int width , int height) {
      vec2 coords = (vec2(xy.x, xy.y) + vec2(0.5,0.5)) / vec2(width, height);
      return texture2D(texture, coords);
    }
  `;
}
function glslTiledCoordsToOriginal() {
  return `
    vec2 tiledCoordsToOriginal(vec2 coords, int width, int height, int tileLength, int originalWidth, int originalHeight) {
      int tl = tileLength;
      int ts = tl*tl; // Tile Size
      ivec2 xy = ivec2(coords * vec2(width, height));
      int tileRows = (xy.y / tl);
      int tileColumns = (xy.x / tl);
      int ty = xy.y - tileRows * tl;
      int tx = xy.x - tileColumns * tl;
      int tilePerRow = width / tl;
      int ti = tileRows * tilePerRow + tileColumns; // Tile Index
      int offset = ti * ts + ty * tl + tx;

      return offsetToCoords(offset, originalWidth, originalHeight);
    }
    `;
}
function glslOriginalCoordsToTiled() {
  return `
    vec2 originalCoordsToTiled(vec2 coords, int width, int height, int tileLength, int tiledWidth, int tiledHeight) {
      int tl = tileLength;
      int ts = tl*tl; // Tile Size

      int offset = coordsToOffset(coords, width, height);
      int ti = offset / ts; // Tile Index
      int inTileOffset = offset - (ti * ts);
      int ty = inTileOffset / tl;
      int tx = inTileOffset - (ty * tl);
      int tilePerRow = tiledWidth / tl;
      int tileRows = ti / tilePerRow;
      int x = (ti - tileRows * tilePerRow) * tl + tx;
      int y = tileRows * tl + ty;
      return (vec2(x,y) + vec2(0.5,0.5)) / vec2(tiledWidth, tiledHeight);
    }
    `;
}
function glslOffsetToCoords() {
  return `
    vec2 offsetToCoords(int offset, int width, int height) {
      int t = offset / width;
      int s = offset - t*width;
      vec2 coords = (vec2(s,t) + vec2(0.5,0.5)) / vec2(width, height);
      return coords;
    }
    `;
}
function glslCoordsToOffset() {
  return `
    int coordsToOffset(vec2 coords, int width, int height) {
      vec2 st = coords * vec2(width, height);
      int offset = int(st.t) * width + int(st.s);
      return offset;
    }
    `;
}
function createRandomArray(size) {
  return new Float32Array(Array.from({length: size}, (v,k) => k % 10));
}