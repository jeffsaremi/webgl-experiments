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
function createTexStorage(gl, internalFormat, format, type, width, height, data) {
  // Create the texture
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texStorage2D(
    gl.TEXTURE_2D,
    1,  // number of texture levels: set to 1 for no mipmaps
    internalFormat, width, height);
  checkError(gl);
  if (data) {
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, format, type, data);
    checkError(gl);
  }
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
  const vertexShaderSource = `        
    precision highp float;
    attribute vec3 position;
    attribute vec2 textureCoord;

    varying vec2 TexCoords;

    void main()
    {
        gl_Position = vec4(position, 1.0);
        TexCoords = textureCoord;
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

function texel(buffer, xy, width, height) {
  const offset = xy[1] * width + xy[0];
  return buffer[offset];
}
// function sim2by2DotProd(leftBuffer, rightBuffer, leftCoords, rightCoords, leftWidth, leftHeight, rightWidth, rightHeight) {
//   let sum = 0.0;
//   // console.log(`leftBuffer: 
//   // ${leftCoords},
//   // ${[leftCoords[0] + 1, leftCoords[1]]},
//   // ${[leftCoords[0], leftCoords[1] + 1]},
//   // ${[leftCoords[0] + 1, leftCoords[1] + 1]}
//   // `);
//   // console.log(`rightBuffer: 
//   // ${rightCoords},
//   // ${[rightCoords[0] + 1, rightCoords[1]]},
//   // ${[rightCoords[0], rightCoords[1] + 1]},
//   // ${[rightCoords[0] + 1, rightCoords[1] + 1]}
//   // `);
//   sum += texel(leftBuffer, leftCoords, leftWidth, leftHeight)            
//       * texel(rightBuffer, rightCoords, rightWidth, rightHeight);
//   sum += texel(leftBuffer, [leftCoords[0] + 1, leftCoords[1]], leftWidth, leftHeight) 
//       * texel(rightBuffer, [rightCoords[0] + 1, rightCoords[1]], rightWidth, rightHeight);
//   sum += texel(leftBuffer, [leftCoords[0], leftCoords[1] + 1], leftWidth, leftHeight) 
//       * texel(rightBuffer, [rightCoords[0], rightCoords[1] + 1], rightWidth, rightHeight);
//   sum += texel(leftBuffer, [leftCoords[0] + 1, leftCoords[1] + 1], leftWidth, leftHeight) 
//       * texel(rightBuffer, [rightCoords[0] + 1, rightCoords[1] + 1], rightWidth, rightHeight);
//   return sum;
// }
function simTileDotProd(leftBuffer, leftWidth, leftHeight, rightBuffer, rightWidth, rightHeight, leftIndex, rightIndex, sharedDim, tileLength, tileIndex) {
  const lcoords = [tileIndex * tileLength, leftIndex * tileLength];
  const rcoords = [tileIndex * tileLength, rightIndex * tileLength];
  let sum = 0.0;
  for(let j =0; j < tileLength; ++j) {
    for(let i =0; i < tileLength; ++i) {
      sum += texel(leftBuffer, [lcoords[0]+i, lcoords[1]+j], leftWidth, leftHeight) * 
             texel(rightBuffer, [rcoords[0]+i, rcoords[1]+j], rightWidth, rightHeight);
    }
  }
  return sum;
}
function simBandDotProd(leftBuffer, leftWidth, leftHeight, rightBuffer, rightWidth, rightHeight, sharedDim, tileLength, leftIndex, rightIndex) {
  if(tileLength === 0) {
    throw 'tileLength  was zero';
  }
  const tileCount = Math.floor(sharedDim / (tileLength*tileLength));
  let sum = 0.0;
  for(let k = 0; k < tileCount; ++k) {
    sum += simTileDotProd(leftBuffer, leftWidth, leftHeight, rightBuffer, rightWidth, rightHeight, leftIndex, rightIndex, sharedDim, tileLength, k);
  }
  return sum;
}
function simTiledDotProduct(leftBuffer, leftWidth, leftHeight, rightBuffer, rightWidth, rightHeight, sharedDim, tileLength, width, height) {
  const newBuffer = new Float32Array(width*height);
  for(let j = 0; j < height; ++j) {
    for(let i = 0; i < width; ++i) {
      const sum = simBandDotProd(leftBuffer, leftWidth, leftHeight, rightBuffer, rightWidth, rightHeight, sharedDim, tileLength, j, i);
      const newOffset = j*width + i;
      newBuffer[newOffset] = sum;
    }
  }
  return newBuffer;
}
function simTiledCoordsToOriginal(coords, width, height, tileLength, originalWidth, originalHeight) {
  const tl = tileLength;
  const ts = tl*tl; // Tile Size
  const xy = coords; // we dont' need this since there is no normalization: [coords[0] * width, coords[1]*height];
  const tileRows = Math.floor(xy[1] / tl);
  const tileColumns = Math.floor(xy[0] / tl);
  const ty = xy[1] - tileRows * tl;
  const tx = xy[0] - tileColumns * tl;
  const tilePerRow = Math.floor(width / tl);
  const ti = tileRows * tilePerRow + tileColumns; // Tile Index
  const offset = ti * ts + ty * tl + tx;

  const originY = Math.floor(offset / originalWidth);
  const originX = offset - originY * originalWidth;
  return [originX, originY];
}

function simConvertOriginalToTiled(buffer, width, height, tileLength) {
  const tiledWidth = Math.ceil(width / tileLength);
  const tiledHeight = height * tileLength;
  const newBuffer = new Float32Array(buffer.length);

  for(let j = 0; j < tiledHeight; ++j) {
    for(let i = 0; i < tiledWidth; ++i) {
      const origCoords = simTiledCoordsToOriginal([i, j], tiledWidth, tiledHeight, tileLength, width, height);
      const newOffset = j*tiledWidth + i;
      const originalOffset = origCoords[1] * width + origCoords[0];
      newBuffer[newOffset] = buffer[originalOffset];
    }
  }
  return newBuffer;
}

// CPU Equivalent for result comparison only
// always stitch on the last dim unlike matmul
function cpuDotProd(a, shapeA, b, shapeB, c) {
  for (let i = 0; i < shapeA[0]; i++) {
    const aoffset = i * shapeA[1];
    for (let j = 0; j < shapeB[0]; j++) {
      const boffset = j * shapeB[1];
      let sum = 0;
      for (let k = 0; k < shapeA[1]; k++) {
        sum += a[aoffset + k] * b[boffset +  k];
      }
      c[i*shapeB[0]+j] = sum;
    }
  }
}
function createRandomArray(size) {
  return new Float32Array(Array.from({length: size}, (v,k) => k % 10));
}