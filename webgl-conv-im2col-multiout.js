function createConvIm2ColProgram(gl, xTD, outputTD) {
  const rank = outputTD.dims.length;

  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    uniform sampler2D X;
    uniform int C1; // input Channels
    uniform int H1; // input Height
    uniform int W1; // Input Width
    uniform int KH; // Kernel Height
    uniform int KW; // Kernel Width
    uniform int DH; // Dilations Height
    uniform int DW; // Dilations Width
    uniform int SH; // Strides Height
    uniform int SW; // Strides Width
    uniform int PH; // Pads Height
    uniform int PW; // Pads Width
    ${getGlslOffsetToCoords()}
    ${getGlslAccessor('X', xTD)}
    ${glslCoordsToOutputIndices(outputTD)}

    float process(int indices[${rank}]) {
      int n  = indices[0];
      int h2 = indices[1];
      int w2 = indices[2];
      int khkwc1 = indices[3];
      int patchStrides[2] = int[2](KH * KW, KW);
      int c1 = khkwc1 / patchStrides[0];
      int kh = (khkwc1 - c1 * patchStrides[0]) / KW;
      int kw = khkwc1 - c1 * patchStrides[0] - kh * patchStrides[1];
  
      int h1 = h2 * SH - PH + kh * DH;
      int w1 = w2 * SW - PW + kw * DW;
      int x[${xTD.dims.length}] = int[${xTD.dims.length}](n, c1, h1, w1);
      float v = (h1 < 0 || h1 >= H1 || w1 < 0 || w1 >= W1) ? 0.0 : _X(x);
      return v;
    }

    void main() {
      int indices[${rank}];
      toIndices(TexCoord, indices);
      TexelValue = vec4(process(indices));
    }
    `;
    //console.log(fragmentShaderSource);
    const program = createProgram(gl, getDefaultVertexShader(gl),
    compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    return program; 
}
async function runConvIm2Col(gl, xTD, kDims, dilations, group, pads, strides, outputTD) {
  const convKey = `conv-im2col-${xTD.dims.toString()}-${outputTD.dims.toString()}`;
  let program = getProgram(convKey);
  if(!program) {
    program = createConvIm2ColProgram(gl, xTD, outputTD);
    cacheProgram(convKey, program);
  }
  const width = outputTD.width;
  const height= outputTD.height;
  gl.useProgram(program);

  attachOutputTexture(gl, outputTD.texture);
  gl.viewport(0, 0, width, height);

  bindInputTexture(gl, program, xTD.texture, 'X', 0);
  gl.uniform1i(gl.getUniformLocation(program, 'C1'), xTD.dims[1]);
  gl.uniform1i(gl.getUniformLocation(program, 'H1'), xTD.dims[2]);
  gl.uniform1i(gl.getUniformLocation(program, 'W1'), xTD.dims[3]);
  gl.uniform1i(gl.getUniformLocation(program, 'KH'), kDims[2]);
  gl.uniform1i(gl.getUniformLocation(program, 'KW'), kDims[3]);
  gl.uniform1i(gl.getUniformLocation(program, 'DH'), dilations[0]);
  gl.uniform1i(gl.getUniformLocation(program, 'DW'), dilations[1]);
  gl.uniform1i(gl.getUniformLocation(program, 'SH'), strides[0]);
  gl.uniform1i(gl.getUniformLocation(program, 'SW'), strides[1]);
  gl.uniform1i(gl.getUniformLocation(program, 'PH'), pads[0]);
  gl.uniform1i(gl.getUniformLocation(program, 'PW'), pads[1]);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
}

function createConvDotMultiProgram(gl, im2colTD, kTD, bTD, tileTDs, originalDims) {
  const tileCount = tileTDs.length;
  const increment = Math.ceil(originalDims[1]/tileCount);
  const rank = tileTDs[0].dims.length;

  const initValueLines = [];
  const layoutLines = [];
  const summationLines = [];
  const texelValueLines = [];
  for(let i=0; i < tileCount; ++i) {
    layoutLines.push(`layout(location = ${i}) out vec4 TexelValue_${i};`);
    initValueLines.push(`b[0] = indices[1] + ${i*increment};`);
    initValueLines.push(`sums[${i}] = ${(bTD) ? _B(b) : '0.0'};`);
    summationLines.push(`kernelOffset = (indices[1] + ${i*increment}) * ${kTD.strides[0]} + k;`);
    summationLines.push(`kernelCoords = offsetToCoords(kernelOffset, ${kTD.width});`);
    summationLines.push(`sums[${i}] += im2colValue * texelFetch(K, kernelCoords, 0).r;`);
    texelValueLines.push(`TexelValue_${i} = vec4(sums[${i}]);`);
  }
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    uniform sampler2D K;
    uniform sampler2D Im2Col;
    ${(bTD) ? `uniform sampler2D B;` : ``}
    
    ${layoutLines.join('\n')}
    ${getGlslOffsetToCoords()}
    ${getGlslAccessor('Im2Col', im2colTD)}
    ${bTD ? getGlslAccessor('B', bTD) : ''}
    ${glslCoordsToOutputIndices(tileTDs[0])}

    float[${tileCount}] process(int indices[${rank}]) {
      int b[1];
      int im2col[${im2colTD.dims.length}];
      im2col[0] = indices[0];
      im2col[1] = indices[2];
      im2col[2] = indices[3];
      float sums[${tileCount}];
      ${initValueLines.join('\n')};
      int kernelOffset = 0;
      ivec2 kernelCoords;
      float im2colValue;
      for (int k = 0; k < ${im2colTD.dims[3]}; ++k) {
        im2col[3] = k;
        im2colValue = _Im2Col(im2col);
        ${summationLines.join('\n')}    
      }
      return sums;
    }

    void main() {
      int indices[${rank}];
      toIndices(TexCoord, indices);
      float[${tileCount}] sums = process(indices);
      ${texelValueLines.join('\n')}
    }
    `;
    //console.log(fragmentShaderSource);
    const program = createProgram(gl, getDefaultVertexShader(gl),
    compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
    return program; 
}
async function runConvDotMulti(gl, im2colTD, kTD, bTD, tileTDs, originalDims) {
  const tileCount = tileTDs.length;
  const tileWidth = tileTDs[0].width;
  const tileHeight = tileTDs[0].height;
  const convKey = `conv-dot-multi-${im2colTD.dims.toString()}-${kTD.dims.toString()}-${bTD===null}-${tileCount}`;
  let program = getProgram(convKey);
  if(!program) {
    program = createConvDotMultiProgram(gl, im2colTD, kTD, bTD, tileTDs, originalDims);
    cacheProgram(convKey, program);
  }
  gl.useProgram(program);

  gl.viewport(0, 0, tileWidth, tileHeight);
  const drawBuffers = [];
  for (let i = 0; i < tileCount; ++i) {
    gl.bindTexture(gl.TEXTURE_2D, tileTDs[i].texture);
    // attach texture to framebuffer
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i,
                            gl.TEXTURE_2D, tileTDs[i].texture, 0);
    drawBuffers.push(gl.COLOR_ATTACHMENT0 + i);
  }

  gl.drawBuffers(drawBuffers);

  bindInputTexture(gl, program, kTD.texture, 'K', 0);
  bindInputTexture(gl, program, im2colTD.texture, 'Im2Col', 1);
  if(bTD) {
    bindInputTexture(gl, program, bTD.texture, 'B', 2);
  }
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};
function createCombineProgram(gl, tileTDs, outputTD) {
  const increment = tileTDs[0].dims[1];
  const tileCount = tileTDs.length;
  const rank = tileTDs[0].dims.length;

  const valueLines = [];
  for(let i=0; i < tileCount; ++i) {
    valueLines.push(`
    if(i == ${i}) {
      indices[1] -= ${i*increment};
      int offset = indicesToOffset_X(indices);
      TexelValue = texelFetch(Tiles[${i}], offsetToCoords(offset, ${tileTDs[0].width}), 0);
      return;
    }
    `);
  }
  const fragmentShaderSource = `#version 300 es
            precision highp float;
            in vec2 TexCoord;
            out vec4 TexelValue;
            uniform sampler2D Tiles[${tileCount}];
            ${getGlslIndicesToOffset('X', tileTDs[0])}
            ${getGlslOffsetToCoords()}
            ${glslCoordsToOutputIndices(outputTD)} 

            void main()
            {
              int indices[${rank}];
              toIndices(TexCoord, indices);
              int stichDim = indices[1];
              int i = stichDim / ${increment};
              ${valueLines.join('\n')}
            }`;

  const program = createProgram(gl, getDefaultVertexShader(gl),
      compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  return program;
}
async function runCombine(gl, tileTDs, outputTD) {
  const tileCount = tileTDs.length;
  const programKey = `conv-combine-${tileTDs[0].dims.toString()}-${outputTD.dims.toString()}`;
  let program = getProgram(programKey);
  if(!program) {
    program = createCombineProgram(gl, tileTDs, outputTD);
    cacheProgram(programKey, program);
  }
  gl.useProgram(program);
  gl.viewport(0, 0, outputTD.width, outputTD.height);
  
  createFrameBuffer(gl);
  attachOutputTexture(gl, outputTD.texture);
  const uniformIndices = [];
  for (let i = 0; i < tileCount; ++i) {
    gl.activeTexture(gl.TEXTURE0 + i);
    gl.bindTexture(gl.TEXTURE_2D, tileTDs[i].texture);
    uniformIndices.push(i);
  }
  gl.uniform1iv(gl.getUniformLocation(program, 'Tiles[0]'), uniformIndices);
  //checkError(gl); // make sure we have bound all input/output properly
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
}
function calcIm2ColDims(inputShape, kernelShape, outputShape, channels= 1) {
  return [outputShape[0], outputShape[2], outputShape[3], inputShape[1]*Math.ceil(kernelShape[2]*kernelShape[3] / channels)];
}
async function convIm2ColMultiOut(input, inputShape, kernel, kernelShape, bias, autoPad, dilations, group, pads, strides) {
    group = (group <= 0) ? 1 : group;
    const tileCount = 4;
    const outputShape = calcOutputShape(inputShape, kernelShape, autoPad, dilations, pads, strides);
    const xTD = createTextureData(gl, 1, gl.FLOAT, inputShape, input);
    const kTD = createTextureData(gl, 1, gl.FLOAT, kernelShape, kernel);
    const bTD = (bias) ? createTextureData(gl, 1, gl.FLOAT, [bias.length], bias) : null;
    const outputTD = createTextureData(gl, 1, gl.FLOAT, outputShape, null);
    const im2colDims = calcIm2ColDims(inputShape, kernelShape, outputShape);
    const im2colTD = createTextureData(gl, 1, gl.FLOAT, im2colDims, null);
    const tileTDs = [];
    const tileShape = [outputShape[0], Math.ceil(outputShape[1]/tileCount), outputShape[2], outputShape[3]];
    for(let i = 0; i < tileCount; ++i) {
      tileTDs[i] = createTextureData(gl, 1, gl.FLOAT, tileShape, null);
    }
    const tileBuffer = new Float32Array(tileTDs[0].width * tileTDs[0].height);
    const buffer = new Float32Array(outputTD.width * outputTD.height);
    // create multiple textures along the dimension M of the output
    console.time('total-conv');
    console.time('im2col');
    await runConvIm2Col(gl, xTD, kernelShape, dilations, group, pads, strides, im2colTD);
    //console.log('im2col Texture');
    //debugPrintIm2ColTexture(gl, im2colTD);
    console.timeEnd('im2col');
    console.time('dot-product-multi');
    await runConvDotMulti(gl, im2colTD, kTD, bTD, tileTDs, outputShape);
    console.timeEnd('dot-product-multi');
    console.time('combine');
    await runCombine(gl, tileTDs, outputTD);
    console.timeEnd('combine');
    console.time('read-pixels');
    readOutput(gl, outputTD.width, outputTD.height, gl.RED, gl.FLOAT, buffer);
    console.timeEnd('read-pixels');
    console.timeEnd('total-conv');
    gl.deleteTexture(xTD.texture);
    gl.deleteTexture(kTD.texture);
    gl.deleteTexture(im2colTD.texture);
    gl.deleteTexture(outputTD.texture);
    for(let i = 0; i < tileCount; ++i) {
      gl.deleteTexture(tileTDs[i].texture);
    }
    if(bias) { gl.deleteTexture(bTD.texture); }
    return buffer;
}

//
// Main
//
const canvas = createCanvas(1, 1);
const gl = getContext(canvas);
setupVBO(gl);
createFrameBuffer(gl);

async function main() {
  console.group('1st time with CPU validation');
  await testMe(convIm2ColMultiOut, true, 1);
  console.groupEnd();
  console.group('Subsequent times using cached programs')
  testMe(convIm2ColMultiOut, false, 1);
  console.groupEnd();
}

main();