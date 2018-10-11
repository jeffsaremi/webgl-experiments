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

function createConvDotProgram(gl, im2colTD, kTD, bTD, outputTD) {
  const rank = outputTD.dims.length;

  const initValue = (!bTD) ? '0.0' : '_B(b)';
  
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    uniform sampler2D K;
    uniform sampler2D Im2Col;
    ${(bTD) ? `uniform sampler2D B;` : ``}
    ${getGlslOffsetToCoords()}
    ${getGlslAccessor('Im2Col', im2colTD)}
    ${bTD ? getGlslAccessor('B', bTD) : ''}
    ${glslCoordsToOutputIndices(outputTD)}

    float process(int indices[${rank}]) {
      int b[1];
      b[0] = indices[1];
      int im2col[${im2colTD.dims.length}];
      im2col[0] = indices[0];
      im2col[1] = indices[2];
      im2col[2] = indices[3];
      float sum = ${initValue};
      for (int i = 0; i < ${Math.floor(im2colTD.dims[3]/4)*4}; i += 4) {
        vec4 a;
        for(int j = 0; j < 4; ++j) {
          im2col[3] = i+j;
          a[j] = _Im2Col(im2col);
        }
        vec4 b;
        for(int j = 0; j < 4; ++j) {
          int kernelOffset = indices[1] * ${kTD.strides[0]} + i + j;
          ivec2 kernelCoords = offsetToCoords(kernelOffset, ${kTD.width});
          b[j] = texelFetch(K, kernelCoords, 0).r;
        }
        sum = sum + dot(a,b);
      }
      for(int i = ${Math.floor(im2colTD.dims[3]/4)*4}; i < ${im2colTD.dims[3]}; ++i) {
        int kernelOffset = indices[1] * ${kTD.strides[0]} + i;
        ivec2 kernelCoords = offsetToCoords(kernelOffset, ${kTD.width});
        im2col[3] = i;
        sum = sum + _Im2Col(im2col) * texelFetch(K, kernelCoords, 0).r;
      }
      return sum;
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
async function runConvDot(gl, im2colTD, kTD, bTD, outputTD) {
  const convKey = `conv-dot-${im2colTD.dims.toString()}-${kTD.dims.toString()}-${bTD===null}`;
  let program = getProgram(convKey);
  if(!program) {
    program = createConvDotProgram(gl, im2colTD, kTD, bTD, outputTD);
    cacheProgram(convKey, program);
  }
  const width = outputTD.width;
  const height= outputTD.height;
  gl.useProgram(program);

  attachOutputTexture(gl, outputTD.texture);
  gl.viewport(0, 0, width, height);

  bindInputTexture(gl, program, kTD.texture, 'K', 0);
  bindInputTexture(gl, program, im2colTD.texture, 'Im2Col', 1);
  if(bTD) {
    bindInputTexture(gl, program, bTD.texture, 'B', 2);
  }
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};
function calcIm2ColDims(inputShape, kernelShape, outputShape) {
  return [outputShape[0], outputShape[2], outputShape[3], inputShape[1]*kernelShape[2]*kernelShape[3]];
}
function padInput(x, pads, padValue=0) {
  const [batchSize, inputChannels, inputRows, inputCols] = x.shape;
  const [paddingRowBefore, paddingColBefore, paddingRowAfter, paddingColAfter] = pads;
  const newRows = inputRows + paddingRowBefore + paddingRowAfter;
  const newCols = inputCols + paddingColBefore + paddingColAfter;
  const padded = ndarray(
      new Float32Array(batchSize * newRows * newCols * inputChannels), [batchSize, inputChannels, newRows, newCols]);
  if (padValue !== 0) {
    nd_ops.assigns(padded, padValue);
  }
  nd_ops.assign(
      padded.hi(batchSize, inputChannels, inputRows + paddingRowBefore, inputCols + paddingColBefore)
          .lo(0, 0, paddingRowBefore, paddingColBefore),
      x);
  return padded;
}
function cpuIm2Col(paddedX, kernelShape, outputShape, dilations, strides) {
  const [batchSize, inputChannels, inputRows, inputCols] = paddedX.shape;
  const nbRow = kernelShape[2];
  const nbCol = kernelShape[3];
  const outputRows = outputShape[2];
  const outputCols = outputShape[3];
  const nbPatches = outputRows * outputCols;
  const patchLen = nbRow * nbCol * inputChannels;

  // effective shape after filter dilation
  const nbRowDilated = nbRow + (nbRow - 1) * (dilations[0] - 1);
  const nbColDilated = nbCol + (nbCol - 1) * (dilations[1] - 1);

  const im2col = ndarray(new Float32Array(batchSize * nbPatches * patchLen), [batchSize, outputRows, outputCols, patchLen]);
  const patch = ndarray(new Float32Array(nbRow * nbCol * inputChannels), [inputChannels, nbRow, nbCol]);

  let offset = 0;
  for (let i = 0, limit = inputRows - nbRowDilated; i <= limit; i += strides[0]) {
    for (let j = 0, limit = inputCols - nbColDilated; j <= limit; j += strides[1]) {
      nd_ops.assign(
        patch,
        paddedX
              .hi(batchSize, inputChannels, i + nbRowDilated, j + nbColDilated)  // lowerright corner
              .lo(0, 0, i, j)                                                    // upperleft corner
              .step(1, dilations[0], dilations[1])                               // step by 1 channel
              .pick(0));
      im2col.data.set(patch.data, offset);
      offset += patchLen;
    }
  }
  return im2col;
}
function debugPrintIm2ColData(data, shape) {
  for(let i =0; i < shape[0]*shape[1]*shape[2]; ++i ) {
    const start = i * shape[3];
    const end = start + shape[3];
    console.log(`[${data.slice(start,end).toString()}]`);
  }
}
function debugPrintIm2ColTexture(gl, im2colTD) {
  createFrameBuffer(gl);
  attachOutputTexture(gl, im2colTD.texture);
  const buffer = new Float32Array(im2colTD.width*im2colTD.height);
  readOutput(gl, im2colTD.width, im2colTD.height, gl.RED, gl.FLOAT, buffer);
  debugPrintIm2ColData(buffer, im2colTD.dims);
}
function getValue(indices, data) {
  let offset = 0;      
  offset += indices[3] * 1;      
  offset += indices[2] * 7;      
  offset += indices[1] * 49;      
  offset += indices[0] * 147;
  return data[offset]
}
function process(indices, input, inputShape, kernelShape, dilations, strides, pads) {
  const C1 = inputShape[1];
  const H1 = inputShape[2];
  const W1 = inputShape[3];
  const KH = kernelShape[2];
  const KW = kernelShape[3];
  const DH = dilations[0];
  const DW = dilations[1];
  const SH = strides[0];
  const SW = strides[1];
  const PH = pads[0];
  const PW = pads[1];

  const n  = indices[0];
  const h2 = indices[1];
  const w2 = indices[2];
  const khkwc1 = indices[3];
  const c1 = Math.floor(khkwc1 / (KH * KW));
  const kh = Math.floor((khkwc1 - c1 * (KH * KW)) / KW);
  const kw = khkwc1 - c1 * (KH * KW) - kh * KW;

  const h1 = h2 * SH - PH + kh * DH;
  const w1 = w2 * SW - PW + kw * DW;
  const x = [];
  x[0] = n;
  x[1] = c1;
  x[2] = h1;
  x[3] = w1;
  const v = (h1 < 0 || h1 >= H1 || w1 < 0 || w1 >= W1) ? 0.0 : getValue(x, input);
  return v;
}
function SimIm2Col(input, inputShape, kernelShape, dilations, strides, pads, outputShape, outputBuffer) {
  const outputStrides = computeStrides(outputShape);
  for(let i=0; i < outputShape[0]; ++i) {
    for(let j=0; j < outputShape[1]; ++j) {
      for(let k=0; k < outputShape[2]; ++k) {
        for(let l=0; l < outputShape[3]; ++l) {
          offset = outputStrides[0] * i + outputStrides[1] * j + outputStrides[2] * k + l;
          outputBuffer[offset] = process([i,j,k,l], input, inputShape, kernelShape, dilations, strides, pads);
        }
      }
    }
  }
}
async function convIm2Col(input, inputShape, kernel, kernelShape, bias, autoPad, dilations, group, pads, strides) {
    group = (group <= 0) ? 1 : group;
    const outputShape = calcOutputShape(inputShape, kernelShape, autoPad, dilations, pads, strides);
    const xTD = createTextureData(gl, 1, gl.FLOAT, inputShape, input);
    const kTD = createTextureData(gl, 1, gl.FLOAT, kernelShape, kernel);
    const bTD = (bias) ? createTextureData(gl, 1, gl.FLOAT, [bias.length], bias) : null;
    const outputTD = createTextureData(gl, 1, gl.FLOAT, outputShape, null);
    const im2colDims = calcIm2ColDims(inputShape, kernelShape, outputShape);
    const im2colTD = createTextureData(gl, 1, gl.FLOAT, im2colDims, null);
    const buffer = new Float32Array(outputTD.width * outputTD.height);
    console.time('total-conv');
    console.time('im2col');
    await runConvIm2Col(gl, xTD, kernelShape, dilations, group, pads, strides, im2colTD);
    //console.log('im2col Texture');
    //debugPrintIm2ColTexture(gl, im2colTD);
    console.timeEnd('im2col');
    console.time('dot-product');
    await runConvDot(gl, im2colTD, kTD, bTD, outputTD);
    console.timeEnd('dot-product');
    console.time('read-pixels');
    readOutput(gl, outputTD.width, outputTD.height, gl.RED, gl.FLOAT, buffer);
    console.timeEnd('read-pixels');
    console.timeEnd('total-conv');
    gl.deleteTexture(xTD.texture);
    gl.deleteTexture(kTD.texture);
    gl.deleteTexture(im2colTD.texture);
    gl.deleteTexture(outputTD.texture);
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
  await testMe(convIm2Col, true, 1);
  console.groupEnd();
  console.group('Subsequent times using cached programs')
  testMe(convIm2Col, false, 1);
  console.groupEnd();
}

main();