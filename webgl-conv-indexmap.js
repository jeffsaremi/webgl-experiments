function createConvIndexMapProgram(gl, xTD, kTD, bTD, dilations, group, pads, strides, indexMapTD, outputTD) {
  const rank = outputTD.dims.length;

  const initValue = (!bTD) ? '0.0' : '_B(b)';
  
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    uniform sampler2D X;
    uniform sampler2D K;
    uniform sampler2D IndexMap;
    ${(bTD) ? `uniform sampler2D B;` : ``}
    ${getGlslOffsetToCoords()}
    ${getGlslAccessor('X', xTD)}
    ${getGlslAccessor('K', kTD)}
    ${getGlslAccessor('IndexMap', indexMapTD)}
    ${bTD ? getGlslAccessor('B', bTD) : ''}
    ${glslCoordsToOutputIndices(outputTD)}

    float process(int indices[${rank}]) {
      int b[1];
      b[0] = indices[1];
      float sum = ${initValue};
      int indexMapIndices[2];
      int indexMapY = indices[${rank - 2}] * ${outputTD.strides[rank - 2]} + indices[${rank - 1}];
      for (int i = 0; i < ${indexMapTD.width}; ++i) {
        int inputOffset = int(texelFetch(IndexMap, ivec2(i, indexMapY), 0).r);
        if (inputOffset != -1) {
          ivec2 inputCoords = offsetToCoords(inputOffset, ${xTD.width});
          // int kernelChannel = i / ${kTD.strides[1]};
          // int xyOffset = i - (kernelChannel * ${kTD.strides[1]});
          // kernelChannel /= ${group};
          // int kernelOffset = indices[1] * ${kTD.strides[0]} + kernelChannel * ${kTD.strides[1]} + xyOffset;
          int kernelOffset = indices[1] * ${kTD.strides[0]} + i;
          ivec2 kernelCoords = offsetToCoords(kernelOffset, ${kTD.width});
          sum += texelFetch(X, inputCoords, 0).r * texelFetch(K, kernelCoords, 0).r;
        }
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
async function runConvIndexMap(gl, xTD, kTD, bTD, dilations, group, pads, strides, indexMapTD, outputTD) {
  const convKey = `conv-${xTD.dims.toString()}-${kTD.dims.toString()}-${bTD===null}-${dilations}-${group}-${pads}-${strides}`;
  let program = getProgram(convKey);
  if(!program) {
    program = createConvIndexMapProgram(gl, xTD, kTD, bTD, dilations, group, pads, strides, indexMapTD, outputTD);
    cacheProgram(convKey, program);
  }
  const width = outputTD.width;
  const height= outputTD.height;
  gl.useProgram(program);

  attachOutputTexture(gl, outputTD.texture);
  gl.viewport(0, 0, width, height);

  bindInputTexture(gl, program, xTD.texture, 'X', 0);
  bindInputTexture(gl, program, kTD.texture, 'K', 1);
  bindInputTexture(gl, program, indexMapTD.texture, 'IndexMap', 2);
  if(bTD) {
    bindInputTexture(gl, program, bTD.texture, 'B', 3);
  }
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};
function padInputOffsets(x, pads, padValue=0) {
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
function createArrayOfOffsets(size) {
  return new Float32Array([...Array(size).keys()]);
}
function createIndexMap(paddedIndices, kernelShape, outputShape, dilations, strides) {
  const [batchSize, inputChannels, inputRows, inputCols] = paddedIndices.shape;
  const nbRow = kernelShape[2];
  const nbCol = kernelShape[3];
  const outputRows = outputShape[2];
  const outputCols = outputShape[3];
  const nbPatches = outputRows * outputCols;
  const patchLen = nbRow * nbCol * inputChannels;

  // effective shape after filter dilation
  const nbRowDilated = nbRow + (nbRow - 1) * (dilations[0] - 1);
  const nbColDilated = nbCol + (nbCol - 1) * (dilations[1] - 1);

  const indexMap = ndarray(new Float32Array(nbPatches * patchLen), [nbPatches, patchLen]);
  const indicesPatch = ndarray(new Float32Array(nbRow * nbCol * inputChannels), [inputChannels, nbRow, nbCol]);

  let offset = 0;
  for (let i = 0, limit = inputRows - nbRowDilated; i <= limit; i += strides[0]) {
    for (let j = 0, limit = inputCols - nbColDilated; j <= limit; j += strides[1]) {
      nd_ops.assign(
          indicesPatch,
          paddedIndices
              .hi(batchSize, inputChannels, i + nbRowDilated, j + nbColDilated)  // lowerright corner
              .lo(0, 0, i, j)                                                    // upperleft corner
              .step(1, dilations[0], dilations[1])                               // step by 1 channel
              .pick(0));
      indexMap.data.set(indicesPatch.data, offset);
      offset += patchLen;
    }
  }
  return indexMap;
}
function createIndexMapTD(inputShape, kernelShape, outputShape, dilations, pads, strides) {
  const inputSize = inputShape.reduce((a, b) => a * b);
  const inputOffsets = ndarray(createArrayOfOffsets(inputSize), inputShape);
  const paddedOffsets = padInputOffsets(inputOffsets, pads, -1);
  const indexMap = createIndexMap(paddedOffsets, kernelShape, outputShape, dilations, strides);
  return createTextureData(gl, 1, gl.FLOAT, indexMap.shape, indexMap.data);
}
async function convIndexMap(input, inputShape, kernel, kernelShape, bias, autoPad, dilations, group,
  pads, strides) {
    group = (group <= 0) ? 1 : group;
    const outputShape = calcOutputShape(inputShape, kernelShape, autoPad, dilations, pads, strides);
    const xTD = createTextureData(gl, 1, gl.FLOAT, inputShape, input);
    const kTD = createTextureData(gl, 1, gl.FLOAT, kernelShape, kernel);
    //debugPrintTexture(gl, kTD.texture, kTD.width, kTD.height, gl.RED, gl.FLOAT);
    const bTD = (bias) ? createTextureData(gl, 1, gl.FLOAT, [bias.length], bias) : null;
    const outputTD = createTextureData(gl, 1, gl.FLOAT, outputShape, null);
    const indexMapTD = createIndexMapTD(xTD.dims, kTD.dims, outputShape, dilations, pads, strides);
    //console.log(`indexMapTD shape=${indexMapTD.width}-${indexMapTD.height}`)
    //debugPrintTexture(gl, indexMapTD.texture, indexMapTD.width, indexMapTD.height, gl.RED, gl.FLOAT);
    const buffer = new Float32Array(outputTD.width * outputTD.height);
    console.time('total-conv');
    console.time('conv');
    await runConvIndexMap(gl, xTD, kTD, bTD, dilations, group, pads, strides, indexMapTD, outputTD);
    console.timeEnd('conv');
    console.time('read-pixels');
    readOutput(gl, outputTD.width, outputTD.height, gl.RED, gl.FLOAT, buffer);
    console.timeEnd('read-pixels');
    console.timeEnd('total-conv');
    gl.deleteTexture(xTD.texture);
    gl.deleteTexture(kTD.texture);
    gl.deleteTexture(indexMapTD.texture);
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
  await testMe(convIndexMap, true, 1);
  console.groupEnd();
  console.group('Subsequent times using cached programs')
  testMe(convIndexMap, false, 1);
  console.groupEnd();
}

main();