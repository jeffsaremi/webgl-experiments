function createConvProgram(gl, xTextureData, kTextureData, bTextureData, dilations, group, pads, strides, outputTextureData) {
  const rank = xTextureData.dims.length;
  const inputSpatialShape = xTextureData.dims.slice(2);
  const [outChannels, kernelIn] = [kTextureData.dims[0], kTextureData.dims[1]];
  const kernelSpatialShape = kTextureData.dims.slice(2);
  const groupOutChanels = outChannels / group;

  const hasNoPadding = pads.every(v => v === 0);
  const initValueSnippet = (!bTextureData) ? '0.0' : '_B(b)';
  const codeLines = [];
  let indent = '          ';
  codeLines.push('');
  for (let i = 2; i < rank; ++i) {
    codeLines.push(`${indent}x[${i}] = indices[${i}] * ${strides[i - 2]} - ${pads[i - 2]};`);
    if(!hasNoPadding) {codeLines.push(
        `${indent}inpad[${i}] = inpad[${i - 1}] || (x[${i}] < 0 || x[${i}] >= ${inputSpatialShape[i - 2]});`);
    }
    codeLines.push(`${indent}for (int k_${i} = 0; k_${i} < ${kernelSpatialShape[i - 2]}; ++k_${i}) {`);
    codeLines.push(`${indent}    k[${i}] = k_${i};`);
    indent += '    ';
  }
  codeLines.push(`${indent}for(int x_1 = 0; x_1 < ${kernelIn}; ++x_1) {`);
  codeLines.push(`${indent}    x[1] = x_1 + xCStart;`);
  codeLines.push(`${indent}    k[1] = x_1 + xCStart;`);
  if(!hasNoPadding) {codeLines.push(`${indent}    float valX = (inpad[${rank - 1}]) ? 0.0 : _X(x);`);}
  else {
    codeLines.push(`${indent}    float valX = _X(x);`);
  }
  codeLines.push(`${indent}    r += valX * _K(k);`);
  codeLines.push(`${indent}}`);

  for (let i = rank - 1; i >= 2; --i) {
    codeLines.push(`${indent}x[${i}] += ${dilations[i - 2]};`);
    if(!hasNoPadding) {codeLines.push(
        `${indent}inpad[${i}] = inpad[${i - 1}] || (x[${i}] < 0 || x[${i}] >= ${inputSpatialShape[i - 2]});`);
    } 
    indent = indent.substring(4);
    codeLines.push(`${indent}}`);
  }
  const paddingInitLines = (!hasNoPadding) ? `
        bool inpad[${rank}]; //cummulative by levels
        inpad[0] = false;
        inpad[1] = false;
  ` : '';
  const snippetMainLoop = codeLines.join('\n');
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    uniform sampler2D X;
    uniform sampler2D K;
    ${(bTextureData) ? `uniform sampler2D B;` : ``}
    ${getGlslOffsetToCoords()}
    ${getGlslAccessor('X', xTextureData)}
    ${getGlslAccessor('K', kTextureData)}
    ${bTextureData ? getGlslAccessor('B', bTextureData) : ''}
    ${glslCoordsToOutputIndices(outputTextureData)}

    float process(int indices[${rank}]) {
        int x[${rank}];
        int k[${rank}];
        int b[1];
        ${paddingInitLines}

        int batchId = indices[0];
        int yC = indices[1];

        int g = int(yC / ${groupOutChanels});
        int xCStart = g * ${kernelIn};

        b[0] = yC;
        x[0] = batchId;
        k[0] = yC;
        float r = ${initValueSnippet};

        ${snippetMainLoop}

        return r;
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
async function runConv(gl, xTextureData, kTextureData, bTextureData, dilations, group, pads, strides, outputTextureData) {
  const convKey = `conv-${xTextureData.dims.toString()}-${kTextureData.dims.toString()}-${bTextureData===null}-${dilations}-${group}-${pads}-${strides}`;
  let program = getProgram(convKey);
  if(!program) {
    program = createConvProgram(gl, xTextureData, kTextureData, bTextureData, dilations, group, pads, strides, outputTextureData);
    cacheProgram(convKey, program);
  }
  const width = outputTextureData.width;
  const height= outputTextureData.height;
  gl.useProgram(program);

  attachOutputTexture(gl, outputTextureData.texture);
  gl.viewport(0, 0, width, height);

  bindInputTexture(gl, program, xTextureData.texture, 'X', 0);
  bindInputTexture(gl, program, kTextureData.texture, 'K', 1);
  if(bTextureData) {
    bindInputTexture(gl, program, bTextureData.texture, 'B', 2);
  }
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};

async function convDefault(input, inputShape, kernel, kernelShape, bias, autoPad, dilations, group,
  pads, strides) {
    group = (group <= 0) ? 1 : group;
    const outputShape = calcOutputShape(inputShape, kernelShape, autoPad, dilations, pads, strides);
    const xTextureData = createTextureData(gl, 1, gl.FLOAT, inputShape, input);
    const kTextureData = createTextureData(gl, 1, gl.FLOAT, kernelShape, kernel);
    const bTextureData = (bias) ? createTextureData(gl, 1, gl.FLOAT, [bias.length], bias) : null;
    const outputTextureData = createTextureData(gl, 1, gl.FLOAT, outputShape, null);
    const buffer = new Float32Array(outputTextureData.width * outputTextureData.height);
    console.time('total-conv');
    console.time('conv');
    await runConv(gl, xTextureData, kTextureData, bTextureData, dilations, 
      group, pads, strides, outputTextureData);
    console.timeEnd('conv');
    console.time('read-pixels');
    readOutput(gl, outputTextureData.width, outputTextureData.height, gl.RED, gl.FLOAT, buffer);
    console.timeEnd('read-pixels');
    console.timeEnd('total-conv');
    gl.deleteTexture(xTextureData.texture);
    gl.deleteTexture(kTextureData.texture);
    if(bias) { gl.deleteTexture(bTextureData.texture); }
    gl.deleteTexture(outputTextureData.texture);
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
  await testMe(convDefault, true, 1);
  console.groupEnd();
  console.group('Subsequent times using cached programs')
  testMe(convDefault, false, 1);
  console.groupEnd();
}

main();