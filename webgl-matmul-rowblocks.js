function createMatmulRowBlocksProgram(gl, sharedDim) {
  const fragmentShaderSource = `#version 300 es
  precision highp float;
  in vec2 TexCoord;
  out vec4 TexelValue;
  // Texture samplers
  uniform sampler2D A;
  uniform sampler2D B;
  uniform int width;
  uniform int height;
  uniform int hOffset;

  void main()
  {
    ivec2 xy = ivec2(TexCoord * vec2(width, height)); // rescale
    float sum = 0.0;
    for(int k = 0; k < ${sharedDim}; ++k) {
      float a = texelFetch(A, ivec2(k, xy.y+hOffset), 0).r;
      float b = texelFetch(B, ivec2(xy.x, k), 0).r;
      sum += a*b;
    }
    TexelValue = vec4(sum);
  }`;

  return createProgram(gl, getDefaultVertexShader(gl),
  compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
}
async function runMatmulRowblocks(gl, program, texA, texB, width, height, texC, blockSize) {
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');
  const hWidth = gl.getUniformLocation(program, 'width');
  const hHeight = gl.getUniformLocation(program, 'height');
  const hHOffset = gl.getUniformLocation(program, 'hOffset');

  gl.useProgram(program);

  attachOutputTexture(gl, texC);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);

  gl.uniform1i(hWidth, width);
  for(let row = 0; row < height; row+=blockSize) {
    const rowCount = Math.min(blockSize, height-row);
    
    gl.uniform1i(hHeight, rowCount);
    gl.uniform1i(hHOffset, row);
    gl.viewport(0, row, width, rowCount);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  await waitForSync(gl);
};

// CPU Equivalent for result comparison only
function cpuMatMul(a, shapeA, b, shapeB, c) {
  let offset = 0;
  for (let i = 0; i < shapeA[0]; i++) {
    for (let j = 0; j < shapeB[1]; j++) {
      let sum = 0;
      for (let k = 0; k < shapeA[1]; k++) {
        sum += a[i * shapeA[1] + k] * b[k * shapeB[1] +  j];
      }
      c[offset] = sum;
      offset++;
    }
  }
}
function simTexelFetch(array, x, y, width) {
  const offset = y * width + x;
  return array[offset];
}
function simFragShader(a, aw, ah, b, bw, bh, sharedDim, i, j) {
  let sum = 0.0;
  for(let k=0; k < sharedDim; ++k) {
    const aval = simTexelFetch(a, k, j, aw);
    const bval = simTexelFetch(b, i, k, bw);
    sum += aval * bval;
  }
  return sum;
}
function simMatmulRowBlock(a, aw, ah, b, bw, bh, sharedDim, width, height, blockSize) {
  const newBuffer = new Float32Array(width*height);
  
  for(let row = 0; row < height; row+=blockSize) {
    const rowCount = Math.min(blockSize, height-row);
    for(let j = row; j < row+rowCount; ++j) {
      for(let i = 0; i < width; ++i) {
        const sum = simFragShader(a,aw,ah, b, bw, bh, sharedDim, i, j);
        const newOffset = j*width + i;
        newBuffer[newOffset] = sum;
      }
    }
  }
  return newBuffer;
}
function getTestData() {
  return [
    { a:[56*56,64], b:[64,64]},
    { a:[56*56,64], b:[64,256]},
    { a:[7*7,256], b:[256,64]},
    { a:[56*56,512], b:[512,256]},
    { a:[28*28,768], b:[768,128]},
    { a:[28*28,2304], b:[2304,128]},
    { a:[14*14,1024], b:[1024,256]},
    { a:[7*7,2048], b:[2048,512]}
  ];
}

//
// Main
//
async function main() {
  const canvas = createCanvas(1, 1);
  const gl = getContext(canvas);
  setupVBO(gl);
  createFrameBuffer(gl);

  const testDatas = getTestData();
  for(let i = 0; i < testDatas.length; ++i) {
    const testData = testDatas[i];
    const sharedDim = testData.a[1];
    const shapeA = testData.a;
    const shapeB = testData.b;
    console.info(`Running matmul for [${shapeA.toString()}]-[${shapeB.toString()}]`);
    // output texture dimensions
    const width = shapeB[1];
    const height = shapeA[0];
    const expected = new Float32Array(width * height);

    const a = createRandomArray(shapeA[0] * shapeA[1]);
    const b = createRandomArray(shapeB[0] * shapeB[1]);
    const c = new Float32Array(width * height);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);
    const program = createMatmulRowBlocksProgram(gl, sharedDim);

    const blockSizes = [2, 4, 8, 16, 32, 64];
    for(let j = 0; j < blockSizes.length; ++j) {
      const blockSize = blockSizes[j];
      console.log(`matmul-rowblock size: ${blockSize}`);
      console.time('matmul-rowblock');
      await runMatmulRowblocks(gl, program, texA, texB, width, height, texC, blockSize);
      console.timeEnd('matmul-rowblock');
      console.time('readpixels');
      readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
      console.timeEnd('readpixels');
      cpuMatMul(a, shapeA, b, shapeB, expected);
      if(!compareOutputs(c, expected, 0.1)) {
        console.error('Expected and Actual did not match');
        console.log(c);
        console.log(expected);
        // const d = simMatmulRowBlock(a, shapeA[1], shapeA[0], b, shapeB[1], shapeB[0], sharedDim, width, height, blockSize);
        // console.log('simulated result:', d);
      } else {
        console.info('Actual and expected matched!');
      }
    }
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();
