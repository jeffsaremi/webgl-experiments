async function runMatMulSharedAlongY(gl, texA, texB, width, height, sharedDim, texC) {
  const fragmentShaderSource = `#version 300 es
    precision highp float;
    in vec2 TexCoord;
    out vec4 TexelValue;
    // Texture samplers
    uniform sampler2D A;
    uniform sampler2D B;

    void main()
    {
      float value = 0.0;
      int x = int(TexCoord.s * ${width}.0); // rescale
      int y = int(TexCoord.t * ${height}.0); // rescale
      // loop over the shared dim
      for(int k=0; k < ${sharedDim}; ++k) {
        float a = texelFetch(A, ivec2(y, k), 0).r;
        float b = texelFetch(B, ivec2(x, k), 0).r;
        value += a * b;
      }
      TexelValue = vec4(value);
    }`;

const program = createProgram(gl, getDefaultVertexShader(gl),
compileShader(gl, fragmentShaderSource, gl.FRAGMENT_SHADER));
  const handleA = gl.getUniformLocation(program, 'A');
  const handleB = gl.getUniformLocation(program, 'B');

  gl.useProgram(program);

  attachOutputTexture(gl, texC);
  gl.viewport(0, 0, width, height);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.uniform1i(handleA, 0);

  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  gl.uniform1i(handleB, 1);

  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  await waitForSync(gl);
};

// CPU Equivalent for result comparison only
function cpuMatMulSharedAlongY(a, shapeA, b, shapeB, c) {
  let offset = 0;
  for (let i = 0; i < shapeA[1]; i++) {
    for (let j = 0; j < shapeB[1]; j++) {
      let sum = 0;
      for (let k = 0; k < shapeA[0]; k++) {
        sum += a[k*shapeA[1] + i] * b[k*shapeB[1] + j];
      }
      c[offset] = sum;
      offset++;
    }
  }
}

function getTestData() {
  return [
    { a:[8,3], b:[8,4]},
    { a:[64,56*56], b:[64,64]},
    { a:[64,56*56], b:[64,256]},
    { a:[256,7*7], b:[256,64]},
    { a:[512,56*56], b:[512,256]},
    { a:[768,28*28], b:[768,128]},
    { a:[2304,28*28], b:[2304,128]},
    { a:[1024,14*14], b:[1024,256]},
    { a:[2048,7*7], b:[2048,512]}
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
    const sharedDim = testData.a[0];
    const shapeA = testData.a;
    const shapeB = testData.b;
    console.info(`Running matmul-shared-alongY for [${shapeA.toString()}]-[${shapeB.toString()}]`);
    // output texture dimensions
    const width = shapeB[1];
    const height = shapeA[1];

    const a = createRandomArray(shapeA[0] * shapeA[1]);
    const texA = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeA[1], shapeA[0], a);
    //debugPrintTexture(gl, texA, shapeA[1], shapeA[0], gl.RED, gl.FLOAT);
    const b = createRandomArray(shapeB[0] * shapeB[1]);
    const texB = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, shapeB[1], shapeB[0], b);
    //debugPrintTexture(gl, texB, shapeB[1], shapeB[0], gl.RED, gl.FLOAT);

    const c = new Float32Array(width * height);
    const texC = createTexture(gl, gl.R32F, gl.RED, gl.FLOAT, width, height, null);

    console.time('matmul');
    await runMatMulSharedAlongY(gl, texA, texB, width, height, sharedDim, texC);
    console.timeEnd('matmul');
    console.time('readpixels');
    readOutput(gl, width, height, gl.RED, gl.FLOAT, c);
    console.timeEnd('readpixels');
    const expected = new Float32Array(width * height);
    cpuMatMulSharedAlongY(a, shapeA, b, shapeB, expected);
    if(!compareOutputs(c, expected, 0.1)) {
      console.error('Expected and Actual did not match');
      console.log(c);
      console.log(expected);
    } else {
      console.info('Actual and expected matched!');
    }
    gl.deleteTexture(texA);
    gl.deleteTexture(texB);
    gl.deleteTexture(texC);
  }
}

main();
